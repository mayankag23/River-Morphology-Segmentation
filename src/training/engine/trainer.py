"""
Core training loop for the Training Engine Framework (Module 14).

Trainer owns the epoch and batch loops. It calls callbacks at precise hook
points and communicates with CheckpointManager, MetricsAccumulator, and
SchedulerFactory through a shared context dict.

Responsibilities
----------------
- Epoch loop (train + validation)
- Batch loop with gradient accumulation
- AMP forward/backward with GradScaler
- Gradient norm clipping
- Scheduler stepping
- Callback dispatch
- Early stopping signal detection
- Validation of input tensors, logits, losses, and gradients

Does NOT:
- Build models, optimizers, schedulers, or loss functions
- Save checkpoints directly
- Compute IoU/Dice/Precision/Recall
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.engine.callbacks import CallbackList
from src.training.engine.contracts import EpochResult, TrainingConfig
from src.training.engine.metrics import MetricsAccumulator
from src.training.engine.scheduler import SchedulerFactory

__all__ = ["Trainer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class Trainer:
    """
    Execute the training and validation loops.

    Args:
        config:
            TrainingConfig controlling loop behaviour.

        model:
            torch.nn.Module segmentation model.

        optimizer:
            Configured torch optimizer.

        scheduler:
            Configured learning-rate scheduler, or None.

        loss_fn:
            Segmentation loss function.

        callbacks:
            CallbackList executed at training hook points.

        device:
            torch.device used for model execution.

        scaler:
            AMP GradScaler, or None when AMP is disabled.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        loss_fn: Any,
        callbacks: CallbackList,
        device: Any,
        scaler: Any = None,
    ) -> None:
        self._config = config
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_fn = loss_fn
        self._callbacks = callbacks
        self._device = device
        self._scaler = scaler

        self._logger: logging.Logger = logging.getLogger(__name__)

    def run(
        self,
        train_loader: Any,
        val_loader: Any | None,
        start_epoch: int,
        context: dict,
    ) -> list[EpochResult]:
        """
        Execute training from start_epoch through config.epochs.

        Args:
            train_loader:
                DataLoader for the training split.

            val_loader:
                DataLoader for the validation split. None disables validation.

            start_epoch:
                First epoch to execute. Epoch numbering is one-based.

            context:
                Mutable state shared with callbacks.

        Returns:
            One EpochResult for every completed epoch.
        """

        results: list[EpochResult] = []

        self._callbacks.on_train_begin(context)

        for epoch in range(start_epoch, self._config.epochs + 1):

            # Prevent a stale best-epoch flag from the previous epoch.
            context["is_best_epoch"] = False

            self._callbacks.on_epoch_begin(epoch, context)

            # ----------------------------------------------------------
            # Training phase
            # ----------------------------------------------------------

            train_metrics = self._run_phase(
                loader=train_loader,
                epoch=epoch,
                is_train=True,
                context=context,
            )

            # ----------------------------------------------------------
            # Validation phase
            # ----------------------------------------------------------

            val_loss = 0.0

            if val_loader is not None:

                val_metrics = self._run_phase(
                    loader=val_loader,
                    epoch=epoch,
                    is_train=False,
                    context=context,
                )

                val_loss = float(val_metrics["loss"])

            # ----------------------------------------------------------
            # Learning-rate scheduler
            # ----------------------------------------------------------

            current_lr = MetricsAccumulator.get_lr(
                self._optimizer
            )

            SchedulerFactory.step(
                self._scheduler,
                self._config.scheduler,
                val_loss,
            )

            # ----------------------------------------------------------
            # Epoch result
            # ----------------------------------------------------------

            result = EpochResult(
                epoch=epoch,
                train_loss=float(train_metrics["loss"]),
                val_loss=val_loss,
                lr=current_lr,
                epoch_time=float(train_metrics["epoch_time"]),
                is_best=False,
            )

            results.append(result)

            context["last_epoch_result"] = result

            # Checkpointing and early-stopping callbacks receive the
            # completed train and validation losses here.
            self._callbacks.on_epoch_end(
                epoch,
                result,
                context,
            )

            if context.get("stop_training", False):

                self._logger.info(
                    "Trainer: early stopping at epoch %d.",
                    epoch,
                )

                break

        self._callbacks.on_train_end(context)

        return results

    # ==================================================================
    # Phase execution
    # ==================================================================

    def _run_phase(
        self,
        loader: Any,
        epoch: int,
        is_train: bool,
        context: dict,
    ) -> dict[str, float]:
        """
        Run one complete training or validation phase.

        Performs checks for:

        - NaN/Inf image values
        - invalid image dimensions
        - invalid mask dimensions
        - NaN/Inf logits
        - NaN/Inf loss
        - NaN/Inf gradients
        """

        import torch

        phase = "train" if is_train else "validation"

        if loader is None:
            raise RuntimeError(
                f"Epoch {epoch}: {phase} DataLoader is None."
            )

        try:
            num_batches = len(loader)
        except TypeError:
            num_batches = None

        if num_batches == 0:
            raise RuntimeError(
                f"Epoch {epoch}: {phase} DataLoader contains no batches."
            )

        if is_train:
            self._model.train()
        else:
            self._model.eval()

        accumulator = MetricsAccumulator()
        accumulator.start()

        accumulation_steps = (
            max(1, int(self._config.accumulation_steps))
            if is_train
            else 1
        )

        clip_value = float(
            self._config.grad_clip_value
        )

        use_amp = (
            self._scaler is not None
            and is_train
        )

        if is_train:
            self._optimizer.zero_grad(
                set_to_none=True
            )

        context_manager = (
            _null_context()
            if is_train
            else torch.no_grad()
        )

        with context_manager:

            for batch_idx, batch in enumerate(loader):

                # ------------------------------------------------------
                # Load batch
                # ------------------------------------------------------

                images, masks = self._unpack_batch(
                    batch=batch,
                    device=self._device,
                )

                self._validate_batch(
                    images=images,
                    masks=masks,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    phase=phase,
                )

                if is_train:
                    self._callbacks.on_batch_begin(
                        batch_idx,
                        context,
                    )

                # ------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------

                if use_amp:

                    with torch.autocast(
                        device_type=self._device.type,
                        dtype=torch.float16,
                    ):

                        logits = self._model(images)

                        self._validate_logits(
                            logits=logits,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            phase=phase,
                        )

                        raw_loss = self._loss_fn(
                            logits,
                            masks,
                        )

                        self._validate_loss(
                            loss=raw_loss,
                            masks=masks,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            phase=phase,
                        )

                        scaled_loss = (
                            raw_loss
                            / accumulation_steps
                        )

                    # --------------------------------------------------
                    # AMP backward pass
                    # --------------------------------------------------

                    self._scaler.scale(
                        scaled_loss
                    ).backward()

                    should_step = self._should_step(
                        batch_idx=batch_idx,
                        loader_length=num_batches,
                        accumulation_steps=accumulation_steps,
                    )

                    if should_step:

                        self._scaler.unscale_(
                            self._optimizer
                        )

                        self._validate_gradients(
                            epoch=epoch,
                            batch_idx=batch_idx,
                        )

                        if clip_value > 0.0:

                            torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                clip_value,
                            )

                        self._scaler.step(
                            self._optimizer
                        )

                        self._scaler.update()

                        self._optimizer.zero_grad(
                            set_to_none=True
                        )

                elif is_train:

                    # --------------------------------------------------
                    # Standard precision training
                    # --------------------------------------------------

                    logits = self._model(images)

                    self._validate_logits(
                        logits=logits,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        phase=phase,
                    )

                    raw_loss = self._loss_fn(
                        logits,
                        masks,
                    )

                    self._validate_loss(
                        loss=raw_loss,
                        masks=masks,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        phase=phase,
                    )

                    scaled_loss = (
                        raw_loss
                        / accumulation_steps
                    )

                    scaled_loss.backward()

                    should_step = self._should_step(
                        batch_idx=batch_idx,
                        loader_length=num_batches,
                        accumulation_steps=accumulation_steps,
                    )

                    if should_step:

                        self._validate_gradients(
                            epoch=epoch,
                            batch_idx=batch_idx,
                        )

                        if clip_value > 0.0:

                            torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                clip_value,
                            )

                        self._optimizer.step()

                        self._optimizer.zero_grad(
                            set_to_none=True
                        )

                else:

                    # --------------------------------------------------
                    # Validation
                    # --------------------------------------------------

                    logits = self._model(images)

                    self._validate_logits(
                        logits=logits,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        phase=phase,
                    )

                    raw_loss = self._loss_fn(
                        logits,
                        masks,
                    )

                    self._validate_loss(
                        loss=raw_loss,
                        masks=masks,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        phase=phase,
                    )

                # ------------------------------------------------------
                # Metrics
                # ------------------------------------------------------

                batch_size = int(
                    images.shape[0]
                )

                loss_scalar = float(
                    raw_loss.detach().cpu().item()
                )

                accumulator.update(
                    loss_scalar,
                    batch_size,
                )

                if is_train:

                    self._callbacks.on_batch_end(
                        batch_idx,
                        loss_scalar,
                        context,
                    )

        metrics = accumulator.compute()

        phase_loss = float(
            metrics["loss"]
        )

        if not self._is_finite_number(
            phase_loss
        ):
            raise RuntimeError(
                f"Epoch {epoch}: {phase} phase returned "
                f"a non-finite aggregate loss: {phase_loss}."
            )

        self._logger.info(
            "Epoch %d %s complete: "
            "loss=%.6f, batches=%s",
            epoch,
            phase,
            phase_loss,
            (
                str(num_batches)
                if num_batches is not None
                else "unknown"
            ),
        )

        return metrics

    # ==================================================================
    # Validation helpers
    # ==================================================================

    @staticmethod
    def _validate_batch(
        images: Any,
        masks: Any,
        epoch: int,
        batch_idx: int,
        phase: str,
    ) -> None:
        """
        Validate model inputs before forward propagation.
        """

        import torch

        if not torch.is_floating_point(images):
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"images must be floating point, "
                f"received dtype={images.dtype}."
            )

        if images.ndim != 4:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"expected image shape (B, C, H, W), "
                f"received {tuple(images.shape)}."
            )

        if masks.ndim != 3:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"expected mask shape (B, H, W), "
                f"received {tuple(masks.shape)}."
            )

        if images.shape[0] != masks.shape[0]:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"image batch size={images.shape[0]}, "
                f"mask batch size={masks.shape[0]}."
            )

        if (
            images.shape[-2:]
            != masks.shape[-2:]
        ):
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"image spatial shape={tuple(images.shape[-2:])}, "
                f"mask spatial shape={tuple(masks.shape[-2:])}."
            )

        finite_mask = torch.isfinite(images)

        if not finite_mask.all():

            nan_count = int(
                torch.isnan(images).sum().item()
            )

            inf_count = int(
                torch.isinf(images).sum().item()
            )

            sanitized = torch.nan_to_num(
                images.detach(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"input images contain non-finite values. "
                f"nan={nan_count}, "
                f"inf={inf_count}, "
                f"shape={tuple(images.shape)}, "
                f"sanitized_min={sanitized.min().item():.6f}, "
                f"sanitized_max={sanitized.max().item():.6f}."
            )

        if masks.numel() == 0:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                "mask tensor is empty."
            )

        if masks.min().item() < 0:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"negative mask class IDs found: "
                f"{torch.unique(masks).detach().cpu().tolist()}."
            )

    @staticmethod
    def _validate_logits(
        logits: Any,
        epoch: int,
        batch_idx: int,
        phase: str,
    ) -> None:
        """
        Validate model outputs before loss calculation.
        """

        import torch

        if logits is None:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                "model returned None."
            )

        # Some deep-supervision models return a tuple/list.
        # The active loss function currently expects one tensor.
        if isinstance(logits, (tuple, list)):
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                "model returned multiple outputs. "
                "Configure deep_supervision=false or update "
                "the loss function to process auxiliary outputs."
            )

        if logits.ndim != 4:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"expected logits shape (B, C, H, W), "
                f"received {tuple(logits.shape)}."
            )

        if not torch.isfinite(logits).all():

            nan_count = int(
                torch.isnan(logits).sum().item()
            )

            inf_count = int(
                torch.isinf(logits).sum().item()
            )

            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"model produced non-finite logits. "
                f"nan={nan_count}, "
                f"inf={inf_count}, "
                f"shape={tuple(logits.shape)}."
            )

    @staticmethod
    def _validate_loss(
        loss: Any,
        masks: Any,
        epoch: int,
        batch_idx: int,
        phase: str,
    ) -> None:
        """
        Validate loss before backward propagation.
        """

        import torch

        if loss is None:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                "loss function returned None."
            )

        if loss.numel() != 1:
            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"loss must be scalar, "
                f"received shape={tuple(loss.shape)}."
            )

        if not torch.isfinite(loss).all():

            classes = (
                torch.unique(masks)
                .detach()
                .cpu()
                .tolist()
            )

            raise RuntimeError(
                f"Epoch {epoch}, {phase} batch {batch_idx}: "
                f"loss is non-finite. "
                f"loss={loss.detach().cpu().item()}, "
                f"mask_classes={classes}."
            )

    def _validate_gradients(
        self,
        epoch: int,
        batch_idx: int,
    ) -> None:
        """
        Fail immediately when model gradients contain NaN or Inf.
        """

        import torch

        for name, parameter in self._model.named_parameters():

            gradient = parameter.grad

            if gradient is None:
                continue

            if not torch.isfinite(
                gradient
            ).all():

                nan_count = int(
                    torch.isnan(
                        gradient
                    ).sum().item()
                )

                inf_count = int(
                    torch.isinf(
                        gradient
                    ).sum().item()
                )

                raise RuntimeError(
                    f"Epoch {epoch}, train batch {batch_idx}: "
                    f"parameter '{name}' has non-finite gradients. "
                    f"nan={nan_count}, "
                    f"inf={inf_count}."
                )

    # ==================================================================
    # Utility helpers
    # ==================================================================

    @staticmethod
    def _should_step(
        batch_idx: int,
        loader_length: int | None,
        accumulation_steps: int,
    ) -> bool:
        """
        Determine whether the optimizer must step.

        The final incomplete accumulation group is also applied rather
        than being silently discarded.
        """

        completed_batches = (
            batch_idx + 1
        )

        accumulation_complete = (
            completed_batches
            % accumulation_steps
            == 0
        )

        is_last_batch = (
            loader_length is not None
            and completed_batches
            == loader_length
        )

        return (
            accumulation_complete
            or is_last_batch
        )

    @staticmethod
    def _is_finite_number(
        value: float,
    ) -> bool:
        """
        Return True only for finite Python numeric values.
        """

        import math

        return math.isfinite(
            float(value)
        )

    @staticmethod
    def _unpack_batch(
        batch: Any,
        device: Any,
    ) -> tuple[Any, Any]:
        """
        Extract images and masks and move them to the target device.

        Supported input formats:

        Module 11:
            (images, masks, metadata_list)

        Module 12:
            (images, masks)
        """

        import torch

        if not isinstance(
            batch,
            (tuple, list),
        ):
            raise TypeError(
                "Trainer expected a tuple/list batch, "
                f"received {type(batch).__name__}."
            )

        if len(batch) < 2:
            raise ValueError(
                "Trainer expected at least "
                "(images, masks), "
                f"received {len(batch)} batch elements."
            )

        images = batch[0].to(
            device,
            non_blocking=True,
        )

        masks = batch[1].to(
            device,
            non_blocking=True,
        )

        if not torch.is_floating_point(
            images
        ):
            images = images.float()

        if masks.dtype != torch.long:
            masks = masks.long()

        return images, masks


class _null_context:
    """
    No-op context manager used during training.

    Training requires gradient tracking, so torch.no_grad() cannot be
    used for the training phase.
    """

    def __enter__(self):
        return self

    def __exit__(
        self,
        *args: Any,
    ) -> bool:
        return False