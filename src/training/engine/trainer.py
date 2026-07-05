"""
Core training loop for the Training Engine Framework (Module 14).

Trainer owns the epoch and batch loops. It calls callbacks at precise hook
points and communicates with CheckpointManager, MetricsAccumulator, and
SchedulerFactory through a shared context dict.

Responsibilities
----------------
- Epoch loop (train + val)
- Batch loop with gradient accumulation
- AMP forward/backward with GradScaler
- Gradient norm clipping
- Scheduler step
- Callback dispatch
- Early stopping signal detection

Does NOT:
- Build models, optimizers, schedulers, or loss functions (TrainingEngine does)
- Save checkpoints directly (CheckpointCallback does)
- Compute IoU/Dice/Precision/Recall (Module 15)
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
    Executes the training loop given pre-built components.

    Args:
        config:      TrainingConfig controlling loop behaviour.
        model:       torch.nn.Module — the segmentation model.
        optimizer:   torch.optim.Optimizer.
        scheduler:   LR scheduler or None.
        loss_fn:     Loss function nn.Module.
        callbacks:   CallbackList executed at each hook point.
        device:      torch.device — target computation device.
        scaler:      torch.cuda.amp.GradScaler or None for AMP.
    """

    def __init__(
        self,
        config:    TrainingConfig,
        model:     Any,
        optimizer: Any,
        scheduler: Any,
        loss_fn:   Any,
        callbacks: CallbackList,
        device:    Any,
        scaler:    Any = None,
    ) -> None:
        self._config    = config
        self._model     = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_fn   = loss_fn
        self._callbacks = callbacks
        self._device    = device
        self._scaler    = scaler
        self._logger: logging.Logger = logging.getLogger(__name__)

    def run(
        self,
        train_loader: Any,
        val_loader:   Any | None,
        start_epoch:  int,
        context:      dict,
    ) -> list[EpochResult]:
        """
        Execute the training loop from start_epoch to config.epochs.

        Args:
            train_loader: DataLoader for training split.
            val_loader:   DataLoader for validation split (None = skip val).
            start_epoch:  First epoch (1-based; >1 when resuming).
            context:      Shared context dict (passed to callbacks; may be
                          mutated by callbacks to signal early stopping).

        Returns:
            List of EpochResult, one per completed epoch.
        """
        results: list[EpochResult] = []
        self._callbacks.on_train_begin(context)

        for epoch in range(start_epoch, self._config.epochs + 1):
            self._callbacks.on_epoch_begin(epoch, context)

            # --- Training phase ---
            train_metrics = self._run_phase(
                loader     = train_loader,
                epoch      = epoch,
                is_train   = True,
                context    = context,
            )

            # --- Validation phase ---
            val_loss = 0.0
            if val_loader is not None:
                val_metrics = self._run_phase(
                    loader   = val_loader,
                    epoch    = epoch,
                    is_train = False,
                    context  = context,
                )
                val_loss = val_metrics["loss"]

            # --- LR retrieval and scheduler step ---
            current_lr = MetricsAccumulator.get_lr(self._optimizer)
            SchedulerFactory.step(self._scheduler, self._config.scheduler, val_loss)

            # --- Assemble epoch result ---
            is_best = context.get("is_best_epoch", False)
            result  = EpochResult(
                epoch      = epoch,
                train_loss = train_metrics["loss"],
                val_loss   = val_loss,
                lr         = current_lr,
                epoch_time = train_metrics["epoch_time"],
                is_best    = is_best,
            )
            results.append(result)
            context["last_epoch_result"] = result

            self._callbacks.on_epoch_end(epoch, result, context)

            if context.get("stop_training", False):
                self._logger.info("Trainer: early stopping at epoch %d.", epoch)
                break

        self._callbacks.on_train_end(context)
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        loader:   Any,
        epoch:    int,
        is_train: bool,
        context:  dict,
    ) -> dict[str, float]:
        """Run one full pass (train or val) over the DataLoader."""
        import torch

        if is_train:
            self._model.train()
        else:
            self._model.eval()

        acc = MetricsAccumulator()
        acc.start()

        accum_steps  = max(1, self._config.accumulation_steps) if is_train else 1
        clip_value   = self._config.grad_clip_value
        use_amp      = (self._scaler is not None) and is_train

        if is_train:
            self._optimizer.zero_grad(set_to_none=True)

        context_mode = torch.no_grad() if not is_train else _null_context()

        with context_mode:
            for batch_idx, batch in enumerate(loader):
                images, masks = self._unpack_batch(batch, self._device)

                if is_train:
                    self._callbacks.on_batch_begin(batch_idx, context)

                if use_amp:
                    with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                        logits = self._model(images)
                        loss   = self._loss_fn(logits, masks) / accum_steps
                    self._scaler.scale(loss).backward()
                    if (batch_idx + 1) % accum_steps == 0:
                        if clip_value > 0.0:
                            self._scaler.unscale_(self._optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(), clip_value
                            )
                        self._scaler.step(self._optimizer)
                        self._scaler.update()
                        self._optimizer.zero_grad(set_to_none=True)
                elif is_train:
                    logits = self._model(images)
                    loss   = self._loss_fn(logits, masks) / accum_steps
                    loss.backward()
                    if (batch_idx + 1) % accum_steps == 0:
                        if clip_value > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(), clip_value
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                else:
                    logits = self._model(images)
                    loss   = self._loss_fn(logits, masks)

                batch_size  = images.shape[0]
                loss_scalar = float((loss * accum_steps).detach().cpu())
                acc.update(loss_scalar, batch_size)

                if is_train:
                    self._callbacks.on_batch_end(batch_idx, loss_scalar, context)

        return acc.compute()

    @staticmethod
    def _unpack_batch(batch: Any, device: Any) -> tuple[Any, Any]:
        """
        Extract (images, masks) from a batch tuple and move to device.

        Module 11's collate_fn returns (images, masks, metadata_list).
        Module 12's AugmentedDataset returns (image_tensor, mask_tensor).
        Both are handled here.
        """
        import torch
        images = batch[0].to(device, non_blocking=True)
        masks  = batch[1].to(device, non_blocking=True)
        if masks.dtype != torch.long:
            masks = masks.long()
        return images, masks


class _null_context:
    """No-op context manager used for training phase (gradients needed)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
