"""
Training engine factory for Module 14.

TrainingEngineFactory assembles Trainer, CheckpointManager, CallbackList,
OptimizerFactory, SchedulerFactory, LossRegistry, and SeedManager into a
ready-to-run training context from a single TrainingConfig.

TrainingEngine calls TrainingEngineFactory.build() internally; users never
call this class directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.training.engine.callbacks import (
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from src.training.engine.checkpoint import CheckpointManager
from src.training.engine.contracts import TrainingConfig
from src.training.engine.logger import TrainingLogger
from src.training.engine.losses import LossRegistry
from src.training.engine.optimizer import OptimizerFactory
from src.training.engine.scheduler import SchedulerFactory
from src.training.engine.seed import SeedManager
from src.training.engine.trainer import Trainer

__all__ = ["TrainingEngineFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TrainingEngineFactory:
    """
    Assembles all training components from a TrainingConfig.

    Returns a dict ("training context") that TrainingEngine passes to Trainer.
    """

    @classmethod
    def build(
        cls,
        config:       TrainingConfig,
        model_result: Any,
        data_result:  Any,
    ) -> dict:
        """
        Build the complete training context.

        Args:
            config:       TrainingConfig.
            model_result: ModelResult from Module 13.
            data_result:  TransformPipelineResult from Module 12.

        Returns:
            Dict containing: model, optimizer, scheduler, loss_fn, scaler,
            device, callbacks, checkpoint_manager, trainer, training_logger,
            train_loader, val_loader, data_result, model_result, config.
        """
        ops: list[str] = []

        # Step 1: Seed everything.
        SeedManager.seed(
            seed            = config.seed,
            deterministic   = config.deterministic,
            cudnn_benchmark = config.cudnn_benchmark,
        )
        ops.append(f"seed: {config.seed}")

        # Step 2: Resolve device with AMP fallback.
        device = cls._resolve_device(config)
        ops.append(f"device: {device}")

        # Step 3: Move model to device.
        model = model_result.model
        model.to(device)
        ops.append(f"model: {model_result.architecture} -> {device}")

        # Step 4: Build optimizer.
        optimizer = OptimizerFactory.build(model, config.optimizer)
        ops.append(f"optimizer: {config.optimizer.name} (lr={config.optimizer.lr})")

        # Step 5: Build scheduler.
        scheduler = SchedulerFactory.build(optimizer, config.scheduler)
        ops.append(f"scheduler: {config.scheduler.name} (enabled={config.scheduler.enabled})")

        # Step 6: Build loss function.
        loss_fn = LossRegistry.build(config.loss)
        loss_fn.to(device)
        ops.append(f"loss: {config.loss.name}")

        # Step 7: Inject class weights into loss if available.
        if hasattr(data_result, "normalization_stats"):
            pass  # class weights come from DataLoaderBundle; injected separately
        ops.append("loss_weights: deferred to caller")

        # Step 8: Build AMP GradScaler (falls back to None on CPU).
        scaler = cls._build_scaler(config, device)
        ops.append(f"amp: {'enabled' if scaler is not None else 'disabled'}")

        # Step 9: Build DataLoaders from data_result.
        train_loader, val_loader = cls._build_loaders(config, data_result)
        ops.append(
            f"loaders: train={len(data_result.train_dataset)}, "
            f"val={getattr(data_result, 'num_val_samples', 0)}"
        )

        # Step 10: Build checkpoint manager.
        ckpt_manager = CheckpointManager(config.checkpoint)
        ops.append(f"checkpoint_dir: {config.checkpoint.checkpoint_dir}")

        # Step 11: Build logger.
        training_logger = TrainingLogger(config, model_result.architecture)

        # Step 12: Assemble shared context.
        context: dict = {
            "model":            model,
            "optimizer":        optimizer,
            "scheduler":        scheduler,
            "scaler":           scaler,
            "loss_fn":          loss_fn,
            "device":           device,
            "model_result":     model_result,
            "data_result":      data_result,
            "config":           config,
            "training_logger":  training_logger,
            "stop_training":    False,
            "is_best_epoch":    False,
        }

        # Step 13: Build callback list.
        callbacks = cls._build_callbacks(config, ckpt_manager, context)
        context["callbacks"] = callbacks
        ops.append(f"callbacks: {len(callbacks)}")

        # Step 14: Build Trainer.
        trainer = Trainer(
            config    = config,
            model     = model,
            optimizer = optimizer,
            scheduler = scheduler,
            loss_fn   = loss_fn,
            callbacks = callbacks,
            device    = device,
            scaler    = scaler,
        )
        context["trainer"]             = trainer
        context["checkpoint_manager"]  = ckpt_manager
        context["train_loader"]        = train_loader
        context["val_loader"]          = val_loader
        context["operations_log"]      = tuple(ops)

        return context

    @staticmethod
    def _resolve_device(config: TrainingConfig) -> Any:
        """Resolve device, falling back to CPU when CUDA is requested but unavailable."""
        try:
            import torch
            requested = config.device
            if requested.startswith("cuda"):
                if torch.cuda.is_available():
                    return torch.device(requested)
                _LOGGER.warning(
                    "TrainingEngineFactory: CUDA requested but not available; "
                    "falling back to CPU."
                )
                return torch.device("cpu")
            return torch.device(requested)
        except ImportError:
            return "cpu"

    @staticmethod
    def _build_scaler(config: TrainingConfig, device: Any) -> Any | None:
        """Build AMP GradScaler when mixed_precision=True and CUDA is active."""
        if not config.mixed_precision:
            return None
        try:
            import torch
            device_type = getattr(device, "type", str(device))
            if device_type == "cuda":
                return torch.cuda.amp.GradScaler()
            _LOGGER.warning(
                "TrainingEngineFactory: mixed_precision=True but device is %s; "
                "AMP disabled (FP32 fallback).",
                device_type,
            )
        except ImportError:
            pass
        return None

    @staticmethod
    def _build_loaders(config: TrainingConfig, data_result: Any) -> tuple[Any, Any]:
        """Build DataLoaders from TransformPipelineResult datasets."""
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            # Return the raw dataset objects when torch is unavailable (test stubs).
            return data_result.train_dataset, data_result.validation_dataset

        val_bs = config.val_batch_size if config.val_batch_size > 0 else config.batch_size

        train_loader = DataLoader(
            data_result.train_dataset,
            batch_size  = config.batch_size,
            shuffle     = True,
            num_workers = config.num_workers,
            pin_memory  = getattr(torch.device(config.device), "type", config.device) == "cuda",
            drop_last   = False,
        )
        val_loader = DataLoader(
            data_result.validation_dataset,
            batch_size  = val_bs,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = getattr(torch.device(config.device), "type", config.device) == "cuda",
        )
        return train_loader, val_loader

    @staticmethod
    def _build_callbacks(
        config:      TrainingConfig,
        ckpt_manager: CheckpointManager,
        context:      dict,
    ) -> CallbackList:
        """Assemble the callback list from config."""
        cbs = CallbackList()

        # Logging callback (always present).
        cbs.append(LoggingCallback(log_every=1))

        # Checkpoint callback.
        if config.checkpoint.save_best or config.checkpoint.save_latest:
            cbs.append(
                CheckpointCallback(
                    checkpoint_manager = ckpt_manager,
                    mode               = config.checkpoint.mode,
                    metric             = config.checkpoint.metric,
                )
            )

        # Early stopping callback.
        if config.early_stopping_patience > 0:
            cbs.append(
                EarlyStoppingCallback(
                    patience = config.early_stopping_patience,
                    mode     = config.checkpoint.mode,
                    metric   = config.checkpoint.metric,
                )
            )

        return cbs
