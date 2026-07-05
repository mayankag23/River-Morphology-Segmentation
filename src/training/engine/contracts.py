"""
Public data contracts for the Training Engine Framework (Module 14).

Contract chain:
    ModelResult            (Module 13)  ──┐
    TransformPipelineResult (Module 12) ──┤──> TrainingEngine.train() ──> TrainingResult
    Config                              ──┘

TrainingResult is the immutable public output of TrainingEngine.train().
Module 15 (Evaluation) and Module 16 (Inference) consume this object.

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- No torch types appear at the module level (lazy import policy).
- TrainingResult carries the trained model and full provenance so any
  downstream module can reconstruct or resume from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "CheckpointConfig",
    "EpochResult",
    "TrainingResult",
]


# ==============================================================================
# OptimizerConfig
# ==============================================================================

@dataclass(frozen=True)
class OptimizerConfig:
    """
    Immutable optimizer configuration.

    Attributes:
        name:         "adam", "adamw", or "sgd".
        lr:           Base learning rate.
        weight_decay: L2 regularisation coefficient.
        momentum:     SGD momentum (ignored by Adam/AdamW).
        betas:        Adam/AdamW beta coefficients (beta1, beta2).
        eps:          Adam/AdamW numerical stability term.
    """

    name:         str               = "adamw"
    lr:           float             = 1e-4
    weight_decay: float             = 1e-4
    momentum:     float             = 0.9
    betas:        tuple[float, ...] = (0.9, 0.999)
    eps:          float             = 1e-8

    @classmethod
    def from_config(cls, config: Any) -> OptimizerConfig:
        train_cfg = getattr(config, "training", None)
        opt_cfg   = getattr(train_cfg, "optimizer", None)
        if opt_cfg is None:
            return cls()
        raw_betas = getattr(opt_cfg, "betas", (0.9, 0.999))
        return cls(
            name         = str(getattr(opt_cfg,   "name",         "adamw")),
            lr           = float(getattr(opt_cfg, "lr",           1e-4)),
            weight_decay = float(getattr(opt_cfg, "weight_decay", 1e-4)),
            momentum     = float(getattr(opt_cfg, "momentum",     0.9)),
            betas        = tuple(float(b) for b in raw_betas),
            eps          = float(getattr(opt_cfg, "eps",          1e-8)),
        )


# ==============================================================================
# SchedulerConfig
# ==============================================================================

@dataclass(frozen=True)
class SchedulerConfig:
    """
    Immutable learning-rate scheduler configuration.

    Attributes:
        name:          "cosine", "step", or "plateau".
        t_max:         CosineAnnealingLR: number of epochs in one cosine cycle.
        eta_min:       CosineAnnealingLR: minimum learning rate.
        step_size:     StepLR: period (in epochs) for LR decay.
        gamma:         StepLR/ReduceLROnPlateau: multiplicative decay factor.
        patience:      ReduceLROnPlateau: epochs with no improvement before decay.
        min_lr:        ReduceLROnPlateau: minimum LR floor.
        mode:          ReduceLROnPlateau: "min" (loss) or "max" (accuracy).
        enabled:       False disables the scheduler entirely.
    """

    name:      str   = "cosine"
    t_max:     int   = 100
    eta_min:   float = 1e-6
    step_size: int   = 30
    gamma:     float = 0.1
    patience:  int   = 10
    min_lr:    float = 1e-7
    mode:      str   = "min"
    enabled:   bool  = True

    @classmethod
    def from_config(cls, config: Any) -> SchedulerConfig:
        train_cfg  = getattr(config, "training", None)
        sched_cfg  = getattr(train_cfg, "scheduler", None)
        if sched_cfg is None:
            return cls()
        return cls(
            name      = str(getattr(sched_cfg,   "name",      "cosine")),
            t_max     = int(getattr(sched_cfg,   "t_max",     100)),
            eta_min   = float(getattr(sched_cfg, "eta_min",   1e-6)),
            step_size = int(getattr(sched_cfg,   "step_size", 30)),
            gamma     = float(getattr(sched_cfg, "gamma",     0.1)),
            patience  = int(getattr(sched_cfg,   "patience",  10)),
            min_lr    = float(getattr(sched_cfg, "min_lr",    1e-7)),
            mode      = str(getattr(sched_cfg,   "mode",      "min")),
            enabled   = bool(getattr(sched_cfg,  "enabled",   True)),
        )


# ==============================================================================
# LossConfig
# ==============================================================================

@dataclass(frozen=True)
class LossConfig:
    """
    Immutable loss function configuration.

    Attributes:
        name:         "cross_entropy", "dice", "focal", or "combined".
        ignore_index: Class index to ignore (nodata pixels). Default 255.
        label_smoothing: Cross-entropy label smoothing in [0.0, 1.0].
        dice_smooth:  Dice loss smoothing factor.
        focal_alpha:  Focal loss alpha (class balance factor).
        focal_gamma:  Focal loss focusing parameter.
        ce_weight:    Weight of CE term in combined loss.
        dice_weight:  Weight of Dice term in combined loss.
    """

    name:            str   = "cross_entropy"
    ignore_index:    int   = 255
    label_smoothing: float = 0.0
    dice_smooth:     float = 1.0
    focal_alpha:     float = 1.0
    focal_gamma:     float = 2.0
    ce_weight:       float = 0.5
    dice_weight:     float = 0.5

    @classmethod
    def from_config(cls, config: Any) -> LossConfig:
        train_cfg = getattr(config, "training", None)
        loss_cfg  = getattr(train_cfg, "loss", None)
        if loss_cfg is None:
            return cls()
        return cls(
            name            = str(getattr(loss_cfg,   "name",            "cross_entropy")),
            ignore_index    = int(getattr(loss_cfg,   "ignore_index",    255)),
            label_smoothing = float(getattr(loss_cfg, "label_smoothing", 0.0)),
            dice_smooth     = float(getattr(loss_cfg, "dice_smooth",     1.0)),
            focal_alpha     = float(getattr(loss_cfg, "focal_alpha",     1.0)),
            focal_gamma     = float(getattr(loss_cfg, "focal_gamma",     2.0)),
            ce_weight       = float(getattr(loss_cfg, "ce_weight",       0.5)),
            dice_weight     = float(getattr(loss_cfg, "dice_weight",     0.5)),
        )


# ==============================================================================
# CheckpointConfig
# ==============================================================================

@dataclass(frozen=True)
class CheckpointConfig:
    """
    Immutable checkpoint configuration.

    Attributes:
        checkpoint_dir:   Directory for saving checkpoints.
        save_best:        Save checkpoint when validation loss improves.
        save_latest:      Always overwrite the latest checkpoint each epoch.
        resume_from:      Path to a checkpoint to resume from. None = fresh start.
        checkpoint_version: Format version string written into every checkpoint.
        metric:           Metric to monitor for best-checkpoint selection.
        mode:             "min" (lower = better) or "max" (higher = better).
    """

    checkpoint_dir:      str       = "checkpoints"
    save_best:           bool      = True
    save_latest:         bool      = True
    resume_from:         str | None = None
    checkpoint_version:  str       = "1.0"
    metric:              str       = "val_loss"
    mode:                str       = "min"

    @classmethod
    def from_config(cls, config: Any) -> CheckpointConfig:
        train_cfg = getattr(config, "training", None)
        ckpt_cfg  = getattr(train_cfg, "checkpoint", None)
        if ckpt_cfg is None:
            return cls()
        return cls(
            checkpoint_dir     = str(getattr(ckpt_cfg,  "checkpoint_dir",     "checkpoints")),
            save_best          = bool(getattr(ckpt_cfg, "save_best",          True)),
            save_latest        = bool(getattr(ckpt_cfg, "save_latest",        True)),
            resume_from        = getattr(ckpt_cfg,      "resume_from",        None),
            checkpoint_version = str(getattr(ckpt_cfg,  "checkpoint_version", "1.0")),
            metric             = str(getattr(ckpt_cfg,  "metric",             "val_loss")),
            mode               = str(getattr(ckpt_cfg,  "mode",               "min")),
        )


# ==============================================================================
# TrainingConfig
# ==============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """
    Top-level immutable training configuration.

    Attributes:
        epochs:            Total training epochs.
        seed:              Global random seed for reproducibility.
        device:            Target device: "cpu", "cuda", "cuda:0", etc.
        mixed_precision:   Enable automatic mixed precision (AMP).
        grad_clip_value:   Max gradient norm for clipping. 0.0 = disabled.
        accumulation_steps: Gradient accumulation steps (effective batch size
                            multiplier). 1 = no accumulation.
        num_workers:       DataLoader worker processes.
        batch_size:        Training batch size.
        val_batch_size:    Validation batch size. 0 = same as batch_size.
        early_stopping_patience: 0 = disabled; >0 = epochs with no improvement.
        optimizer:         OptimizerConfig.
        scheduler:         SchedulerConfig.
        loss:              LossConfig.
        checkpoint:        CheckpointConfig.
        deterministic:     Enable PyTorch deterministic algorithms.
        cudnn_benchmark:   Enable cuDNN autotuner (disable for determinism).
    """

    epochs:                    int              = 100
    seed:                      int              = 42
    device:                    str              = "cpu"
    mixed_precision:           bool             = False
    grad_clip_value:           float            = 0.0
    accumulation_steps:        int              = 1
    num_workers:               int              = 4
    batch_size:                int              = 8
    val_batch_size:            int              = 0
    early_stopping_patience:   int              = 0
    optimizer:                 OptimizerConfig  = field(default_factory=OptimizerConfig)
    scheduler:                 SchedulerConfig  = field(default_factory=SchedulerConfig)
    loss:                      LossConfig       = field(default_factory=LossConfig)
    checkpoint:                CheckpointConfig = field(default_factory=CheckpointConfig)
    deterministic:             bool             = False
    cudnn_benchmark:           bool             = False

    @classmethod
    def from_config(cls, config: Any) -> TrainingConfig:
        train_cfg = getattr(config, "training", None)
        if train_cfg is None:
            return cls()
        return cls(
            epochs                  = int(getattr(train_cfg,   "epochs",                  100)),
            seed                    = int(getattr(train_cfg,   "seed",                    42)),
            device                  = str(getattr(train_cfg,   "device",                  "cpu")),
            mixed_precision         = bool(getattr(train_cfg,  "mixed_precision",         False)),
            grad_clip_value         = float(getattr(train_cfg, "grad_clip_value",         0.0)),
            accumulation_steps      = int(getattr(train_cfg,   "accumulation_steps",      1)),
            num_workers             = int(getattr(train_cfg,   "num_workers",             4)),
            batch_size              = int(getattr(train_cfg,   "batch_size",              8)),
            val_batch_size          = int(getattr(train_cfg,   "val_batch_size",          0)),
            early_stopping_patience = int(getattr(train_cfg,   "early_stopping_patience", 0)),
            optimizer               = OptimizerConfig.from_config(config),
            scheduler               = SchedulerConfig.from_config(config),
            loss                    = LossConfig.from_config(config),
            checkpoint              = CheckpointConfig.from_config(config),
            deterministic           = bool(getattr(train_cfg,  "deterministic",           False)),
            cudnn_benchmark         = bool(getattr(train_cfg,  "cudnn_benchmark",         False)),
        )


# ==============================================================================
# EpochResult
# ==============================================================================

@dataclass(frozen=True)
class EpochResult:
    """
    Immutable record for one training epoch.

    Attributes:
        epoch:       1-based epoch number.
        train_loss:  Mean training loss over all batches.
        val_loss:    Mean validation loss (0.0 when no val dataset).
        lr:          Learning rate at end of epoch.
        epoch_time:  Wall-clock seconds for this epoch.
        is_best:     True when this epoch achieved the best monitored metric.
        extra:       Free-form dict for any additional metrics logged
                     by callbacks (e.g. per-class loss terms).
    """

    epoch:       int
    train_loss:  float
    val_loss:    float
    lr:          float
    epoch_time:  float
    is_best:     bool
    extra:       dict = field(default_factory=dict)


# ==============================================================================
# TrainingResult
# ==============================================================================

@dataclass(frozen=True)
class TrainingResult:
    """
    Immutable public output of TrainingEngine.train().

    Module 15 (Evaluation) and Module 16 (Inference) consume this object.

    Attributes:
        model:              The trained torch.nn.Module (on CPU after training).
        history:            List of EpochResult, one per completed epoch.
        best_epoch:         1-based epoch index with the best monitored metric.
        best_metric:        Best monitored metric value.
        best_checkpoint:    Absolute path to the best checkpoint file. None if
                            checkpointing was disabled.
        latest_checkpoint:  Absolute path to the latest checkpoint file.
        total_epochs:       Total epochs completed (may be < config.epochs on
                            early stopping).
        architecture:       Model architecture name.
        num_parameters:     Total model parameter count.
        seed:               Random seed used for this training run.
        training_config:    The TrainingConfig used for this run.
        operations_log:     Ordered log of engine construction steps.
        stopped_early:      True if training was halted by early stopping.
    """

    model:              Any
    history:            tuple[EpochResult, ...]
    best_epoch:         int
    best_metric:        float
    best_checkpoint:    Path | None
    latest_checkpoint:  Path | None
    total_epochs:       int
    architecture:       str
    num_parameters:     int
    seed:               int
    training_config:    TrainingConfig
    operations_log:     tuple[str, ...]
    stopped_early:      bool

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        best_loss = f"{self.best_metric:.6f}" if self.best_metric else "n/a"
        return [
            f"  architecture:    {self.architecture}",
            f"  total_epochs:    {self.total_epochs}",
            f"  best_epoch:      {self.best_epoch}",
            f"  best_metric:     {best_loss}",
            f"  stopped_early:   {self.stopped_early}",
            f"  num_parameters:  {self.num_parameters:,}",
            f"  seed:            {self.seed}",
            f"  best_ckpt:       {self.best_checkpoint}",
        ]
