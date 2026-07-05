"""
Pre-flight validation for the Training Engine Framework (Module 14).

TrainingValidator checks configuration consistency, model/data compatibility,
and device availability before training begins. It never raises; it accumulates
issues and the caller decides whether to abort.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.training.engine.contracts import TrainingConfig

__all__ = ["TrainingValidator", "TrainingValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TrainingValidationResult:
    """Result of pre-flight validation."""

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


class TrainingValidator:
    """
    Validates all inputs to TrainingEngine before training begins.
    """

    def validate(
        self,
        config:       TrainingConfig,
        model_result: Any,
        data_result:  Any,
    ) -> TrainingValidationResult:
        """
        Run all pre-flight checks.

        Args:
            config:       TrainingConfig.
            model_result: ModelResult from Module 13.
            data_result:  TransformPipelineResult from Module 12.

        Returns:
            TrainingValidationResult with all detected issues.
        """
        issues: list[str] = []
        issues.extend(self._check_config(config))
        issues.extend(self._check_model(model_result))
        issues.extend(self._check_data(data_result))
        issues.extend(self._check_compatibility(model_result, data_result))
        issues.extend(self._check_device(config))
        issues.extend(self._check_checkpoint(config))
        return TrainingValidationResult(issues)

    @staticmethod
    def _check_config(cfg: TrainingConfig) -> list[str]:
        issues: list[str] = []
        if cfg.epochs < 1:
            issues.append(f"epochs must be >= 1, got {cfg.epochs}.")
        if cfg.optimizer.lr <= 0:
            issues.append(f"optimizer.lr must be > 0, got {cfg.optimizer.lr}.")
        if cfg.batch_size < 1:
            issues.append(f"batch_size must be >= 1, got {cfg.batch_size}.")
        if cfg.accumulation_steps < 1:
            issues.append(f"accumulation_steps must be >= 1, got {cfg.accumulation_steps}.")
        if not (0.0 <= cfg.loss.label_smoothing < 1.0):
            issues.append(
                f"loss.label_smoothing must be in [0, 1), got {cfg.loss.label_smoothing}."
            )
        if cfg.grad_clip_value < 0.0:
            issues.append(f"grad_clip_value must be >= 0, got {cfg.grad_clip_value}.")
        return issues

    @staticmethod
    def _check_model(model_result: Any) -> list[str]:
        issues: list[str] = []
        if model_result is None:
            issues.append("model_result is None.")
            return issues
        if getattr(model_result, "model", None) is None:
            issues.append("model_result.model is None.")
        if getattr(model_result, "num_trainable", 0) == 0:
            issues.append("model has 0 trainable parameters.")
        return issues

    @staticmethod
    def _check_data(data_result: Any) -> list[str]:
        issues: list[str] = []
        if data_result is None:
            issues.append("data_result (TransformPipelineResult) is None.")
            return issues
        if getattr(data_result, "num_train_samples", 0) == 0:
            issues.append("Training dataset is empty (num_train_samples == 0).")
        if not getattr(data_result, "is_valid", True):
            issues.append(
                f"TransformPipelineResult is invalid: "
                f"{getattr(data_result, 'validation_issues', [])}"
            )
        return issues

    @staticmethod
    def _check_compatibility(model_result: Any, data_result: Any) -> list[str]:
        issues: list[str] = []
        if model_result is None or data_result is None:
            return issues
        model_classes = getattr(model_result, "num_classes",  None)
        data_classes  = getattr(data_result,  "num_classes",  None)
        if (
            model_classes is not None
            and data_classes  is not None
            and model_classes != data_classes
        ):
            issues.append(
                f"num_classes mismatch: model={model_classes}, "
                f"data={data_classes}."
            )
        model_bands = getattr(model_result, "in_channels", None)
        data_bands  = getattr(data_result,  "num_bands",   None)
        if (
            model_bands is not None
            and data_bands  is not None
            and model_bands != data_bands
        ):
            issues.append(
                f"in_channels/num_bands mismatch: model={model_bands}, "
                f"data={data_bands}."
            )
        return issues

    @staticmethod
    def _check_device(cfg: TrainingConfig) -> list[str]:
        issues: list[str] = []
        if cfg.device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append(
                        f"Device '{cfg.device}' requested but CUDA is not available. "
                        f"Training will fall back to CPU."
                    )
            except ImportError:
                issues.append("torch is not installed; cannot validate device.")
        return issues

    @staticmethod
    def _check_checkpoint(cfg: TrainingConfig) -> list[str]:
        issues: list[str] = []
        resume = cfg.checkpoint.resume_from
        if resume is not None:
            p = Path(resume)
            if not p.exists():
                issues.append(f"resume_from checkpoint not found: {p}")
        return issues
