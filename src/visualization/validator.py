"""
Validation for the Visualization Framework (Module 18).

VisualizationValidator checks VisualizationConfig and RiverMorphologyResult
compatibility before rendering begins. Never raises; accumulates issues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.visualization.contracts import VisualizationConfig

__all__ = ["VisualizationValidator", "VisualizationValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class VisualizationValidationResult:
    """Result of one validation pass."""

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


class VisualizationValidator:
    """Pre-flight validation for the visualization engine."""

    def validate(
        self,
        config:            VisualizationConfig,
        morphology_result: Any,
    ) -> VisualizationValidationResult:
        """
        Validate VisualizationConfig and RiverMorphologyResult compatibility.

        Args:
            config:            VisualizationConfig.
            morphology_result: RiverMorphologyResult from Module 17.

        Returns:
            VisualizationValidationResult.
        """
        issues: list[str] = []

        if morphology_result is None:
            issues.append("morphology_result is None.")
            return VisualizationValidationResult(issues)

        # Config validation.
        if config.dpi < 1:
            issues.append(f"dpi must be >= 1, got {config.dpi}.")
        if config.figure_width <= 0 or config.figure_height <= 0:
            issues.append(
                f"figure_width and figure_height must be > 0; "
                f"got ({config.figure_width}, {config.figure_height})."
            )
        if not (0.0 <= config.alpha_overlay <= 1.0):
            issues.append(
                f"alpha_overlay must be in [0, 1], got {config.alpha_overlay}."
            )
        if not (0.0 <= config.alpha_confidence <= 1.0):
            issues.append(
                f"alpha_confidence must be in [0, 1], got {config.alpha_confidence}."
            )
        if config.max_samples < 0:
            issues.append(
                f"max_samples must be >= 0, got {config.max_samples}."
            )

        # Validate colors when specified.
        for cls_name, color in config.class_colors.items():
            if len(color) < 3:
                issues.append(
                    f"class_colors['{cls_name}'] must be a 3-tuple (R, G, B), "
                    f"got length {len(color)}."
                )
            else:
                for i, c in enumerate(color[:3]):
                    if not (0.0 <= c <= 1.0):
                        issues.append(
                            f"class_colors['{cls_name}'][{i}]={c} is outside [0, 1]."
                        )
                        break

        # Output dir.
        if config.output_dir:
            try:
                Path(config.output_dir)
            except Exception:
                issues.append(f"output_dir='{config.output_dir}' is an invalid path.")

        # Result validation.
        if getattr(morphology_result, "num_samples", 0) == 0:
            issues.append("morphology_result has 0 samples; nothing to visualize.")

        class_names = getattr(morphology_result, "class_names", ())
        if len(class_names) == 0:
            issues.append("morphology_result.class_names is empty.")

        return VisualizationValidationResult(issues)