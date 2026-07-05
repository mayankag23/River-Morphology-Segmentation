"""
Quality assessment for generated pseudo-label masks (Module 9).

QualityAssessment evaluates a MorphologyResult against configurable
thresholds and produces a QualityResult with:

    valid_pixel_ratio:    Fraction of pixels with a known class assignment.
    unclassified_ratio:    Fraction of pixels with no class assignment.
    class coverage:         At least one class must have >= min_class_pixels.
    quality_score:          Composite [0, 1] score from the three checks above.
    metric_scores:           Per-component scores for diagnostics.

Masks whose quality_score falls below min_quality_score, or that violate any
individual threshold, are marked is_acceptable=False. Rejection is recorded
here; actual exclusion from the dataset is handled downstream by LabelManager
(which also incorporates the ConfidenceEstimator result).

All thresholds come from Config. No threshold is hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.labels.contracts import MorphologyResult, QualityResult

__all__ = ["QualityConfig", "QualityAssessment"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityConfig:
    """
    Immutable quality assessment thresholds (all from Config).

    Attributes:
        min_valid_pixel_ratio:   Minimum fraction of classified pixels.
        min_quality_score:        Minimum composite quality score [0, 1].
        max_unclassified_ratio:   Maximum fraction of unclassified pixels.
        min_class_pixels:         Minimum pixels for a class to be counted as
                                  present.
    """

    min_valid_pixel_ratio:  float = 0.50
    min_quality_score:       float = 0.30
    max_unclassified_ratio:  float = 0.30
    min_class_pixels:         int   = 5

    @classmethod
    def from_config(cls, config: Any) -> QualityConfig:
        """Build QualityConfig from config.labels.quality."""
        q = getattr(getattr(config, "labels", None), "quality", None)
        if q is None:
            return cls()
        return cls(
            min_valid_pixel_ratio = float(getattr(q, "min_valid_pixel_ratio", 0.50)),
            min_quality_score      = float(getattr(q, "min_quality_score",     0.30)),
            max_unclassified_ratio = float(getattr(q, "max_unclassified_ratio", 0.30)),
            min_class_pixels       = int(getattr(q,   "min_class_pixels",        5)),
        )


class QualityAssessment:
    """
    Evaluates the quality of a generated class map.

    Args:
        quality_config: QualityConfig with acceptance thresholds.
        class_schema:    ClassSchema defining valid class IDs and names.
        nodata_value:    Sentinel value for unclassified pixels.
    """

    def __init__(
        self,
        quality_config: QualityConfig,
        class_schema:    Any,        # ClassSchema from src.labels.schema
        nodata_value:    int = 255,
    ) -> None:
        self._cfg          = quality_config
        self._schema       = class_schema
        self._nodata_value = int(nodata_value)
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Any, class_schema: Any) -> QualityAssessment:
        """Build from Config and ClassSchema."""
        nodata = int(getattr(getattr(config, "labels", None), "nodata_value", 255))
        return cls(
            quality_config=QualityConfig.from_config(config),
            class_schema=class_schema,
            nodata_value=nodata,
        )

    def assess(self, morph_result: MorphologyResult) -> QualityResult:
        """
        Evaluate the quality of a morphologically processed class map.

        Args:
            morph_result: MorphologyResult from MorphologyProcessor.process().

        Returns:
            QualityResult with quality_score, is_acceptable flag, and
            metric_scores for diagnostics.
        """
        class_map   = morph_result.class_map
        total_px    = int(class_map.size)
        issues: list[str] = []

        # Valid pixels: those with a known class ID in the schema.
        valid_ids  = set(self._schema.class_ids)
        valid_mask = np.isin(class_map, list(valid_ids))
        valid_px   = int(valid_mask.sum())

        unclassified_px    = int((class_map == self._nodata_value).sum())
        valid_ratio        = valid_px / total_px if total_px > 0 else 0.0
        unclassified_ratio = unclassified_px / total_px if total_px > 0 else 1.0

        # Per-class coverage.
        class_fractions: dict[str, float] = {}
        present_classes = 0
        for defn in self._schema.classes:
            count = int(np.sum(class_map == defn.class_id))
            frac  = count / total_px if total_px > 0 else 0.0
            class_fractions[defn.name] = frac
            if count >= self._cfg.min_class_pixels:
                present_classes += 1

        # Quality checks.
        if valid_ratio < self._cfg.min_valid_pixel_ratio:
            issues.append(
                f"valid_pixel_ratio {valid_ratio:.2%} below "
                f"threshold {self._cfg.min_valid_pixel_ratio:.2%}"
            )
        if unclassified_ratio > self._cfg.max_unclassified_ratio:
            issues.append(
                f"unclassified_ratio {unclassified_ratio:.2%} exceeds "
                f"threshold {self._cfg.max_unclassified_ratio:.2%}"
            )
        if present_classes == 0:
            issues.append("no class has sufficient pixel coverage")

        # Composite quality score (weighted average of three sub-scores).
        valid_score = min(
            valid_ratio / max(self._cfg.min_valid_pixel_ratio, 1e-8), 1.0
        )
        unclassified_score = max(
            1.0 - unclassified_ratio / max(self._cfg.max_unclassified_ratio, 1e-8),
            0.0,
        )
        class_score = min(
            present_classes / max(len(self._schema.classes), 1), 1.0
        )

        quality_score = float(
            0.5 * valid_score + 0.3 * unclassified_score + 0.2 * class_score
        )
        quality_score = round(min(quality_score, 1.0), 4)

        is_acceptable = (
            len(issues) == 0 and quality_score >= self._cfg.min_quality_score
        )

        metric_scores: dict[str, float] = {
            "valid_pixel_score":     round(valid_score, 4),
            "unclassified_score":    round(unclassified_score, 4),
            "class_coverage_score":  round(class_score, 4),
        }

        self._logger.debug(
            "QualityAssessment: score=%.3f, valid=%.1f%%, "
            "unclassified=%.1f%%, classes=%d, acceptable=%s",
            quality_score, valid_ratio * 100, unclassified_ratio * 100,
            present_classes, is_acceptable,
        )

        return QualityResult(
            quality_score=quality_score,
            is_acceptable=is_acceptable,
            valid_pixel_ratio=valid_ratio,
            unclassified_ratio=unclassified_ratio,
            class_pixel_fractions=class_fractions,
            num_classes_present=present_classes,
            issues=issues,
            metric_scores=metric_scores,
        )



# """
# Quality assessment for generated pseudo-label masks (Module 9).

# QualityAssessment evaluates a MorphologyResult against configurable
# thresholds:
#     - valid_pixel_ratio:   fraction of pixels with a class assignment
#     - unclassified_ratio:   fraction of pixels with no class
#     - class_pixel counts:   ensure at least one class has enough pixels
#     - quality_score:         composite [0, 1] score summarising all checks

# Masks below configured thresholds are marked is_acceptable=False.
# The LabelManifestEntry's is_valid field is set based on is_acceptable AND
# LabelValidator's output.

# All thresholds come from Config. No thresholds are hardcoded.
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from typing import Any

# import numpy as np

# from src.labels.contracts import MorphologyResult, QualityResult

# __all__ = ["QualityConfig", "QualityAssessment"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)


# @dataclass(frozen=True)
# class QualityConfig:
#     """
#     Immutable quality assessment thresholds.

#     Attributes:
#         min_valid_pixel_ratio:   Minimum fraction of classified pixels.
#         min_quality_score:        Minimum composite quality score [0, 1].
#         max_unclassified_ratio:   Maximum fraction of unclassified pixels.
#         min_class_pixels:         Minimum pixels for any class to be counted.
#     """

#     min_valid_pixel_ratio:  float = 0.50
#     min_quality_score:       float = 0.30
#     max_unclassified_ratio:  float = 0.30
#     min_class_pixels:         int   = 5

#     @classmethod
#     def from_config(cls, config: Any) -> QualityConfig:
#         """Build QualityConfig from config.labels.quality."""
#         q = getattr(getattr(config, "labels", None), "quality", None)
#         if q is None:
#             return cls()
#         return cls(
#             min_valid_pixel_ratio = float(getattr(q, "min_valid_pixel_ratio", 0.50)),
#             min_quality_score      = float(getattr(q, "min_quality_score",     0.30)),
#             max_unclassified_ratio = float(getattr(q, "max_unclassified_ratio", 0.30)),
#             min_class_pixels       = int(getattr(q,   "min_class_pixels",        5)),
#         )


# class QualityAssessment:
#     """
#     Evaluates the quality of a generated class map.

#     Args:
#         quality_config: QualityConfig with acceptance thresholds.
#         class_schema:    ClassSchema defining valid class IDs and names.
#         nodata_value:    Sentinel value written to unclassified pixels.
#     """

#     def __init__(
#         self,
#         quality_config: QualityConfig,
#         class_schema:    Any,        # ClassSchema from src.labels.schema
#         nodata_value:    int = 255,
#     ) -> None:
#         self._cfg         = quality_config
#         self._schema      = class_schema
#         self._nodata_value = int(nodata_value)
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @classmethod
#     def from_config(cls, config: Any, class_schema: Any) -> QualityAssessment:
#         """Build from Config and ClassSchema."""
#         nodata = int(getattr(getattr(config, "labels", None), "nodata_value", 255))
#         return cls(
#             quality_config=QualityConfig.from_config(config),
#             class_schema=class_schema,
#             nodata_value=nodata,
#         )

#     def assess(self, morph_result: MorphologyResult) -> QualityResult:
#         """
#         Evaluate the quality of a morphologically processed class map.

#         Args:
#             morph_result: MorphologyResult from MorphologyProcessor.process().

#         Returns:
#             QualityResult with quality_score and is_acceptable flag.
#         """
#         class_map   = morph_result.class_map
#         total_px    = int(class_map.size)
#         issues: list[str] = []

#         # Identify valid pixels: those with a known class ID in the schema.
#         valid_ids  = set(self._schema.class_ids)
#         valid_mask = np.isin(class_map, list(valid_ids))
#         valid_px   = int(valid_mask.sum())

#         unclassified_px   = int((class_map == self._nodata_value).sum())
#         valid_ratio       = valid_px / total_px if total_px > 0 else 0.0
#         unclassified_ratio = unclassified_px / total_px if total_px > 0 else 1.0

#         # Per-class coverage
#         class_fractions: dict[str, float] = {}
#         present_classes = 0
#         for defn in self._schema.classes:
#             count = int(np.sum(class_map == defn.class_id))
#             frac  = count / total_px if total_px > 0 else 0.0
#             class_fractions[defn.name] = frac
#             if count >= self._cfg.min_class_pixels:
#                 present_classes += 1

#         # Quality checks
#         if valid_ratio < self._cfg.min_valid_pixel_ratio:
#             issues.append(
#                 f"valid_pixel_ratio {valid_ratio:.2%} < "
#                 f"threshold {self._cfg.min_valid_pixel_ratio:.2%}"
#             )
#         if unclassified_ratio > self._cfg.max_unclassified_ratio:
#             issues.append(
#                 f"unclassified_ratio {unclassified_ratio:.2%} > "
#                 f"threshold {self._cfg.max_unclassified_ratio:.2%}"
#             )
#         if present_classes == 0:
#             issues.append("no class has sufficient pixel coverage")

#         # Composite quality score: weighted average of sub-scores.
#         valid_score         = min(valid_ratio / max(self._cfg.min_valid_pixel_ratio, 1e-8), 1.0)
#         unclassified_score  = max(
#             1.0 - unclassified_ratio / max(self._cfg.max_unclassified_ratio, 1e-8),
#             0.0,
#         )
#         class_score = min(present_classes / max(len(self._schema.classes), 1), 1.0)

#         quality_score = float(
#             0.5 * valid_score + 0.3 * unclassified_score + 0.2 * class_score
#         )
#         quality_score = round(min(quality_score, 1.0), 4)

#         is_acceptable = (
#             len(issues) == 0 and quality_score >= self._cfg.min_quality_score
#         )

#         self._logger.debug(
#             "QualityAssessment: score=%.2f, valid=%.1f%%, unclassified=%.1f%%, "
#             "classes=%d, acceptable=%s",
#             quality_score, valid_ratio * 100, unclassified_ratio * 100,
#             present_classes, is_acceptable,
#         )

#         return QualityResult(
#             quality_score=quality_score,
#             is_acceptable=is_acceptable,
#             valid_pixel_ratio=valid_ratio,
#             unclassified_ratio=unclassified_ratio,
#             class_pixel_fractions=class_fractions,
#             num_classes_present=present_classes,
#             issues=issues,
#         )