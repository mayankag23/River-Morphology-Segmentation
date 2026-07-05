"""
Confidence estimation for the pseudo-label generation pipeline (Module 9).

ConfidenceEstimator computes three confidence metrics for a generated mask:

    pixel_confidence:   Per-pixel confidence map (Level 3 -- resolved).
    mask_confidence:    Scalar mask-level confidence = mean of valid pixel
                        resolved confidences.
    agreement_score:    Fraction of valid pixels where two or more rules
                        proposed the same winning class (higher agreement
                        indicates the classification is more trustworthy).

The estimator uses ClassificationResult.resolved_confidence_map (Level 3)
when it is populated by ConflictResolver; it falls back to confidence_map
(Level 2) when resolved_confidence_map is None, ensuring backward
compatibility with code that bypasses ConflictResolver.

Masks whose mask_confidence is below config.labels.confidence.min_mask_confidence
are flagged but NOT rejected here -- rejection is handled by LabelManager,
which also incorporates QualityAssessment results.

All thresholds come exclusively from Config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.labels.contracts import ClassificationResult, ConfidenceResult

__all__ = ["ConfidenceConfig", "ConfidenceEstimator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    Immutable confidence estimation thresholds (all from Config).

    Attributes:
        min_pixel_confidence:  Pixels below this are considered low-confidence.
        min_mask_confidence:    Masks below this threshold are flagged.
    """

    min_pixel_confidence: float = 0.20
    min_mask_confidence:  float = 0.30

    @classmethod
    def from_config(cls, config: Any) -> ConfidenceConfig:
        """Build ConfidenceConfig from config.labels.confidence."""
        c = getattr(getattr(config, "labels", None), "confidence", None)
        if c is None:
            return cls()
        return cls(
            min_pixel_confidence = float(getattr(c, "min_pixel_confidence", 0.20)),
            min_mask_confidence  = float(getattr(c, "min_mask_confidence",  0.30)),
        )


class ConfidenceEstimator:
    """
    Computes per-pixel and mask-level confidence for a ClassificationResult.

    Args:
        conf_config:  ConfidenceConfig with threshold settings.
        nodata_value: Sentinel for pixels with no valid classification.
    """

    def __init__(
        self,
        conf_config:  ConfidenceConfig,
        nodata_value: int = 255,
    ) -> None:
        self._cfg          = conf_config
        self._nodata_value = int(nodata_value)
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Any) -> ConfidenceEstimator:
        """Build from Config."""
        nodata = int(getattr(getattr(config, "labels", None), "nodata_value", 255))
        return cls(
            conf_config=ConfidenceConfig.from_config(config),
            nodata_value=nodata,
        )

    @property
    def min_mask_confidence(self) -> float:
        """Public access to the mask-level confidence threshold."""
        return self._cfg.min_mask_confidence

    def estimate(
        self,
        classification: ClassificationResult,
        class_map:      np.ndarray,
    ) -> ConfidenceResult:
        """
        Compute confidence metrics for a classification result.

        Uses Level 3 resolved_confidence_map when available; falls back to
        Level 2 confidence_map when resolved_confidence_map is None.

        Args:
            classification: ClassificationResult from ConflictResolver.resolve().
            class_map:       Final (H, W) uint8 class map after morphology.

        Returns:
            ConfidenceResult with pixel_confidence (Level 3), mask_confidence,
            agreement_score, and component_scores for diagnostics.
        """
        # Select Level 3 source (preferred) or fall back to Level 2.
        if classification.resolved_confidence_map is not None:
            pixel_confidence = classification.resolved_confidence_map.copy()
        else:
            pixel_confidence = classification.confidence_map.copy()

        # Valid pixels: classified (not nodata) and not in the nodata input mask.
        valid_mask = (class_map != self._nodata_value) & ~classification.nodata_mask

        # Mask confidence = mean over valid pixels.
        valid_confs = pixel_confidence[valid_mask]
        if len(valid_confs) == 0:
            mask_confidence = 0.0
        else:
            mask_confidence = float(np.mean(valid_confs))

        # Agreement score: fraction of valid pixels where >= 2 rules
        # proposed the same winning class.
        total_valid = int(valid_mask.sum())
        if total_valid == 0 or len(classification.rule_results) < 2:
            agreement_score = 0.0
        else:
            winner_class = class_map[valid_mask]
            agree_count  = 0
            rule_results = classification.rule_results
            for r_idx in range(len(rule_results)):
                r_result = rule_results[r_idx]
                for s_idx in range(r_idx + 1, len(rule_results)):
                    s_result = rule_results[s_idx]
                    # Both rules voted (>0 confidence) for the same class
                    # that actually won.
                    agree = (
                        (r_result.confidence[valid_mask] > 0)
                        & (s_result.confidence[valid_mask] > 0)
                        & (r_result.class_id == s_result.class_id)
                        & (r_result.class_id == winner_class)
                    )
                    agree_count = max(agree_count, int(agree.sum()))
            agreement_score = agree_count / total_valid

        mask_confidence = round(mask_confidence, 4)
        agreement_score = round(agreement_score, 4)

        component_scores: dict[str, float] = {
            "mask_confidence":  mask_confidence,
            "agreement_score":  agreement_score,
        }

        self._logger.debug(
            "ConfidenceEstimator: mask_confidence=%.3f, agreement=%.3f",
            mask_confidence, agreement_score,
        )
        return ConfidenceResult(
            pixel_confidence=pixel_confidence,
            mask_confidence=mask_confidence,
            agreement_score=agreement_score,
            component_scores=component_scores,
        )

# """
# Confidence estimation for the pseudo-label generation pipeline (Module 9).

# ConfidenceEstimator computes:
#     pixel_confidence:  Per-pixel confidence map derived from the winning
#                        rule's confidence score.
#     mask_confidence:   Scalar mask-level confidence = mean of valid pixel
#                        confidences.
#     agreement_score:   Fraction of valid pixels where two or more rules
#                        proposed the same winning class (higher agreement
#                        means the classification is more trustworthy).

# Masks whose mask_confidence is below config.labels.confidence.min_mask_confidence
# are flagged but NOT rejected here (rejection is handled by QualityAssessment).

# All thresholds come from Config.
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from typing import Any

# import numpy as np

# from src.labels.contracts import ClassificationResult, ConfidenceResult

# __all__ = ["ConfidenceConfig", "ConfidenceEstimator"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)


# @dataclass(frozen=True)
# class ConfidenceConfig:
#     """
#     Immutable confidence estimation thresholds.

#     Attributes:
#         min_pixel_confidence:  Pixels below this are flagged as low-confidence.
#         min_mask_confidence:    Masks below this confidence are flagged.
#     """

#     min_pixel_confidence: float = 0.20
#     min_mask_confidence:  float = 0.30

#     @classmethod
#     def from_config(cls, config: Any) -> ConfidenceConfig:
#         """Build ConfidenceConfig from config.labels.confidence."""
#         c = getattr(getattr(config, "labels", None), "confidence", None)
#         if c is None:
#             return cls()
#         return cls(
#             min_pixel_confidence = float(getattr(c, "min_pixel_confidence", 0.20)),
#             min_mask_confidence  = float(getattr(c, "min_mask_confidence",  0.30)),
#         )


# class ConfidenceEstimator:
#     """
#     Computes per-pixel and mask-level confidence for a ClassificationResult.

#     Args:
#         conf_config:  ConfidenceConfig with threshold settings.
#         nodata_value: Sentinel for pixels with no valid classification.
#     """

#     def __init__(
#         self,
#         conf_config:  ConfidenceConfig,
#         nodata_value: int = 255,
#     ) -> None:
#         self._cfg         = conf_config
#         self._nodata_value = int(nodata_value)
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @classmethod
#     def from_config(cls, config: Any) -> ConfidenceEstimator:
#         """Build from Config."""
#         nodata = int(getattr(getattr(config, "labels", None), "nodata_value", 255))
#         return cls(
#             conf_config=ConfidenceConfig.from_config(config),
#             nodata_value=nodata,
#         )

#     def estimate(
#         self,
#         classification: ClassificationResult,
#         class_map:      np.ndarray,
#     ) -> ConfidenceResult:
#         """
#         Compute confidence metrics for a classification result.

#         Args:
#             classification: ClassificationResult from ConflictResolver.
#             class_map:       Final (H, W) uint8 class map (after morphology).

#         Returns:
#             ConfidenceResult with pixel_confidence, mask_confidence,
#             and agreement_score.
#         """
#         pixel_confidence = classification.confidence_map.copy()

#         # Mask out nodata and unclassified pixels.
#         valid_mask = (class_map != self._nodata_value) & ~classification.nodata_mask

#         # Mask confidence = mean over valid pixels.
#         valid_confs = pixel_confidence[valid_mask]
#         if len(valid_confs) == 0:
#             mask_confidence = 0.0
#         else:
#             mask_confidence = float(np.mean(valid_confs))

#         # Agreement score: fraction of valid pixels where >= 2 rules
#         # proposed the same class (i.e. where the winning class was
#         # not the ONLY class proposed).
#         total_valid = int(valid_mask.sum())
#         if total_valid == 0 or len(classification.rule_results) < 2:
#             agreement_score = 0.0
#         else:
#             # Count pixels where at least 2 rules have >0 confidence for the winner.
#             winner_class = class_map[valid_mask]
#             agree_count  = 0
#             for r_idx in range(len(classification.rule_results)):
#                 rule_r = classification.rule_results[r_idx]
#                 for s_idx in range(r_idx + 1, len(classification.rule_results)):
#                     rule_s = classification.rule_results[s_idx]
#                     # Both rules vote for the same class at that pixel
#                     agree = (
#                         (rule_r.confidence[valid_mask] > 0)
#                         & (rule_s.confidence[valid_mask] > 0)
#                         & (rule_r.class_id == rule_s.class_id)
#                         & (rule_r.class_id == winner_class)
#                     )
#                     agree_count = max(agree_count, int(agree.sum()))
#             agreement_score = agree_count / total_valid

#         self._logger.debug(
#             "Confidence: mask=%.3f, agreement=%.3f",
#             mask_confidence, agreement_score,
#         )
#         return ConfidenceResult(
#             pixel_confidence=pixel_confidence,
#             mask_confidence=round(mask_confidence, 4),
#             agreement_score=round(agreement_score, 4),
#         )