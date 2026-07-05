"""
Conflict resolution for the pseudo-label generation pipeline (Module 9).

When multiple rules assign overlapping pixel proposals, the ConflictResolver
determines the final class assignment and populates the Level 3
resolved_confidence_map on the returned ClassificationResult.

Two strategies are supported:

HIGHEST_CONFIDENCE
    The class with the highest per-pixel evidence wins. This is already
    applied by SpectralClassificationEngine.classify(), so this strategy
    is effectively a pass-through. resolved_confidence_map is set equal
    to confidence_map (same object; no copy).

PRIORITY_ORDER
    Classes are ranked by a configurable priority list. In a tie or where
    multiple rules propose the same pixel, the higher-priority class wins
    regardless of confidence magnitude. resolved_confidence_map is
    recomputed from the rule whose class won at each pixel.

All strategy parameters come from Config. No class ID is hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.labels.contracts import ClassificationResult

__all__ = ["ConflictResolver"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_STRATEGY_HIGHEST_CONFIDENCE: str = "highest_confidence"
_STRATEGY_PRIORITY_ORDER:      str = "priority_order"


class ConflictResolver:
    """
    Resolves per-pixel classification conflicts between multiple rules.

    Also responsible for populating ClassificationResult.resolved_confidence_map
    (Level 3), which ConfidenceEstimator uses for mask-level confidence.

    Args:
        strategy:       "highest_confidence" or "priority_order".
        priority_order: Ordered list of class_ids from highest to lowest
                        priority. Only used when strategy is "priority_order".
        nodata_value:   Sentinel written to pixels where no rule voted.
    """

    def __init__(
        self,
        strategy:       str,
        priority_order:  list[int],
        nodata_value:    int = 255,
    ) -> None:
        if strategy not in (_STRATEGY_HIGHEST_CONFIDENCE, _STRATEGY_PRIORITY_ORDER):
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="conflict_resolution.strategy",
                value=strategy,
                reason=(
                    f"must be '{_STRATEGY_HIGHEST_CONFIDENCE}' or "
                    f"'{_STRATEGY_PRIORITY_ORDER}'"
                ),
            )
        self._strategy       = strategy
        self._priority_order  = list(priority_order)
        self._nodata_value    = int(nodata_value)
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Any) -> ConflictResolver:
        """
        Build a ConflictResolver from Config.

        Reads:
            config.labels.conflict_resolution.strategy
            config.labels.conflict_resolution.*_priority
            config.labels.nodata_value
        """
        labels_cfg = getattr(config, "labels", None)
        cr_cfg     = getattr(labels_cfg, "conflict_resolution", None)

        strategy = str(getattr(cr_cfg, "strategy", _STRATEGY_HIGHEST_CONFIDENCE))
        nodata   = int(getattr(labels_cfg, "nodata_value", 255))

        # Build priority order from individual class priority integers.
        # Lower integer -> higher priority.
        raw: dict[int, int] = {
            int(getattr(cr_cfg, "water_priority",      0)): 1,
            int(getattr(cr_cfg, "vegetation_priority", 1)): 3,
            int(getattr(cr_cfg, "sand_priority",       2)): 2,
            int(getattr(cr_cfg, "background_priority", 3)): 0,
        }
        priority_order = [class_id for _, class_id in sorted(raw.items())]
        return cls(strategy=strategy, priority_order=priority_order, nodata_value=nodata)

    def resolve(self, classification: ClassificationResult) -> ClassificationResult:
        """
        Apply the conflict resolution strategy to a ClassificationResult.

        Populates resolved_confidence_map (Level 3) on the returned object.

        Args:
            classification: Output from SpectralClassificationEngine.classify().

        Returns:
            Updated ClassificationResult with resolved_confidence_map set.
        """
        if self._strategy == _STRATEGY_HIGHEST_CONFIDENCE:
            result = self._resolve_highest_confidence(classification)
        else:
            result = self._resolve_priority_order(classification)

        # print(result.class_map.shape)
        # print(result.nodata_mask.shape)
        # print(result.unclassified_mask.shape)
        # print(result.confidence_map.shape)
        # print(result.resolved_confidence_map.shape)

        self._logger.debug(
            "Conflict resolved (%s): unique classes=%s",
            self._strategy,
            np.unique(
                result.class_map[~result.nodata_mask & ~result.unclassified_mask]
            ).tolist(),
        )
        return result

    # ------------------------------------------------------------------
    # Private strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_highest_confidence(
        classification: ClassificationResult,
    ) -> ClassificationResult:
        """
        Highest-confidence strategy.

        The classifier already applied this; resolved_confidence_map equals
        confidence_map (same object, no copy required).
        """
        classification.resolved_confidence_map = classification.confidence_map
        return classification

    def _resolve_priority_order(
        self,
        classification: ClassificationResult,
    ) -> ClassificationResult:
        """
        Priority-order strategy.

        Applies per-class pixel masks in reverse priority order so that the
        highest-priority class overwrites lower ones. Then recomputes
        resolved_confidence_map from the winning class at each pixel.
        """
        h, w      = classification.class_map.shape
        class_map = np.zeros((h, w), dtype=np.uint8)

        # Apply in REVERSE priority order: highest priority class is applied
        # last and therefore overwrites all others.
        for class_id in reversed(self._priority_order):
            for rule_result in classification.rule_results:
                if rule_result.class_id == class_id:
                    class_map[rule_result.pixel_mask] = class_id
                    break

        # Recompute Level 3 confidence: evidence of the rule that won at each pixel.
        resolved_confidence = np.zeros((h, w), dtype=np.float32)
        for rule_result in classification.rule_results:
            winner = class_map == rule_result.class_id
            resolved_confidence[winner] = rule_result.confidence[winner]

        unclassified = resolved_confidence == 0.0

        return ClassificationResult(
            class_map=class_map,
            confidence_map=classification.confidence_map,
            rule_results=classification.rule_results,
            unclassified_mask=unclassified,
            nodata_mask=classification.nodata_mask,
            resolved_confidence_map=resolved_confidence,
        )


# """
# Conflict resolution for the pseudo-label generation pipeline (Module 9).

# When multiple rules assign overlapping pixel proposals, the ConflictResolver
# determines the final class assignment. Supports two strategies:

# HIGHEST_CONFIDENCE:
#     The class with the highest per-pixel confidence wins.
#     Simple and effective; confidence weights implicitly encode priority.
#     Recommended for well-calibrated rules.

# PRIORITY_ORDER:
#     Classes are ranked by a configurable priority list. In a tie, the
#     higher-priority class wins regardless of confidence differences.
#     Useful when one class should always dominate (e.g. water over sand in
#     active river channels where MNDWI is unreliable).

# The ConflictResolver also fills remaining unclassified pixels (where all
# rules returned zero confidence) with the nodata sentinel value so that
# downstream quality assessment can detect them.

# All strategy parameters come from Config.
# """

# from __future__ import annotations

# import logging
# from typing import Any

# import numpy as np

# from src.labels.contracts import ClassificationResult

# __all__ = ["ConflictResolver"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)

# _STRATEGY_HIGHEST_CONFIDENCE = "highest_confidence"
# _STRATEGY_PRIORITY_ORDER      = "priority_order"


# class ConflictResolver:
#     """
#     Resolves per-pixel classification conflicts between multiple rules.

#     Args:
#         strategy:       "highest_confidence" or "priority_order".
#         priority_order: Ordered list of class_ids from highest to lowest
#                         priority. Only used when strategy is "priority_order".
#         nodata_value:   Sentinel written to pixels where no rule voted.
#     """

#     def __init__(
#         self,
#         strategy:       str,
#         priority_order:  list[int],
#         nodata_value:    int = 255,
#     ) -> None:
#         if strategy not in (_STRATEGY_HIGHEST_CONFIDENCE, _STRATEGY_PRIORITY_ORDER):
#             from src.core.exceptions import InvalidValueError
#             raise InvalidValueError(
#                 field="conflict_resolution.strategy",
#                 value=strategy,
#                 reason=(
#                     f"must be '{_STRATEGY_HIGHEST_CONFIDENCE}' or "
#                     f"'{_STRATEGY_PRIORITY_ORDER}'"
#                 ),
#             )
#         self._strategy       = strategy
#         self._priority_order  = list(priority_order)
#         self._nodata_value    = int(nodata_value)
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @classmethod
#     def from_config(cls, config: Any) -> ConflictResolver:
#         """
#         Build a ConflictResolver from Config.

#         Reads:
#             config.labels.conflict_resolution.strategy
#             config.labels.conflict_resolution.*_priority
#             config.labels.nodata_value
#         """
#         labels_cfg = getattr(config, "labels", None)
#         cr_cfg     = getattr(labels_cfg, "conflict_resolution", None)

#         strategy  = str(getattr(cr_cfg, "strategy", _STRATEGY_HIGHEST_CONFIDENCE))
#         nodata    = int(getattr(labels_cfg, "nodata_value", 255))

#         # Build priority order from individual class priority values.
#         priority_map = {
#             int(getattr(cr_cfg, "water_priority",      0)): 1,
#             int(getattr(cr_cfg, "vegetation_priority", 1)): 3,
#             int(getattr(cr_cfg, "sand_priority",       2)): 2,
#             int(getattr(cr_cfg, "background_priority", 3)): 0,
#         }
#         priority_order = [
#             class_id for _, class_id in sorted(priority_map.items())
#         ]
#         return cls(strategy=strategy, priority_order=priority_order, nodata_value=nodata)

#     def resolve(self, classification: ClassificationResult) -> ClassificationResult:
#         """
#         Apply the conflict resolution strategy to a ClassificationResult.

#         The ClassificationResult from SpectralClassificationEngine already
#         uses highest-confidence assignment. This method post-processes
#         it according to the configured strategy.

#         Args:
#             classification: Output from SpectralClassificationEngine.classify().

#         Returns:
#             Updated ClassificationResult with resolved class_map.
#         """
#         if self._strategy == _STRATEGY_HIGHEST_CONFIDENCE:
#             result = self._resolve_highest_confidence(classification)
#         else:
#             result = self._resolve_priority_order(classification)

#         self._logger.debug(
#             "Conflict resolved (%s): unique classes=%s",
#             self._strategy,
#             np.unique(result.class_map[~result.nodata_mask & ~result.unclassified_mask]).tolist(),
#         )
#         return result

#     # ------------------------------------------------------------------
#     # Private strategies
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _resolve_highest_confidence(
#         classification: ClassificationResult,
#     ) -> ClassificationResult:
#         """
#         Highest-confidence strategy: already applied by the classifier.
#         Just mark fully unclassified pixels (zero confidence everywhere).
#         """
#         return classification

#     def _resolve_priority_order(
#         self,
#         classification: ClassificationResult,
#     ) -> ClassificationResult:
#         """
#         Priority-order strategy: among pixels where multiple rules overlap,
#         enforce class precedence based on the configured priority_order.

#         Applies per-rule masks in reverse priority order (lowest first),
#         so higher-priority classes overwrite lower ones.
#         """
#         h, w      = classification.class_map.shape
#         class_map = np.zeros((h, w), dtype=np.uint8)

#         # Apply in REVERSE priority order so highest priority overwrites last.
#         for class_id in reversed(self._priority_order):
#             for result in classification.rule_results:
#                 if result.class_id == class_id:
#                     class_map[result.pixel_mask] = class_id
#                     break

#         # Rebuild confidence_map: use the confidence of the winning class.
#         confidence_map = np.zeros((h, w), dtype=np.float32)
#         for result in classification.rule_results:
#             winner = class_map == result.class_id
#             confidence_map[winner] = result.confidence[winner]

#         unclassified = confidence_map == 0.0

#         return ClassificationResult(
#             class_map=class_map,
#             confidence_map=confidence_map,
#             rule_results=classification.rule_results,
#             unclassified_mask=unclassified,
#             nodata_mask=classification.nodata_mask,
#         )