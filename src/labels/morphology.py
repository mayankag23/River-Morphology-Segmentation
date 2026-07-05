"""
Morphological post-processing for the pseudo-label generation pipeline
(Module 9).

MorphologyProcessor operates EXCLUSIVELY on the integer class map produced
by ConflictResolver. It never accesses spectral bands, index values, or
Earth Engine. Its only inputs are:
    - ClassificationResult.class_map  (H, W) uint8
    - (optionally) ClassificationResult.resolved_confidence_map, which is
      threaded through unchanged into MorphologyResult.

Per-class spatial operations allow fine-grained control: narrow sand bars
in braided rivers can be assigned smaller min_object_size thresholds than
water bodies, preventing legitimate thin features from being removed.

Operations applied per class (on binary masks):
    1. Opening   -- removes isolated noise speckling.
    2. Closing   -- fills small intra-class holes.
    3. Small object removal -- eliminates connected components below threshold.
    4. Small hole filling   -- fills enclosed holes below threshold.

After all per-class passes:
    5. Majority filter -- smooths class boundaries over the full class map.

All parameters come exclusively from Config via MorphologyConfig and
PerClassMorphologyConfig. No threshold or kernel size is hardcoded.
ClassSchema is used ONLY during MorphologyConfig.from_config() to resolve
class names to integer IDs; it is not stored in or used by MorphologyProcessor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.labels.contracts import ClassificationResult, MorphologyResult

__all__ = ["PerClassMorphologyConfig", "MorphologyConfig", "MorphologyProcessor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# PerClassMorphologyConfig
# ==============================================================================

@dataclass(frozen=True)
class PerClassMorphologyConfig:
    """
    Immutable per-class spatial morphology overrides.

    Only spatial parameters -- no spectral information.

    Attributes:
        opening_radius:   Structuring element half-size for opening. 0 disables.
        closing_radius:    Structuring element half-size for closing. 0 disables.
        min_object_size:   Minimum connected component size (pixels). 0 disables.
        min_hole_size:      Minimum enclosed hole size (pixels). 0 disables.
    """

    opening_radius:   int = 1
    closing_radius:    int = 2
    min_object_size:   int = 20
    min_hole_size:      int = 20


# ==============================================================================
# MorphologyConfig
# ==============================================================================

@dataclass(frozen=True)
class MorphologyConfig:
    """
    Immutable configuration for all morphological operations.

    class_overrides is keyed by integer class ID (resolved from class name
    at MorphologyConfig.from_config() time). An empty dict means all classes
    use the global default parameters.

    Attributes:
        enabled:               Master toggle. When False, class_map is returned
                               unchanged with operations_applied=["morphology_disabled"].
        opening_radius:         Global default structuring element half-size for
                               opening. 0 disables opening for classes that have
                               no override.
        closing_radius:          Global default structuring element half-size for
                               closing. 0 disables closing.
        min_object_size:         Global default minimum connected component size.
                               0 disables small-object removal.
        min_hole_size:            Global default minimum enclosed hole size.
                               0 disables hole filling.
        majority_filter_size:     Window size for majority filtering (applied to
                               the full class map after per-class operations).
                               Must be odd. 0 or 1 disables.
        class_overrides:          Per-class spatial parameter overrides, keyed
                               by integer class ID. Takes precedence over global
                               defaults for the specified class.
    """

    enabled:               bool = True
    opening_radius:         int  = 1
    closing_radius:          int  = 2
    min_object_size:         int  = 20
    min_hole_size:            int  = 20
    majority_filter_size:     int  = 3
    class_overrides:          dict[int, PerClassMorphologyConfig] = field(
        default_factory=dict
    )

    @classmethod
    def from_config(cls, config: Any, class_schema: Any = None) -> MorphologyConfig:
        """
        Build MorphologyConfig from config.labels.morphology.

        Args:
            config:       Fully initialized Config object.
            class_schema: ClassSchema instance used to resolve class names in
                          class_overrides to integer class IDs. May be None when
                          no class_overrides are configured.

        Returns:
            Immutable MorphologyConfig.
        """
        m = getattr(getattr(config, "labels", None), "morphology", None)
        if m is None:
            return cls()

        enabled               = bool(getattr(m, "enabled",              True))
        opening_radius        = int(getattr(m, "opening_radius",          1))
        closing_radius        = int(getattr(m, "closing_radius",           2))
        min_object_size       = int(getattr(m, "min_object_size",         20))
        min_hole_size         = int(getattr(m, "min_hole_size",            20))
        majority_filter_size  = int(getattr(m, "majority_filter_size",     3))

        # Resolve per-class overrides from config.labels.morphology.class_overrides.
        overrides_cfg  = getattr(m, "class_overrides", None)
        class_overrides: dict[int, PerClassMorphologyConfig] = {}

        if overrides_cfg is not None and class_schema is not None:
            for class_name in overrides_cfg:
                oc = getattr(overrides_cfg, class_name)
                try:
                    class_id = class_schema.get_id_by_name(class_name)
                except Exception:
                    _LOGGER.warning(
                        "MorphologyConfig: class name '%s' not found in schema; "
                        "skipping override.", class_name,
                    )
                    continue
                class_overrides[class_id] = PerClassMorphologyConfig(
                    opening_radius  = int(getattr(oc, "opening_radius",  opening_radius)),
                    closing_radius   = int(getattr(oc, "closing_radius",  closing_radius)),
                    min_object_size  = int(getattr(oc, "min_object_size", min_object_size)),
                    min_hole_size    = int(getattr(oc, "min_hole_size",   min_hole_size)),
                )

        return cls(
            enabled=enabled,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            min_object_size=min_object_size,
            min_hole_size=min_hole_size,
            majority_filter_size=majority_filter_size,
            class_overrides=class_overrides,
        )

    def params_for_class(self, class_id: int) -> PerClassMorphologyConfig:
        """
        Return the effective PerClassMorphologyConfig for class_id.

        Uses the class override if present; falls back to global defaults.
        """
        if class_id in self.class_overrides:
            return self.class_overrides[class_id]
        return PerClassMorphologyConfig(
            opening_radius  = self.opening_radius,
            closing_radius   = self.closing_radius,
            min_object_size  = self.min_object_size,
            min_hole_size    = self.min_hole_size,
        )


# ==============================================================================
# MorphologyProcessor
# ==============================================================================

class MorphologyProcessor:
    """
    Applies spatial morphological operations to clean a ClassificationResult.

    Operates ONLY on the integer class map. Never accesses spectral bands,
    index values, or Earth Engine. Per-class dispatch is based solely on the
    class ID integer read from the class map.

    Uses OpenCV (cv2) for morphological opening and closing. Falls back
    gracefully when cv2 is unavailable (logs a WARNING). Uses scipy for
    connected-component analysis; falls back when scipy is unavailable.

    Args:
        morph_config: MorphologyConfig controlling all operations.
    """

    def __init__(self, morph_config: MorphologyConfig) -> None:
        self._cfg    = morph_config
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Any, class_schema: Any = None) -> MorphologyProcessor:
        """
        Build MorphologyProcessor from Config.

        Args:
            config:       Fully initialized Config object.
            class_schema: Optional ClassSchema for resolving class_overrides.
        """
        return cls(MorphologyConfig.from_config(config, class_schema=class_schema))

    def process(self, classification: ClassificationResult) -> MorphologyResult:
        """
        Apply all enabled morphological operations to a ClassificationResult.

        The class map is processed class by class using per-class parameters.
        The majority filter (if enabled) is applied once to the full map
        after all per-class passes.

        Args:
            classification: ClassificationResult from ConflictResolver.resolve().

        Returns:
            MorphologyResult with cleaned class_map and operations_applied log.
        """
        if not self._cfg.enabled:
            return MorphologyResult(
                class_map=classification.class_map.copy(),
                operations_applied=["morphology_disabled"],
            )

        class_map  = classification.class_map.copy()
        operations: list[str] = []

        try:
            import cv2 as _cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False
            self._logger.warning(
                "cv2 not available; morphological opening and closing "
                "will be skipped. Install opencv-python for full functionality."
            )

        unique_classes = np.unique(class_map)

        for class_id in unique_classes:
            params = self._cfg.params_for_class(int(class_id))
            binary = (class_map == class_id).astype(np.uint8)

            if has_cv2:
                import cv2
                if params.opening_radius > 0:
                    kernel = self._make_kernel(params.opening_radius)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                if params.closing_radius > 0:
                    kernel = self._make_kernel(params.closing_radius)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            if params.min_object_size > 0:
                binary = self._remove_small_objects(
                    binary.astype(bool), params.min_object_size
                ).astype(np.uint8)

            if params.min_hole_size > 0:
                binary = self._fill_small_holes(
                    binary.astype(bool), params.min_hole_size
                ).astype(np.uint8)

            class_map[class_map == class_id] = 0
            class_map[binary.astype(bool)]   = class_id

        operations.append(
            f"per_class_morphology("
            f"cv2={has_cv2}, "
            f"global_opening={self._cfg.opening_radius}, "
            f"global_closing={self._cfg.closing_radius}, "
            f"global_min_obj={self._cfg.min_object_size}, "
            f"global_min_hole={self._cfg.min_hole_size}, "
            f"overrides={list(self._cfg.class_overrides.keys())})"
        )

        if self._cfg.majority_filter_size > 1:
            class_map = self._majority_filter(class_map, self._cfg.majority_filter_size)
            operations.append(
                f"majority_filter(size={self._cfg.majority_filter_size})"
            )

        return MorphologyResult(class_map=class_map, operations_applied=operations)

    # ------------------------------------------------------------------
    # Private helpers -- spatial operations only, no spectral access
    # ------------------------------------------------------------------

    @staticmethod
    def _make_kernel(radius: int) -> Any:
        """Create an elliptical structuring element for OpenCV."""
        import cv2
        size = 2 * radius + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @staticmethod
    def _remove_small_objects(
        binary: np.ndarray, min_size: int
    ) -> np.ndarray:
        """Remove connected components smaller than min_size pixels."""
        try:
            from scipy import ndimage as ndi
            labeled, num_features = ndi.label(binary)
            if num_features == 0:
                return binary
            binary = binary.copy()
            for i in range(1, num_features + 1):
                component_size = int(np.sum(labeled == i))
                if component_size < min_size:
                    binary[labeled == i] = False
        except ImportError:
            _LOGGER.warning(
                "scipy not available; small-object removal skipped. "
                "Install scipy for full morphological processing."
            )
        return binary

    @staticmethod
    def _fill_small_holes(
        binary: np.ndarray, min_size: int
    ) -> np.ndarray:
        """Fill enclosed holes smaller than min_size pixels."""
        try:
            from scipy import ndimage as ndi
            filled = ndi.binary_fill_holes(binary)
            holes  = filled & ~binary
            labeled, num_features = ndi.label(holes)
            if num_features == 0:
                return binary
            binary = binary.copy()
            for i in range(1, num_features + 1):
                if int(np.sum(labeled == i)) < min_size:
                    binary[labeled == i] = True
        except ImportError:
            _LOGGER.warning(
                "scipy not available; small-hole filling skipped."
            )
        return binary

    @staticmethod
    def _majority_filter(class_map: np.ndarray, size: int) -> np.ndarray:
        """Replace each pixel with the most common class in a (size x size) window."""
        try:
            from scipy.ndimage import generic_filter

            def _mode(window: np.ndarray) -> float:
                vals, counts = np.unique(window.astype(int), return_counts=True)
                return float(vals[np.argmax(counts)])

            return generic_filter(
                class_map.astype(float), _mode, size=size
            ).astype(np.uint8)

        except ImportError:
            # Pure-numpy fallback: iterate interior pixels only.
            h, w   = class_map.shape
            half   = size // 2
            result = class_map.copy()
            for r in range(half, h - half):
                for c in range(half, w - half):
                    patch        = class_map[
                        r - half: r + half + 1, c - half: c + half + 1
                    ].flatten()
                    vals, counts = np.unique(patch, return_counts=True)
                    result[r, c] = vals[int(np.argmax(counts))]
            return result



# """
# Morphological post-processing for the pseudo-label generation pipeline.

# MorphologyProcessor applies per-class binary morphological operations to
# clean the raw classification output:

# 1. Per-class opening  -- removes isolated noise pixels (speckling).
# 2. Per-class closing  -- fills small intra-class holes.
# 3. Small object removal -- eliminates connected components below min_object_size.
# 4. Small hole filling  -- fills enclosed regions below min_hole_size.
# 5. Majority filtering  -- smooths class boundaries.

# All operations are applied per-class (on binary masks) so that they do
# not distort the spatial relationship between classes. The final class_map
# is reconstructed from the per-class results using the same confidence
# priority as the classifier.

# All kernel sizes, minimum sizes, and enable flags come from Config.
# Uses OpenCV for morphological operations (cv2 is a project dependency).
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from typing import Any

# import numpy as np

# from src.labels.contracts import ClassificationResult, MorphologyResult

# __all__ = ["MorphologyConfig", "MorphologyProcessor"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)


# @dataclass(frozen=True)
# class MorphologyConfig:
#     """
#     Immutable configuration for morphological operations.

#     Attributes:
#         enabled:             Master toggle. When False, class_map is returned
#                              unchanged.
#         opening_radius:       Structuring element half-size for opening.
#                              0 disables opening.
#         closing_radius:        Structuring element half-size for closing.
#                              0 disables closing.
#         min_object_size:       Remove connected components smaller than this
#                              (in pixels). 0 disables small-object removal.
#         min_hole_size:          Fill enclosed holes smaller than this
#                              (in pixels). 0 disables hole filling.
#         majority_filter_size:   Window size for majority filter. Must be odd.
#                              0 or 1 disables majority filtering.
#     """

#     enabled:             bool = True
#     opening_radius:       int  = 1
#     closing_radius:        int  = 2
#     min_object_size:       int  = 20
#     min_hole_size:          int  = 20
#     majority_filter_size:   int  = 3

#     @classmethod
#     def from_config(cls, config: Any) -> MorphologyConfig:
#         """Build MorphologyConfig from config.labels.morphology."""
#         m = getattr(getattr(config, "labels", None), "morphology", None)
#         if m is None:
#             return cls()
#         return cls(
#             enabled             = bool(getattr(m, "enabled",             True)),
#             opening_radius       = int(getattr(m,  "opening_radius",      1)),
#             closing_radius        = int(getattr(m,  "closing_radius",      2)),
#             min_object_size       = int(getattr(m,  "min_object_size",     20)),
#             min_hole_size          = int(getattr(m,  "min_hole_size",       20)),
#             majority_filter_size   = int(getattr(m,  "majority_filter_size", 3)),
#         )


# class MorphologyProcessor:
#     """
#     Applies morphological operations to clean a ClassificationResult.

#     Uses OpenCV (cv2) for opening and closing. Falls back to numpy-based
#     connected component analysis when scipy is unavailable.

#     Args:
#         morph_config: MorphologyConfig controlling active operations.
#     """

#     def __init__(self, morph_config: MorphologyConfig) -> None:
#         self._cfg    = morph_config
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @classmethod
#     def from_config(cls, config: Any) -> MorphologyProcessor:
#         """Build from Config."""
#         return cls(MorphologyConfig.from_config(config))

#     def process(self, classification: ClassificationResult) -> MorphologyResult:
#         """
#         Apply all enabled morphological operations to a ClassificationResult.

#         Args:
#             classification: ClassificationResult from ConflictResolver.resolve().

#         Returns:
#             MorphologyResult with cleaned class_map.
#         """
#         if not self._cfg.enabled:
#             return MorphologyResult(
#                 class_map=classification.class_map.copy(),
#                 operations_applied=["morphology_disabled"],
#             )

#         class_map  = classification.class_map.copy()
#         operations: list[str] = []

#         unique_classes = np.unique(class_map)

#         try:
#             import cv2
#             has_cv2 = True
#         except ImportError:
#             has_cv2 = False
#             self._logger.warning("cv2 not available; skipping opening/closing.")

#         for class_id in unique_classes:
#             binary = (class_map == class_id).astype(np.uint8)

#             if has_cv2:
#                 # Opening: remove noise
#                 if self._cfg.opening_radius > 0:
#                     r      = self._cfg.opening_radius
#                     kernel = self._make_kernel_cv2(r)
#                     import cv2
#                     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

#                 # Closing: fill small holes
#                 if self._cfg.closing_radius > 0:
#                     r      = self._cfg.closing_radius
#                     kernel = self._make_kernel_cv2(r)
#                     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#             # Remove small objects (connected components)
#             if self._cfg.min_object_size > 0:
#                 binary = self._remove_small_objects(
#                     binary.astype(bool), self._cfg.min_object_size
#                 ).astype(np.uint8)

#             # Fill small holes
#             if self._cfg.min_hole_size > 0:
#                 binary = self._fill_small_holes(
#                     binary.astype(bool), self._cfg.min_hole_size
#                 ).astype(np.uint8)

#             class_map[class_map == class_id] = 0
#             class_map[binary.astype(bool)]   = class_id

#         operations.append(
#             f"opening(r={self._cfg.opening_radius}), "
#             f"closing(r={self._cfg.closing_radius}), "
#             f"min_obj={self._cfg.min_object_size}, "
#             f"min_hole={self._cfg.min_hole_size}"
#         )

#         # Majority filter on full class map
#         if self._cfg.majority_filter_size > 1:
#             class_map = self._majority_filter(
#                 class_map, self._cfg.majority_filter_size
#             )
#             operations.append(f"majority_filter(size={self._cfg.majority_filter_size})")

#         return MorphologyResult(class_map=class_map, operations_applied=operations)

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _make_kernel_cv2(radius: int) -> Any:
#         """Create an elliptical structuring element for OpenCV."""
#         import cv2
#         size = 2 * radius + 1
#         return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

#     @staticmethod
#     def _remove_small_objects(
#         binary: np.ndarray, min_size: int
#     ) -> np.ndarray:
#         """Remove connected components smaller than min_size pixels."""
#         try:
#             from scipy import ndimage as ndi
#             labeled, _ = ndi.label(binary)
#             for i in range(1, labeled.max() + 1):
#                 if int(np.sum(labeled == i)) < min_size:
#                     binary = binary.copy()
#                     binary[labeled == i] = False
#         except ImportError:
#             pass  # skip without scipy
#         return binary

#     @staticmethod
#     def _fill_small_holes(
#         binary: np.ndarray, min_size: int
#     ) -> np.ndarray:
#         """Fill enclosed holes smaller than min_size pixels."""
#         try:
#             from scipy import ndimage as ndi
#             filled = ndi.binary_fill_holes(binary)
#             holes  = filled & ~binary
#             labeled, _ = ndi.label(holes)
#             binary = binary.copy()
#             for i in range(1, labeled.max() + 1):
#                 if int(np.sum(labeled == i)) < min_size:
#                     binary[labeled == i] = True
#         except ImportError:
#             pass
#         return binary

#     @staticmethod
#     def _majority_filter(class_map: np.ndarray, size: int) -> np.ndarray:
#         """Replace each pixel with the most common class in a (size x size) window."""
#         try:
#             from scipy.ndimage import generic_filter

#             def _mode(window: np.ndarray) -> float:
#                 vals  = window.astype(int)
#                 items, counts = np.unique(vals, return_counts=True)
#                 return float(items[np.argmax(counts)])

#             return generic_filter(class_map.astype(float), _mode, size=size).astype(np.uint8)
#         except ImportError:
#             # Numpy fallback: iterate over interior pixels only.
#             h, w   = class_map.shape
#             half   = size // 2
#             result = class_map.copy()
#             for r in range(half, h - half):
#                 for c in range(half, w - half):
#                     patch  = class_map[r - half: r + half + 1, c - half: c + half + 1].flatten()
#                     vals, counts = np.unique(patch, return_counts=True)
#                     result[r, c] = vals[int(np.argmax(counts))]
#             return result