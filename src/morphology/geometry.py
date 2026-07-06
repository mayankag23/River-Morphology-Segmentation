"""
River geometry analysis for Module 17 (src.morphology package).

GeometryAnalyzer produces two levels of output:

1. Per-class generic ConnectedRegionStats:
   Object-level statistics for every connected region of each class.
   These are intentionally generic — the terms "island", "bar", "patch"
   are NOT used here. Downstream modules derive those concepts.

2. Aggregated ClassRegionMetrics per class:
   region_count, largest region, mean/std region size, fragmentation index,
   estimated width of the largest region.

3. Optional shape descriptors (config.compute_shape_descriptors=True):
   perimeter, compactness, elongation, aspect_ratio — all per region.

Connectivity rules
------------------
- Water:      4-connectivity (edge-adjacent pixels only) — standard for
              hydrological connectivity (no diagonal water links).
- Sand and vegetation: 8-connectivity (diagonal OK) — broader connectivity
              for sediment bars and vegetation patches.
- Background: not analysed for regions.

Confidence weighting
---------------------
Each ConnectedRegionStats.mean_confidence is the mean of the confidence
map over the region's pixels. This allows downstream filtering of low-
confidence regions before deriving morphological indices.

Scipy fallback
--------------
All operations are wrapped in ImportError guards. When scipy is unavailable,
a WARNING is logged and an empty GeometryMetrics is returned so the pipeline
continues uninterrupted.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.morphology.contracts import (
    AnalyticsConfig,
    ClassRegionMetrics,
    ConnectedRegionStats,
    GeometryMetrics,
)

__all__ = ["GeometryAnalyzer"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_EPS: float = 1e-10


def _zero_geometry(
    class_names: tuple[str, ...],
    pixel_width_m: float,
    pixel_height_m: float,
) -> GeometryMetrics:
    """Return an empty GeometryMetrics when scipy is unavailable."""
    per_class = {
        name: ClassRegionMetrics(
            class_name=name, class_id=i, region_count=0,
            largest_region_px=0, largest_region_m2=0.0,
            mean_region_size_px=0.0, std_region_size_px=0.0,
            fragmentation_index=0.0, estimated_width_px=0.0,
            regions=(),
        )
        for i, name in enumerate(class_names)
    }
    return GeometryMetrics(
        per_class_regions          = per_class,
        estimated_channel_width_px = 0.0,
        pixel_width_m              = pixel_width_m,
        pixel_height_m             = pixel_height_m,
    )


class GeometryAnalyzer:
    """
    Extracts generic connected-region geometry from a predicted class-ID mask.

    Args:
        config:      AnalyticsConfig.
        class_names: Ordered class names from InferenceResult.
    """

    def __init__(
        self,
        config:      AnalyticsConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config      = config
        self._class_names = class_names
        self._class_index = {name: i for i, name in enumerate(class_names)}

    def compute(
        self,
        mask:       np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> GeometryMetrics:
        """
        Compute GeometryMetrics from a (H, W) uint8 mask.

        Args:
            mask:       (H, W) uint8 predicted class-ID mask.
            confidence: (H, W) float32 per-pixel confidence. When None, all
                        region mean_confidence values default to 0.0.

        Returns:
            GeometryMetrics with per_class_regions for every class.
            Returns zero-valued metrics when scipy is unavailable.
        """
        try:
            from scipy.ndimage import label, find_objects
        except ImportError:
            _LOGGER.warning(
                "GeometryAnalyzer: scipy unavailable; returning zero geometry."
            )
            return _zero_geometry(
                self._class_names,
                self._config.pixel_width_m,
                self._config.pixel_height_m,
            )

        if confidence is None:
            confidence = np.zeros_like(mask, dtype=np.float32)

        min_area       = self._config.min_component_area
        pixel_area_m2  = self._config.pixel_area_m2
        shape_desc     = self._config.compute_shape_descriptors
        background_cls = self._config.background_class

        per_class: dict[str, ClassRegionMetrics] = {}

        for class_name in self._class_names:
            class_id = self._class_index.get(class_name, -1)
            if class_id < 0:
                continue

            # Background is not analysed for object-level regions.
            if class_name == background_cls:
                per_class[class_name] = _empty_region_metrics(class_name, class_id)
                continue

            # Choose connectivity by class type.
            struct = _struct4() if class_name == self._config.water_class else _struct8()

            class_binary          = (mask == class_id).astype(np.int32)
            labeled, n_components = label(class_binary, structure=struct)

            if n_components == 0:
                per_class[class_name] = _empty_region_metrics(class_name, class_id)
                continue

            slices    = find_objects(labeled)
            total_cls = int(class_binary.sum())
            regions   = _build_regions(
                labeled, slices, n_components, confidence,
                pixel_area_m2, min_area, shape_desc,
            )

            per_class[class_name] = _aggregate_regions(
                class_name, class_id, regions, total_cls,
                labeled, min_area, pixel_area_m2,
            )

        # Channel width from the water class.
        water_cls     = self._config.water_class
        channel_width = 0.0
        if water_cls in per_class and per_class[water_cls].region_count > 0:
            channel_width = per_class[water_cls].estimated_width_px

        return GeometryMetrics(
            per_class_regions          = per_class,
            estimated_channel_width_px = channel_width,
            pixel_width_m              = self._config.pixel_width_m,
            pixel_height_m             = self._config.pixel_height_m,
        )


# ==============================================================================
# Private helpers
# ==============================================================================

def _struct4() -> np.ndarray:
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)


def _struct8() -> np.ndarray:
    return np.ones((3, 3), dtype=np.int32)


def _empty_region_metrics(class_name: str, class_id: int) -> ClassRegionMetrics:
    return ClassRegionMetrics(
        class_name=class_name, class_id=class_id, region_count=0,
        largest_region_px=0, largest_region_m2=0.0,
        mean_region_size_px=0.0, std_region_size_px=0.0,
        fragmentation_index=0.0, estimated_width_px=0.0,
        regions=(),
    )


def _build_regions(
    labeled:       np.ndarray,
    slices:        list,
    n_components:  int,
    confidence:    np.ndarray,
    pixel_area_m2: float,
    min_area:      int,
    shape_desc:    bool,
) -> list[ConnectedRegionStats]:
    """
    Build ConnectedRegionStats for every component >= min_area.

    Returns a list sorted by area descending (largest region first).
    """
    regions: list[ConnectedRegionStats] = []

    for comp_idx in range(n_components):
        slc = slices[comp_idx]
        if slc is None:
            continue

        region_label = comp_idx + 1
        region_mask  = (labeled == region_label)
        area_px      = int(region_mask.sum())

        if area_px < min_area:
            continue

        r0 = slc[0].start; r1 = slc[0].stop
        c0 = slc[1].start; c1 = slc[1].stop
        h  = r1 - r0
        w  = c1 - c0
        aspect_ratio = float(w / h) if h > 0 else 0.0
        area_m2      = float(area_px * pixel_area_m2) if pixel_area_m2 > 0 else 0.0

        # Per-region mean confidence.
        mean_conf = float(confidence[region_mask].mean()) if area_px > 0 else 0.0

        # Optional shape descriptors.
        perimeter   = 0.0
        compactness = 0.0
        elongation  = 0.0

        if shape_desc:
            perimeter   = _compute_perimeter(region_mask)
            # print(area_px, perimeter)
            compactness = _compute_compactness(area_px, perimeter)
            elongation  = _compute_elongation(h, w)

        regions.append(ConnectedRegionStats(
            region_id       = region_label,
            area_px         = area_px,
            area_m2         = area_m2,
            bbox_row_min    = r0,
            bbox_row_max    = r1,
            bbox_col_min    = c0,
            bbox_col_max    = c1,
            bbox_height     = h,
            bbox_width      = w,
            aspect_ratio    = aspect_ratio,
            mean_confidence = mean_conf,
            perimeter_px    = perimeter,
            compactness     = compactness,
            elongation      = elongation,
        ))

    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions


def _aggregate_regions(
    class_name:    str,
    class_id:      int,
    regions:       list[ConnectedRegionStats],
    total_cls_px:  int,
    labeled:       np.ndarray,
    min_area:      int,
    pixel_area_m2: float,
) -> ClassRegionMetrics:
    """Aggregate list of ConnectedRegionStats into ClassRegionMetrics."""
    if not regions:
        return _empty_region_metrics(class_name, class_id)

    sizes     = [r.area_px for r in regions]
    largest   = regions[0]   # already sorted descending
    mean_size = float(np.mean(sizes))
    std_size  = float(np.std(sizes))
    frag_idx  = float(len(regions) / total_cls_px) if total_cls_px > 0 else 0.0

    # Estimated width: mean row-width of the largest valid region.
    width_est = _estimate_row_width(labeled, largest.region_id)

    return ClassRegionMetrics(
        class_name           = class_name,
        class_id             = class_id,
        region_count         = len(regions),
        largest_region_px    = largest.area_px,
        largest_region_m2    = largest.area_m2,
        mean_region_size_px  = mean_size,
        std_region_size_px   = std_size,
        fragmentation_index  = frag_idx,
        estimated_width_px   = width_est,
        regions              = tuple(regions),
    )


def _estimate_row_width(labeled: np.ndarray, region_label: int) -> float:
    """Estimate the width of one labeled region as its mean row-pixel count."""
    body_mask  = (labeled == region_label)
    row_widths = body_mask.sum(axis=1)
    non_zero   = row_widths[row_widths > 0]
    return float(non_zero.mean()) if len(non_zero) > 0 else 0.0


# def _compute_perimeter(region_mask: np.ndarray) -> float:
#     """
#     Compute the 4-connectivity perimeter of a binary region.

#     The perimeter is the count of foreground pixels that have at least one
#     background neighbour (including image boundary). Fully vectorized.
#     """
#     from scipy.ndimage import binary_erosion
#     eroded     = binary_erosion(region_mask, structure=_struct4())
#     boundary   = region_mask & ~eroded
#     return float(boundary.sum())
def _compute_perimeter(region_mask: np.ndarray) -> float:
    """
    Compute the 4-connected edge perimeter of a binary region.

    Each exposed pixel edge contributes one unit.
    """
    mask = region_mask.astype(np.uint8)

    perimeter = 0

    # Up
    perimeter += np.sum(mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0))))

    # Down
    perimeter += np.sum(mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0))))

    # Left
    perimeter += np.sum(mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0))))

    # Right
    perimeter += np.sum(mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1))))

    return float(perimeter)


def _compute_compactness(area_px: int, perimeter_px: float) -> float:
    """
    Compactness = 4*pi*area / perimeter^2.

    1.0 for a perfect circle, < 1.0 for elongated or complex shapes.
    Returns 0.0 when perimeter == 0.
    """
    if perimeter_px < _EPS:
        return 0.0
    return float(4.0 * math.pi * area_px / (perimeter_px ** 2))


def _compute_elongation(bbox_height: int, bbox_width: int) -> float:
    """
    Elongation = 1 - (short_side / long_side).

    0.0 for a square bounding box, approaches 1.0 for very elongated shapes.
    Returns 0.0 when both dimensions are zero.
    """
    if bbox_height == 0 and bbox_width == 0:
        return 0.0
    short = min(bbox_height, bbox_width)
    long_ = max(bbox_height, bbox_width)
    return float(1.0 - short / long_) if long_ > 0 else 0.0


def _component_sizes(labeled: np.ndarray, n: int) -> list[int]:
    """Return pixel count for each component label 1..n. (kept for test compatibility)"""
    if n == 0:
        return []
    counts = np.bincount(labeled.ravel())
    return [int(counts[i]) for i in range(1, n + 1) if i < len(counts)]


def _estimate_width(
    water_labeled: np.ndarray,
    water_sizes:   list[int],
    min_area:      int,
) -> float:
    """Legacy helper kept for test compatibility."""
    if not water_sizes:
        return 0.0
    valid = [(s, i + 1) for i, s in enumerate(water_sizes) if s >= min_area]
    if not valid:
        return 0.0
    largest_label = max(valid, key=lambda x: x[0])[1]
    return _estimate_row_width(water_labeled, largest_label)
