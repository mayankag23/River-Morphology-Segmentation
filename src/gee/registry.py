"""
Spectral index registry for the River Morphology Segmentation System.

Provides IndexMetadata (immutable descriptor per index) and FeatureRegistry
(the central directory mapping index names to their metadata and computation
functions).

The registry is populated at module load time from BUILT_IN_INDICES. It is
read-only at runtime: no new indices can be registered after import.

Usage:

    from src.gee.registry import FeatureRegistry, IndexMetadata

    # Retrieve metadata for a specific index
    meta = FeatureRegistry.get("MNDWI")
    print(meta.formula_str)     # "(Green - SWIR1) / (Green + SWIR1)"
    print(meta.category)        # "water"

    # List all registered indices
    all_names = FeatureRegistry.names()

    # Filter by category
    water_indices = FeatureRegistry.get_by_category("water")

    # Get the computation function
    fn = FeatureRegistry.get_function("BSI")
    result = fn(composite_image)

    # Check membership
    if FeatureRegistry.is_registered("NDWI"):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.core.exceptions import InvalidValueError
from src.gee.indices import (
    compute_awei_nsh,
    compute_awei_sh,
    compute_bsi,
    compute_mndwi,
    compute_ndbi,
    compute_ndmi,
    compute_ndvi,
    compute_ndwi,
    compute_savi,
)

__all__ = [
    "IndexMetadata",
    "FeatureRegistry",
    "BUILT_IN_INDICES",
    "INDEX_CATEGORIES",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# All valid index category names.
INDEX_CATEGORIES: frozenset[str] = frozenset({
    "water",
    "vegetation",
    "bare_soil",
    "moisture",
    "built_up",
})


# ==============================================================================
# IndexMetadata
# ==============================================================================

@dataclass(frozen=True)
class IndexMetadata:
    """
    Immutable descriptor for a single spectral index.

    Contains everything needed to compute, label, and explain an index.
    Used by the FeatureRegistry and downstream reporting modules.

    Attributes:
        name:               Canonical index name, e.g. "MNDWI".
                            Must be unique within the registry.
        output_band_name:   Band name in the output ee.Image. Matches name.
        formula_str:        Human-readable formula string. ASCII only.
        description:        One-sentence description of what the index measures.
        river_relevance:    Why this index is useful for river morphology.
        bands_required:     Harmonized band names the formula needs.
        category:           One of INDEX_CATEGORIES.
        reference:          Publication citation for the index.
        is_optional:        False = recommended on by default.
                            True = off by default (user must enable explicitly).
        config_key:         The key in config.yaml features section,
                            e.g. "ndwi" for features.ndwi.
    """

    name:             str
    output_band_name: str
    formula_str:      str
    description:      str
    river_relevance:  str
    bands_required:   tuple[str, ...]
    category:         str
    reference:        str
    is_optional:      bool
    config_key:       str


# ==============================================================================
# BUILT_IN_INDICES — single source of truth for all registered indices
# ==============================================================================

BUILT_IN_INDICES: tuple[IndexMetadata, ...] = (
    IndexMetadata(
        name="NDWI",
        output_band_name="NDWI",
        formula_str="(Green - NIR) / (Green + NIR)",
        description="Normalized Difference Water Index for open water detection.",
        river_relevance=(
            "Positive for open water; negative for vegetation and dry soil. "
            "Secondary water indicator complementing MNDWI for clear channels."
        ),
        bands_required=("Green", "NIR"),
        category="water",
        reference="McFeeters (1996), Remote Sensing of Environment, 57(2), 167-182.",
        is_optional=False,
        config_key="ndwi",
    ),
    IndexMetadata(
        name="MNDWI",
        output_band_name="MNDWI",
        formula_str="(Green - SWIR1) / (Green + SWIR1)",
        description=(
            "Modified Normalized Difference Water Index. "
            "Primary discriminator between water and sand."
        ),
        river_relevance=(
            "Most effective single index for separating water (positive) "
            "from sand/sediment (negative) in river systems. "
            "Threshold: MNDWI > 0.2 = water, MNDWI < 0.2 and BSI > 0.1 = sand."
        ),
        bands_required=("Green", "SWIR1"),
        category="water",
        reference="Xu (2006), International Journal of Remote Sensing, 27(14), 3025-3033.",
        is_optional=False,
        config_key="mndwi",
    ),
    IndexMetadata(
        name="AWEI_sh",
        output_band_name="AWEI_sh",
        formula_str="Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2",
        description=(
            "Automated Water Extraction Index (shadow variant) "
            "for shadow-affected water detection."
        ),
        river_relevance=(
            "Captures water in shadowed river channels caused by riparian "
            "vegetation canopy and steep valley walls. Positive for water. "
            "Complements MNDWI where shadows create false negatives."
        ),
        bands_required=("Blue", "Green", "NIR", "SWIR1", "SWIR2"),
        category="water",
        reference="Feyisa et al. (2014), Remote Sensing of Environment, 140, 23-35.",
        is_optional=False,
        config_key="awei_sh",
    ),
    IndexMetadata(
        name="AWEI_nsh",
        output_band_name="AWEI_nsh",
        formula_str="4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)",
        description=(
            "Automated Water Extraction Index (no-shadow variant) "
            "for open water body extraction."
        ),
        river_relevance=(
            "Effective for turbid, wide river channels in flat terrain "
            "without significant shadowing. High SWIR2 coefficient strongly "
            "suppresses dry sandbars. Positive for open water."
        ),
        bands_required=("Green", "NIR", "SWIR1", "SWIR2"),
        category="water",
        reference="Feyisa et al. (2014), Remote Sensing of Environment, 140, 23-35.",
        is_optional=False,
        config_key="awei_nsh",
    ),
    IndexMetadata(
        name="NDVI",
        output_band_name="NDVI",
        formula_str="(NIR - Red) / (NIR + Red)",
        description="Normalized Difference Vegetation Index.",
        river_relevance=(
            "Primary vegetation suppressor. High NDVI (> 0.3) indicates "
            "riparian vegetation that must be excluded from the water/sand "
            "classification. Negative NDVI reliably identifies water surfaces."
        ),
        bands_required=("NIR", "Red"),
        category="vegetation",
        reference="Rouse et al. (1973), Third ERTS Symposium, NASA SP-351, 309-317.",
        is_optional=False,
        config_key="ndvi",
    ),
    IndexMetadata(
        name="SAVI",
        output_band_name="SAVI",
        formula_str="((NIR - Red) / (NIR + Red + L)) * (1 + L), L = 0.5",
        description=(
            "Soil-Adjusted Vegetation Index. Reduces soil background noise "
            "in vegetation detection."
        ),
        river_relevance=(
            "Improves vegetation detection on bright sand backgrounds in "
            "braided river systems where sparse vegetation grows on sandbars. "
            "L = 0.5 recommended for intermediate cover conditions."
        ),
        bands_required=("NIR", "Red"),
        category="vegetation",
        reference="Huete (1988), Remote Sensing of Environment, 25(3), 295-309.",
        is_optional=False,
        config_key="savi",
    ),
    IndexMetadata(
        name="BSI",
        output_band_name="BSI",
        formula_str=(
            "((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))"
        ),
        description=(
            "Bare Soil Index. Detects exposed soil and sand surfaces."
        ),
        river_relevance=(
            "PRIMARY sand/sediment discriminator. Strongly positive for "
            "dry, exposed sandbars and fluvial sediment. Used with MNDWI "
            "to create the two-threshold water/sand/background classifier: "
            "MNDWI > 0.2 -> water, BSI > 0.1 and MNDWI < 0.2 -> sand."
        ),
        bands_required=("Blue", "Red", "NIR", "SWIR1"),
        category="bare_soil",
        reference="Rikimaru et al. (2002), FAO Forestry Department.",
        is_optional=False,
        config_key="bsi",
    ),
    IndexMetadata(
        name="NDMI",
        output_band_name="NDMI",
        formula_str="(NIR - SWIR1) / (NIR + SWIR1)",
        description="Normalized Difference Moisture Index. Measures surface moisture.",
        river_relevance=(
            "Distinguishes wet from dry sand and maps recently inundated "
            "floodplain areas. Wet sand (moderate NDMI) vs dry sand "
            "(low NDMI) vs water (MNDWI more effective). "
            "Useful for mapping flood extent and recent channel migration."
        ),
        bands_required=("NIR", "SWIR1"),
        category="moisture",
        reference="Gao (1996), Remote Sensing of Environment, 58(3), 257-266.",
        is_optional=False,
        config_key="ndmi",
    ),
    IndexMetadata(
        name="NDBI",
        output_band_name="NDBI",
        formula_str="(SWIR1 - NIR) / (SWIR1 + NIR)",
        description=(
            "Normalized Difference Built-Up Index. "
            "Highlights impervious surfaces and built-up areas."
        ),
        river_relevance=(
            "OPTIONAL. Detects concrete bridges, embankments, and urban "
            "encroachment in river corridors. Can mask anthropogenic "
            "structures from morphology maps. Note: inverse of NDWI."
        ),
        bands_required=("SWIR1", "NIR"),
        category="built_up",
        reference="Zha et al. (2003), International Journal of Remote Sensing, 24(3).",
        is_optional=True,
        config_key="ndbi",
    ),
)


# ==============================================================================
# FeatureRegistry
# ==============================================================================

class FeatureRegistry:
    """
    Read-only registry mapping spectral index names to their metadata and
    computation functions.

    Populated at module load time from BUILT_IN_INDICES. No entries can
    be added or removed at runtime. All class methods raise InvalidValueError
    for unrecognized index names.

    All methods are class methods; no instantiation is required.

    Usage:
        meta = FeatureRegistry.get("MNDWI")
        fn   = FeatureRegistry.get_function("BSI")
        bsi  = fn(composite_image)
    """

    # Populated at class definition time, after BUILT_IN_INDICES is defined.
    _metadata:  dict[str, IndexMetadata]
    _functions: dict[str, Callable[..., Any]]

    @classmethod
    def get(cls, name: str) -> IndexMetadata:
        """
        Retrieve metadata for a registered spectral index.

        Args:
            name: Canonical index name, e.g. "MNDWI". Case-sensitive.

        Returns:
            IndexMetadata for the requested index.

        Raises:
            InvalidValueError: name is not registered.
        """
        meta = cls._metadata.get(name)
        if meta is None:
            raise InvalidValueError(
                field="index_name",
                value=name,
                reason=(
                    f"'{name}' is not a registered spectral index. "
                    f"Available indices: {sorted(cls._metadata)}"
                ),
            )
        return meta

    @classmethod
    def get_function(cls, name: str) -> Callable[..., Any]:
        """
        Retrieve the computation function for a registered spectral index.

        The returned callable has signature: (image: Any) -> Any
        where image is an ee.Image with harmonized band names.
        Exception: compute_savi has an additional soil_factor parameter.

        Args:
            name: Canonical index name. Case-sensitive.

        Returns:
            Callable for computing the index on an ee.Image.

        Raises:
            InvalidValueError: name is not registered.
        """
        cls.get(name)  # validate name first; raises if not registered
        return cls._functions[name]

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return the names of all registered indices in a stable order."""
        return tuple(cls._metadata.keys())

    @classmethod
    def get_by_category(cls, category: str) -> tuple[IndexMetadata, ...]:
        """
        Return all registered indices belonging to the given category.

        Args:
            category: One of INDEX_CATEGORIES (e.g., "water", "vegetation").

        Returns:
            Tuple of IndexMetadata instances in registration order.

        Raises:
            InvalidValueError: category is not in INDEX_CATEGORIES.
        """
        if category not in INDEX_CATEGORIES:
            raise InvalidValueError(
                field="category",
                value=category,
                reason=(
                    f"'{category}' is not a valid category. "
                    f"Valid categories: {sorted(INDEX_CATEGORIES)}"
                ),
            )
        return tuple(
            meta for meta in cls._metadata.values()
            if meta.category == category
        )

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if name is a registered spectral index."""
        return name in cls._metadata

    @classmethod
    def get_default_enabled(cls) -> tuple[str, ...]:
        """
        Return names of all non-optional (default-enabled) indices.

        Returns:
            Tuple of index names where is_optional is False.
        """
        return tuple(
            meta.name for meta in cls._metadata.values()
            if not meta.is_optional
        )

    @classmethod
    def get_config_key_map(cls) -> dict[str, str]:
        """
        Return a mapping from config_key to canonical index name.

        Example: {"ndwi": "NDWI", "mndwi": "MNDWI", ...}

        Useful for resolving config.yaml feature keys to index names.
        """
        return {meta.config_key: meta.name for meta in cls._metadata.values()}


# Populate class-level dictionaries after class definition to ensure
# BUILT_IN_INDICES is already defined at the point of use.
FeatureRegistry._metadata = {
    meta.name: meta for meta in BUILT_IN_INDICES
}

FeatureRegistry._functions = {
    "NDWI":     compute_ndwi,
    "MNDWI":    compute_mndwi,
    "AWEI_sh":  compute_awei_sh,
    "AWEI_nsh": compute_awei_nsh,
    "NDVI":     compute_ndvi,
    "SAVI":     compute_savi,
    "BSI":      compute_bsi,
    "NDMI":     compute_ndmi,
    "NDBI":     compute_ndbi,
}