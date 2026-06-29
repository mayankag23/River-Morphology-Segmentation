"""
Feature stack assembly and FeatureStackResult for the River Morphology system.

Contains:
    FeatureStackResult  -- Immutable result of spectral feature generation.
    FeatureStackAssembler -- Builds the stacked ee.Image by appending indices.
    list_available_features()  -- List all registered index names.
    validate_harmonization()   -- Check that input uses harmonized band names.
    describe_feature_stack()   -- ASCII description of a FeatureStackResult.

This module does NOT compute any spectral indices. It only assembles the
final image stack and manages the band name bookkeeping.

Input:  A base ee.Image (composite) and a dict of {name: ee.Image} indices.
Output: FeatureStackResult with the complete stacked ee.Image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError
from src.gee.harmonization import COMMON_BAND_NAMES
from src.gee.registry import FeatureRegistry

if TYPE_CHECKING:
    from src.gee.composite import CompositeResult
    from src.gee.features import FeatureConfig

__all__ = [
    "FeatureStackResult",
    "FeatureStackAssembler",
    "list_available_features",
    "validate_harmonization",
    "describe_feature_stack",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# FeatureStackResult
# ==============================================================================

@dataclass(frozen=True)
class FeatureStackResult:
    """
    Immutable result of SpectralFeatureGenerator.generate().

    Contains the stacked ee.Image (composite bands + index bands) and
    complete provenance metadata. All future modules (Module 7: download;
    Module 8: patch generation) use this as their primary input.

    Attributes:
        image:
            ee.Image with all bands stacked:
            [composite bands] + [computed spectral index bands].
            All operations are lazy; no getInfo() has been called.
        features_computed:
            Ordered tuple of index names that were computed and appended.
            Order matches the order the bands appear in the image.
        features_skipped:
            Tuple of index names that were registered but not computed
            (disabled via FeatureConfig or flagged not-optional by default).
        composite_band_names:
            Band names from the original composite image (source bands).
        index_band_names:
            Band names added by this feature engineering stage.
        all_band_names:
            Union: composite_band_names + index_band_names.
            Module 7 uses this to select specific bands for download.
        source_composite:
            The CompositeResult this was derived from. Provides access to
            the compositing method, sensor info, and date range.
        feature_config:
            The FeatureConfig that controlled which indices were computed.
    """

    image:                Any                     # ee.Image (lazy computation)
    features_computed:    tuple[str, ...]
    features_skipped:     tuple[str, ...]
    composite_band_names: tuple[str, ...]
    index_band_names:     tuple[str, ...]
    all_band_names:       tuple[str, ...]
    source_composite:     Any                     # CompositeResult (TYPE_CHECKING)
    feature_config:       Any                     # FeatureConfig (TYPE_CHECKING)

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines for logging and display."""
        lines = [
            f"  Indices computed:  {list(self.features_computed)}",
            f"  Indices skipped:   {list(self.features_skipped)}",
            f"  Composite bands:   {list(self.composite_band_names)}",
            f"  Index bands:       {list(self.index_band_names)}",
            f"  Total bands:       {len(self.all_band_names)}",
        ]
        return lines


# ==============================================================================
# FeatureStackAssembler
# ==============================================================================

class FeatureStackAssembler:
    """
    Assembles a stacked ee.Image by appending spectral index bands.

    Tracks band names through each append operation so that the final
    FeatureStackResult has accurate band name metadata without calling
    getInfo() on the image.

    Args:
        base_image:      The composite ee.Image to which indices are appended.
        base_band_names: Tuple of band names already in the base_image.
                         Used to build the all_band_names list accurately.
    """

    def __init__(
        self,
        base_image: Any,
        base_band_names: tuple[str, ...],
    ) -> None:
        self._image:          Any         = base_image
        self._base_bands:     tuple[str, ...] = base_band_names
        self._index_bands:    list[str]   = []
        self._logger: logging.Logger = logging.getLogger(__name__)

    def add_index(
        self,
        index_image: Any,
        index_name: str,
    ) -> None:
        """
        Append a single-band index image to the current stack.

        Uses addBands() which adds the index as an additional band without
        modifying the existing bands. The band name is tracked internally.

        Args:
            index_image: Single-band ee.Image named after the index.
            index_name:  Canonical name of the index (e.g. "MNDWI").

        Raises:
            GEEAPIError: addBands() call failed.
        """
        self._logger.debug("Appending index band: %s", index_name)
        try:
            self._image = self._image.addBands(index_image)
            self._index_bands.append(index_name)
        except Exception as exc:
            raise GEEAPIError(
                operation=f"add_index_band_{index_name}",
                reason=f"addBands() failed for index '{index_name}': {exc}",
            ) from exc

    def build(self) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
        """
        Finalize the image stack and return results.

        Returns:
            Tuple of:
                (stacked_ee_image,
                 all_band_names,     # base + index bands, in order
                 index_band_names)   # only the appended index bands
        """
        index_bands = tuple(self._index_bands)
        all_bands   = self._base_bands + index_bands
        self._logger.debug(
            "Feature stack built: %d composite + %d index = %d total bands.",
            len(self._base_bands),
            len(index_bands),
            len(all_bands),
        )
        return self._image, all_bands, index_bands

    @property
    def index_count(self) -> int:
        """Number of index bands appended so far."""
        return len(self._index_bands)


# ==============================================================================
# Public utility functions
# ==============================================================================

def list_available_features() -> list[str]:
    """
    Return the names of all registered spectral indices.

    Returns:
        Sorted list of canonical index names.
    """
    return sorted(FeatureRegistry.names())


def validate_harmonization(composite_result: Any) -> None:
    """
    Verify that a CompositeResult has harmonized band names.

    Spectral index computation in this module requires bands to be named
    with COMMON_BAND_NAMES (Blue, Green, Red, NIR, SWIR1, SWIR2, Thermal,
    QA_PIXEL). If the composite was built without harmonization, indices
    that reference 'Green' or 'NIR' will fail with an EE server-side error
    when the image is eventually evaluated.

    Args:
        composite_result: CompositeResult from LandsatCompositor.build_composite().

    Raises:
        InvalidValueError: CompositeResult.band_names does not match
                           COMMON_BAND_NAMES, indicating harmonization
                           was not applied.
    """
    band_names = tuple(composite_result.band_names)

    if band_names and band_names != COMMON_BAND_NAMES:
        raise InvalidValueError(
            field="composite_result.band_names",
            value=list(band_names),
            reason=(
                f"Expected harmonized band names: {list(COMMON_BAND_NAMES)}. "
                f"Got: {list(band_names)}. "
                "Ensure LandsatPreprocessor was run with "
                "apply_harmonization=True before compositing."
            ),
        )

    if not band_names:
        # Empty band_names means harmonization was not applied.
        # Log as warning and allow to proceed; errors will surface at eval.
        _LOGGER.warning(
            "CompositeResult.band_names is empty, indicating "
            "harmonization was not applied. Spectral index computation "
            "may fail when the image is eventually evaluated. "
            "Consider running LandsatPreprocessor with apply_harmonization=True."
        )


def describe_feature_stack(result: FeatureStackResult) -> str:
    """
    Return a multi-line ASCII description of a FeatureStackResult.

    Args:
        result: A FeatureStackResult from SpectralFeatureGenerator.generate().

    Returns:
        Multi-line string suitable for logging or display.
    """
    lines = [
        "FeatureStackResult",
        "  " + "-" * 40,
        *result.summary_lines(),
    ]
    return "\n".join(lines)