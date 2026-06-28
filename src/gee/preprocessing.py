"""
Landsat image preprocessing pipeline for the River Morphology system.

LandsatPreprocessor transforms a CollectionResult (from Module 4) into a
ProcessedCollectionResult ready for spectral index computation (Module 6).

Pipeline stages (all server-side, all via .map()):
    1. Scale factors:   SR = DN * 0.0000275 + (-0.2)  [optical bands]
                        ST = DN * 0.00341802 + 149.0   [thermal bands]
                        Applied to SR_B.* and ST_B.* respectively.
                        QA_PIXEL is preserved without scaling.
    2. QA masking:      Remove cloud, cloud shadow, cirrus, dilated cloud,
                        fill, and optionally snow. Configured via QAMaskConfig.
    3. Harmonization:   Rename bands to common schema so downstream modules
                        use consistent names regardless of sensor family.

Each stage is independently enabled/disabled via process() parameters.
Configuration values are read from config.preprocessing.

No getInfo() calls are made. No imagery is downloaded. No spectral indices
are computed. The output ee.ImageCollection is a lazy server-side graph.

Input:  CollectionResult  (from src.gee.collections.LandsatCollectionBuilder)
Output: ProcessedCollectionResult

Usage:

    from src.gee.preprocessing import LandsatPreprocessor

    preprocessor = LandsatPreprocessor(client, config)
    result = preprocessor.process(collection_result)

    # Access the processed collection
    processed_collection = result.collection

    # Or disable specific stages:
    result = preprocessor.process(
        collection_result,
        apply_scaling=True,
        apply_masking=True,
        apply_harmonization=False,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError, GEENotInstalledError
from src.gee.collections import CollectionResult
from src.gee.harmonization import COMMON_BAND_NAMES, BandHarmonizer
from src.gee.masking import LandsatQAMasker, QAMaskConfig

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

__all__ = [
    "LandsatPreprocessor",
    "ProcessedCollectionResult",
    "ScalingConfig",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Landsat Collection 2 Level-2 USGS-specified scale factors.
# Defined as module constants as fallback defaults when not in config.
_DEFAULT_SR_SCALE_FACTOR:     float = 0.0000275
_DEFAULT_SR_OFFSET:           float = -0.2
_DEFAULT_THERMAL_SCALE_FACTOR: float = 0.00341802
_DEFAULT_THERMAL_OFFSET:      float = 149.0
_DEFAULT_SR_MIN:              float = 0.0
_DEFAULT_SR_MAX:              float = 1.0


# ==============================================================================
# ScalingConfig
# ==============================================================================

@dataclass(frozen=True)
class ScalingConfig:
    """
    Immutable configuration for Landsat Collection 2 Level-2 scale factors.

    Attributes:
        sr_scale_factor:       Multiplier for SR bands (optical).
        sr_offset:             Additive offset for SR bands.
        thermal_scale_factor:  Multiplier for ST (thermal) bands.
        thermal_offset:        Additive offset for ST bands (produces Kelvin).
        sr_min:                Lower clip bound for optical bands (post-scale).
        sr_max:                Upper clip bound for optical bands (post-scale).
        clip_to_valid_range:   If True, clamp optical bands to [sr_min, sr_max].
    """

    sr_scale_factor:      float = _DEFAULT_SR_SCALE_FACTOR
    sr_offset:            float = _DEFAULT_SR_OFFSET
    thermal_scale_factor: float = _DEFAULT_THERMAL_SCALE_FACTOR
    thermal_offset:       float = _DEFAULT_THERMAL_OFFSET
    sr_min:               float = _DEFAULT_SR_MIN
    sr_max:               float = _DEFAULT_SR_MAX
    clip_to_valid_range:  bool  = True


# ==============================================================================
# ProcessedCollectionResult
# ==============================================================================

@dataclass(frozen=True)
class ProcessedCollectionResult:
    """
    Immutable output of LandsatPreprocessor.process().

    Wraps the processed ee.ImageCollection alongside provenance metadata
    describing what operations were applied. Future modules (Module 6:
    spectral indices; Module 7: patch generation) receive this object
    as their primary input.

    Attributes:
        collection:           Processed ee.ImageCollection (lazy graph).
        source_result:        Original CollectionResult from Module 4.
        operations_applied:   Ordered tuple of operation names completed.
        band_names:           Current band names. Equals COMMON_BAND_NAMES
                              if harmonization_applied is True, otherwise
                              empty (band names depend on the sensor family).
        scale_applied:        True if USGS scale factors were applied.
        masking_applied:      True if QA_PIXEL masking was applied.
        harmonization_applied: True if band names were harmonized.
    """

    collection:            Any                     # ee.ImageCollection
    source_result:         CollectionResult
    operations_applied:    tuple[str, ...]
    band_names:            tuple[str, ...]
    scale_applied:         bool
    masking_applied:       bool
    harmonization_applied: bool

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines describing this result."""
        lines = [
            f"  Scale applied:         {self.scale_applied}",
            f"  Masking applied:       {self.masking_applied}",
            f"  Harmonization applied: {self.harmonization_applied}",
            f"  Band names:            "
            f"{list(self.band_names) if self.band_names else 'source-dependent'}",
            f"  Operations:            {list(self.operations_applied)}",
        ]
        if self.source_result.has_mixed_sensor_families:
            lines.append(
                "  [INFO] Source is a mixed-sensor collection. "
                "Harmonization is required before spectral index computation."
            )
        return lines


# ==============================================================================
# LandsatPreprocessor
# ==============================================================================

class LandsatPreprocessor:
    """
    Applies USGS scale factors, QA masking, and band harmonization to a
    Landsat CollectionResult, producing a ProcessedCollectionResult.

    All operations use collection.map() and are lazy server-side EE
    computations. No imagery is downloaded; no getInfo() calls are made.

    Args:
        client: Initialized EarthEngineClient. Required so that future
                extensions can call execute_with_retry if needed.
        config: Fully initialized Config. Scaling and masking parameters
                are read from config.preprocessing and config.satellite.

    Example:

        preprocessor = LandsatPreprocessor(client, config)

        result = preprocessor.process(collection_result)
        # All three stages applied.

        result = preprocessor.process(
            collection_result,
            apply_scaling=True,
            apply_masking=False,
            apply_harmonization=True,
        )
        # Skip masking stage.
    """

    def __init__(
        self,
        client: EarthEngineClient,
        config: Config,
    ) -> None:
        self._client = client
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        # Build sub-components from config at construction time.
        self._scaling_config = self._build_scaling_config()
        self._masker         = LandsatQAMasker.from_preprocessing_config(config)
        self._harmonizer     = BandHarmonizer()

        self._logger.debug(
            "LandsatPreprocessor initialized. scaling=%s, masking=%s",
            self._scaling_config,
            self._masker.mask_config.summary(),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        collection_result: CollectionResult,
        apply_scaling: bool = True,
        apply_masking: bool = True,
        apply_harmonization: bool = True,
    ) -> ProcessedCollectionResult:
        """
        Apply the preprocessing pipeline to a CollectionResult.

        Stages are applied in order: scaling -> masking -> harmonization.
        Each stage is independently toggled. Disabled stages are skipped
        without creating unnecessary EE computation nodes.

        For mixed-sensor collections (has_mixed_sensor_families = True),
        harmonization is strongly recommended. Without it, a merged L7+L8
        collection will have conflicting band names in a single ImageCollection
        (e.g., both SR_B1 and SR_B2 representing "Blue" from different images).

        Args:
            collection_result:   CollectionResult from LandsatCollectionBuilder.
            apply_scaling:       Apply USGS Collection 2 Level-2 scale factors.
            apply_masking:       Apply QA_PIXEL masking per QAMaskConfig.
            apply_harmonization: Rename bands to COMMON_BAND_NAMES.

        Returns:
            ProcessedCollectionResult with the processed collection.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          Any pipeline stage EE call failed.
        """
        if apply_scaling and apply_harmonization and not apply_masking:
            self._logger.warning(
                "apply_masking=False with apply_scaling=True: "
                "clouds will be included in the processed collection."
            )

        if collection_result.has_mixed_sensor_families and not apply_harmonization:
            self._logger.warning(
                "Mixed-sensor collection detected but apply_harmonization=False. "
                "Downstream spectral index computation may fail due to "
                "inconsistent band names across sensor families."
            )

        collection      = collection_result.collection
        operations: list[str] = []

        if apply_scaling:
            collection = self._apply_scaling(collection)
            operations.append("scaling")
            self._logger.info("Scale factors applied.")

        if apply_masking:
            collection = self._apply_masking(collection)
            operations.append("qa_masking")
            self._logger.info(
                "QA masking applied. %s",
                self._masker.mask_config.summary(),
            )

        if apply_harmonization:
            collection = self._apply_harmonization(collection)
            operations.append("harmonization")
            self._logger.info("Band harmonization applied.")

        band_names = COMMON_BAND_NAMES if apply_harmonization else ()

        result = ProcessedCollectionResult(
            collection=collection,
            source_result=collection_result,
            operations_applied=tuple(operations),
            band_names=band_names,
            scale_applied=apply_scaling,
            masking_applied=apply_masking,
            harmonization_applied=apply_harmonization,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------

    def _apply_scaling(self, collection: Any) -> Any:
        """
        Apply USGS Collection 2 Level-2 radiometric scale factors.

        Maps _make_scale_function() over every image in the collection.
        The function operates on:
            SR_B.* bands: multiply by sr_scale_factor, add sr_offset.
            ST_B.* bands: multiply by thermal_scale_factor, add thermal_offset.
            QA_PIXEL:     preserved unchanged (no scaling applied).

        Args:
            collection: An ee.ImageCollection with unscaled DN values.

        Returns:
            The same collection with physical reflectance/temperature values.

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
            GEEAPIError:          collection.map() failed.
        """
        scale_fn = self._make_scale_function()
        try:
            return collection.map(scale_fn)
        except Exception as exc:
            raise GEEAPIError(
                operation="apply_scaling",
                reason=f"collection.map() for scaling failed: {exc}",
            ) from exc

    def _make_scale_function(self) -> Any:
        """
        Return a closure for use with collection.map() that applies scale factors.

        The closure captures ScalingConfig values so that the returned
        function references only Python primitives, not the Config object.
        This avoids serialization issues if EE attempts to inspect the closure.

        Returns:
            A callable (image) -> ee.Image for use with collection.map().

        Raises:
            GEENotInstalledError: earthengine-api is not installed.
        """
        try:
            import ee  # noqa: F401
        except ImportError as exc:
            raise GEENotInstalledError(
                "earthengine-api is not installed."
            ) from exc

        sc        = self._scaling_config
        sr_scale  = sc.sr_scale_factor
        sr_off    = sc.sr_offset
        th_scale  = sc.thermal_scale_factor
        th_off    = sc.thermal_offset
        clip      = sc.clip_to_valid_range
        sr_min    = sc.sr_min
        sr_max    = sc.sr_max

        def _scale_image(image: Any) -> Any:
            """Scale a single Landsat C2 L2 image. For use with .map()."""
            import ee  # noqa: F401

            # Scale optical surface reflectance bands.
            optical = (
                image.select("SR_B.*")
                .multiply(sr_scale)
                .add(sr_off)
            )

            # Scale thermal surface temperature bands (result is in Kelvin).
            thermal = (
                image.select("ST_B.*")
                .multiply(th_scale)
                .add(th_off)
            )

            # Replace the original DN bands with scaled values.
            # addBands(overwrite=True) replaces matching band names.
            scaled = (
                image
                .addBands(optical, names=None, overwrite=True)
                .addBands(thermal, names=None, overwrite=True)
            )

            if clip:
                # Clamp optical bands to the physically valid SR range [0, 1].
                # Thermal bands are not clamped (Kelvin values exceed 1.0).
                clipped_optical = scaled.select("SR_B.*").clamp(sr_min, sr_max)
                scaled = scaled.addBands(clipped_optical, names=None, overwrite=True)

            return scaled

        return _scale_image

    def _apply_masking(self, collection: Any) -> Any:
        """
        Apply QA_PIXEL masking to the collection using LandsatQAMasker.

        Delegates to self._masker.apply_to_collection() which applies
        the configured mask flags via collection.map().

        Args:
            collection: An ee.ImageCollection (scaled or unscaled).

        Returns:
            The collection with masked pixels set to nodata.

        Raises:
            GEEAPIError: apply_to_collection() raised.
        """
        return self._masker.apply_to_collection(collection)

    def _apply_harmonization(self, collection: Any) -> Any:
        """
        Rename bands in the collection to COMMON_BAND_NAMES.

        Delegates to self._harmonizer.harmonize_collection() which applies
        sensor-specific rename mappings via collection.map().

        Args:
            collection: An ee.ImageCollection (any sensor family).

        Returns:
            The collection with all images using COMMON_BAND_NAMES.

        Raises:
            GEEAPIError: harmonize_collection() raised.
        """
        return self._harmonizer.harmonize_collection(collection)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _build_scaling_config(self) -> ScalingConfig:
        """
        Read scaling parameters from config.satellite and config.preprocessing.

        Falls back to module-level defaults (_DEFAULT_*) when config keys
        are absent. This allows Module 5 to work with Module 1's config.yaml
        without requiring the thermal_* keys to be present.

        Returns:
            ScalingConfig with all values populated.
        """
        sat = self._config.satellite
        pp  = self._config.preprocessing

        return ScalingConfig(
            sr_scale_factor=float(getattr(sat, "scale_factor", _DEFAULT_SR_SCALE_FACTOR)),
            sr_offset=float(getattr(sat, "offset", _DEFAULT_SR_OFFSET)),
            thermal_scale_factor=float(
                getattr(pp, "thermal_scale_factor", _DEFAULT_THERMAL_SCALE_FACTOR)
            ),
            thermal_offset=float(
                getattr(pp, "thermal_offset", _DEFAULT_THERMAL_OFFSET)
            ),
            sr_min=float(getattr(sat, "sr_min", _DEFAULT_SR_MIN)),
            sr_max=float(getattr(sat, "sr_max", _DEFAULT_SR_MAX)),
            clip_to_valid_range=bool(
                getattr(pp, "clip_to_valid_range", True)
            ),
        )