"""
Spectral feature engineering pipeline for the River Morphology system.

SpectralFeatureGenerator transforms a CompositeResult (from Module 5) into
a FeatureStackResult by computing configurable spectral indices and appending
them to the composite image as additional bands.

Pipeline:
    1. Read FeatureConfig (from Config or explicit argument).
    2. Validate that the input CompositeResult has harmonized band names.
    3. For each enabled index (in registry order):
         a. Retrieve the computation function from FeatureRegistry.
         b. Call the function on the composite image.
         c. Append the single-band result using FeatureStackAssembler.
    4. Return FeatureStackResult with the stacked image and metadata.

All operations are lazy server-side EE computations. No getInfo() calls.
No imagery is downloaded. The output is a single ee.Image containing the
original composite bands plus all enabled index bands.

Input:  CompositeResult from LandsatCompositor.build_composite()
Output: FeatureStackResult

Usage:

    from src.gee.features import SpectralFeatureGenerator, FeatureConfig

    generator = SpectralFeatureGenerator(client, config)
    result    = generator.generate(composite_result)

    # All composite + index bands are in result.image
    # Band list: result.all_band_names
    # E.g.: ('Blue', 'Green', ..., 'NDWI', 'MNDWI', 'BSI', ...)

    # Disable specific indices:
    custom_config = FeatureConfig(ndbi=True, awei_sh=False)
    result = generator.generate(composite_result, feature_config=custom_config)
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
from src.gee.harmonization import OPTICAL_BAND_NAMES

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.gee import GEEAPIError
from src.gee.composite import CompositeResult
from src.gee.feature_stack import (
    FeatureStackAssembler,
    FeatureStackResult,
    validate_harmonization,
)
from src.gee.indices import compute_savi
from src.gee.registry import BUILT_IN_INDICES, FeatureRegistry

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

__all__ = [
    "FeatureConfig",
    "SpectralFeatureGenerator",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_DEFAULT_SAVI_SOIL_FACTOR: float = 0.5


# ==============================================================================
# FeatureConfig
# ==============================================================================

@dataclass(frozen=True)
class FeatureConfig:
    """
    Immutable configuration controlling which spectral indices are computed.

    Each flag corresponds to one spectral index. True = compute and append.
    False = skip (the index will appear in FeatureStackResult.features_skipped).

    Attributes:
        ndwi:             Compute NDWI (McFeeters 1996).
        mndwi:            Compute MNDWI (Xu 2006). PRIMARY water/sand discriminator.
        awei_sh:          Compute AWEI_sh (Feyisa 2014). Shadow-affected water.
        awei_nsh:         Compute AWEI_nsh (Feyisa 2014). Open water.
        ndvi:             Compute NDVI (Rouse 1973). Vegetation suppressor.
        savi:             Compute SAVI (Huete 1988). Soil-adjusted vegetation.
        bsi:              Compute BSI (Rikimaru 2002). PRIMARY sand/sediment detector.
        ndmi:             Compute NDMI (Gao 1996). Surface moisture.
        ndbi:             Compute NDBI (Zha 2003). Built-up surfaces. OPTIONAL.
        savi_soil_factor: Soil brightness correction factor L for SAVI [0, 1].
                          0.5 is recommended for intermediate vegetation conditions.
    """

    ndwi:             bool  = True
    mndwi:            bool  = True
    awei_sh:          bool  = True
    awei_nsh:         bool  = True
    ndvi:             bool  = True
    savi:             bool  = True
    bsi:              bool  = True
    ndmi:             bool  = True
    ndbi:             bool  = False       # optional; off by default
    savi_soil_factor: float = _DEFAULT_SAVI_SOIL_FACTOR

    @classmethod
    def from_config(cls, config: Config) -> FeatureConfig:
        """
        Construct a FeatureConfig from a Config object.

        Reads the 'features' section of config.yaml. If the section is
        absent (config has no 'features' attribute), all defaults apply.
        This ensures backward compatibility with configs written before
        Module 6 was introduced.

        Args:
            config: Fully initialized Config object.

        Returns:
            FeatureConfig with flags sourced from config.features.
        """
        features = getattr(config, "features", None)
        if features is None:
            _LOGGER.debug(
                "No 'features' section found in config.yaml. "
                "Using FeatureConfig defaults."
            )
            return cls()

        return cls(
            ndwi=bool(getattr(features, "ndwi",             True)),
            mndwi=bool(getattr(features, "mndwi",           True)),
            awei_sh=bool(getattr(features, "awei_sh",       True)),
            awei_nsh=bool(getattr(features, "awei_nsh",     True)),
            ndvi=bool(getattr(features, "ndvi",             True)),
            savi=bool(getattr(features, "savi",             True)),
            bsi=bool(getattr(features, "bsi",               True)),
            ndmi=bool(getattr(features, "ndmi",             True)),
            ndbi=bool(getattr(features, "ndbi",             False)),
            savi_soil_factor=float(
                getattr(features, "savi_soil_factor", _DEFAULT_SAVI_SOIL_FACTOR)
            ),
        )

    def is_enabled(self, config_key: str) -> bool:
        """
        Return True if the index with the given config_key is enabled.

        Args:
            config_key: The config.yaml features key, e.g. "ndwi", "bsi".

        Returns:
            True if the corresponding flag is True.
        """
        return bool(getattr(self, config_key, False))

    def enabled_features(self) -> list[str]:
        """
        Return canonical names of all enabled spectral indices.

        Returns:
            List of index names (e.g. ["NDWI", "MNDWI", "BSI"]) in
            the same order as BUILT_IN_INDICES registration.
        """
        key_to_name = FeatureRegistry.get_config_key_map()
        return [
            name
            for key, name in key_to_name.items()
            if self.is_enabled(key)
        ]

    def disabled_features(self) -> list[str]:
        """
        Return canonical names of all disabled spectral indices.

        Returns:
            List of index names not enabled in this config.
        """
        enabled_set = set(self.enabled_features())
        return [
            meta.name
            for meta in BUILT_IN_INDICES
            if meta.name not in enabled_set
        ]


# ==============================================================================
# SpectralFeatureGenerator
# ==============================================================================

class SpectralFeatureGenerator:
    """
    Computes spectral indices and appends them to the composite image.

    Reads which indices to compute from FeatureConfig (which is read
    from config.yaml). Indices are computed in registry registration order
    to ensure consistent band ordering in the output regardless of config.

    The generator is stateless after construction and reusable across
    multiple CompositeResult inputs with the same configuration.

    Args:
        client: Initialized EarthEngineClient (used for future extensions
                where retry logic may be needed on rare EE compute failures).
        config: Fully initialized Config object. Reads config.features.

    Example:

        generator = SpectralFeatureGenerator(client, config)
        result    = generator.generate(composite_result)

        # Use a custom config for this call only:
        result = generator.generate(
            composite_result,
            feature_config=FeatureConfig(ndbi=True, savi=False),
        )
    """

    def __init__(
        self,
        client: EarthEngineClient,
        config: Config,
    ) -> None:
        self._client:         EarthEngineClient = client
        self._config:         Config            = config
        self._default_config: FeatureConfig     = FeatureConfig.from_config(config)
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._logger.debug(
            "SpectralFeatureGenerator initialized. "
            "Default enabled: %s",
            self._default_config.enabled_features(),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        composite_result: CompositeResult,
        feature_config: FeatureConfig | None = None,
    ) -> FeatureStackResult:
        """
        Compute enabled spectral indices and append them to the composite image.

        For each enabled index (in BUILT_IN_INDICES registration order):
            1. Retrieve computation function from FeatureRegistry.
            2. Call the function on the composite image.
            3. Append the single-band result to the stack.

        SAVI is handled specially: its computation function receives the
        soil_factor parameter from FeatureConfig.savi_soil_factor.

        Args:
            composite_result: CompositeResult from LandsatCompositor.
            feature_config:   Override FeatureConfig for this call.
                              If None, uses the default from config.yaml.

        Returns:
            FeatureStackResult with the stacked image and provenance.

        Raises:
            InvalidValueError:   Input band names are incompatible with
                                 harmonized index computation.
            GEEAPIError:         An index computation or addBands() failed.
        """
        cfg = feature_config if feature_config is not None else self._default_config

        # Validate harmonization (warns or raises based on band names).
        validate_harmonization(composite_result)

        # Determine which indices to compute vs. skip.
        enabled  = set(cfg.enabled_features())
        skipped  = set(cfg.disabled_features())

        self._logger.info(
            "Generating feature stack. enabled=%s skipped=%s",
            sorted(enabled),
            sorted(skipped),
        )

        # Determine composite band names for provenance tracking.
        composite_band_names = tuple(composite_result.band_names)
        if not composite_band_names:
            composite_band_names = tuple(str(b) for b in BUILT_IN_INDICES[0:0])


        # Keep the full harmonized composite available for spectral-index
        # computation, but expose only the six optical reflectance bands in
        # the final machine-learning feature stack. Thermal and QA_PIXEL are
        # preprocessing/quality-control bands and are not model inputs.
        model_base_band_names = tuple(OPTICAL_BAND_NAMES)

        try:
            model_base_image = composite_result.image.select(
                list(model_base_band_names)
            )
        except Exception as exc:
            raise GEEAPIError(
                operation="select_model_base_bands",
                reason=(
                    "Failed to select the optical model-input bands "
                    f"{list(model_base_band_names)}: {exc}"
                ),
            ) from exc

        assembler = FeatureStackAssembler(
            base_image=model_base_image,
            base_band_names=model_base_band_names,
        )

        # # Initialize assembler with the composite image.
        # assembler = FeatureStackAssembler(
        #     base_image=composite_result.image,
        #     base_band_names=composite_band_names,
        # )

        # Build a function map that handles SAVI's extra parameter.
        function_map = self._build_function_map(cfg)

        # Compute enabled indices in registry registration order.
        computed: list[str] = []
        for meta in BUILT_IN_INDICES:
            if meta.name not in enabled:
                continue

            self._logger.debug("Computing index: %s", meta.name)

            fn           = function_map[meta.name]
            index_image  = self._call_index_function(fn, composite_result.image, meta.name)
            assembler.add_index(index_image, meta.name)
            computed.append(meta.name)

        stacked_image, all_band_names, index_band_names = assembler.build()

        result = FeatureStackResult(
            image=stacked_image,
            features_computed=tuple(computed),
            features_skipped=tuple(sorted(skipped)),
            composite_band_names=composite_band_names,
            index_band_names=index_band_names,
            all_band_names=all_band_names,
            source_composite=composite_result,
            feature_config=cfg,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    def generate_single_index(
        self,
        composite_result: CompositeResult,
        index_name: str,
        soil_factor: float | None = None,
    ) -> Any:
        """
        Compute and return a single spectral index without stacking.

        Useful for ad-hoc exploration or when only one index is needed.
        Does NOT append the result to the composite image.

        Args:
            composite_result: CompositeResult with harmonized bands.
            index_name:       Canonical index name, e.g. "BSI".
            soil_factor:      Override SAVI soil factor. Only used for SAVI.

        Returns:
            Single-band ee.Image named after the index.

        Raises:
            InvalidValueError:   index_name is not registered.
            GEEAPIError:         Computation failed.
        """
        validate_harmonization(composite_result)

        if index_name == "SAVI":
            factor = (
                soil_factor
                if soil_factor is not None
                else self._default_config.savi_soil_factor
            )
            return compute_savi(composite_result.image, soil_factor=factor)

        fn = FeatureRegistry.get_function(index_name)
        return self._call_index_function(fn, composite_result.image, index_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_function_map(
        self,
        cfg: FeatureConfig,
    ) -> dict[str, Callable[..., Any]]:
        """
        Build a {name: callable} map for all registered indices.

        SAVI is wrapped in a functools.partial to embed the soil_factor
        from the FeatureConfig. All other functions are used as-is.

        Args:
            cfg: The active FeatureConfig (provides savi_soil_factor).

        Returns:
            Dict mapping each registered index name to its callable.
        """
        function_map: dict[str, Callable[..., Any]] = {}
        for name in FeatureRegistry.names():
            if name == "SAVI":
                function_map[name] = functools.partial(
                    compute_savi,
                    soil_factor=cfg.savi_soil_factor,
                )
            else:
                function_map[name] = FeatureRegistry.get_function(name)
        return function_map

    def _call_index_function(
        self,
        fn: Callable[..., Any],
        image: Any,
        index_name: str,
    ) -> Any:
        """
        Call a single index function with exception wrapping.

        Catches any exception from the computation function and wraps it
        in GEEAPIError with the index name for clear error attribution.

        Args:
            fn:         The computation callable.
            image:      The ee.Image to compute the index on.
            index_name: Name of the index (for error messages).

        Returns:
            Single-band ee.Image from the computation function.

        Raises:
            GEEAPIError: The computation raised any exception.
        """
        try:
            return fn(image)
        except GEEAPIError:
            raise
        except Exception as exc:
            raise GEEAPIError(
                operation=f"generate_index_{index_name}",
                reason=f"Index computation for '{index_name}' failed: {exc}",
            ) from exc