"""
Spectral band reading and classification engine for Module 9.

SpectralBandReader reads a multi-band feature patch GeoTIFF (produced by
Modules 7 and 8) into a SpectralBandData object. Band names are read from
the GeoTIFF band descriptions written by Module 7's DatasetExporter so
that no band name is hardcoded here.

SpectralClassificationEngine applies a RuleEngine to the band data and
assembles the per-pixel class_map (Level 2 classification confidence).
An optional ClassificationContext is forwarded to each rule to support
future seasonal or sensor-specific adaptation without architecture changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.labels.contracts import (
    ClassificationContext,
    ClassificationResult,
    SpectralBandData,
)
from src.labels.rules import RuleEngine

__all__ = ["SpectralBandReader", "SpectralClassificationEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class SpectralBandReader:
    """
    Reads a multi-band patch GeoTIFF into a SpectralBandData object.

    Bands are keyed by their rasterio band description. Bands without a
    description are keyed as "band_<i>" where i is the 1-based band index.
    Pixels equal to the rasterio nodata value are set to NaN.
    """

    def read(self, patch_path: Path) -> SpectralBandData:
        """
        Read all bands from a patch GeoTIFF.

        Args:
            patch_path: Path to the source feature-stack GeoTIFF.

        Returns:
            SpectralBandData with all named band arrays.

        Raises:
            OSError: rasterio is not installed or the file cannot be opened.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise OSError("rasterio is not installed.") from exc

        try:
            with rasterio.open(patch_path) as ds:
                h, w      = ds.height, ds.width
                crs       = ds.crs.to_string() if ds.crs else ""
                transform = ds.transform

                bands: dict[str, np.ndarray] = {}
                for i in range(1, ds.count + 1):
                    desc = ds.descriptions[i - 1]
                    name = desc if desc else f"band_{i}"
                    data = ds.read(i).astype(np.float32)
                    if ds.nodata is not None:
                        data[data == ds.nodata] = np.nan
                    bands[name] = data

            band_names = tuple(bands.keys())
            _LOGGER.debug(
                "Read %d band(s) from %s: %s",
                len(bands), patch_path.name, band_names,
            )
            return SpectralBandData(
                bands=bands, height=h, width=w,
                crs=crs, transform=transform, band_names=band_names,
            )
        except Exception as exc:
            raise OSError(
                f"Failed to read patch GeoTIFF '{patch_path}': {exc}"
            ) from exc


class SpectralClassificationEngine:
    """
    Classifies pixels in a patch using the RuleEngine (Level 2 confidence).

    For each pixel the rule with the highest evidence score wins. Pixels
    where all rules return zero evidence are marked as unclassified. Pixels
    where all input bands are NaN are marked as nodata.

    Args:
        rule_engine: Configured RuleEngine built via RuleEngine.from_config().
    """

    def __init__(self, rule_engine: RuleEngine) -> None:
        self._rule_engine = rule_engine
        self._reader      = SpectralBandReader()
        self._logger: logging.Logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Any) -> SpectralClassificationEngine:
        """Build from Config using the registry-based RuleEngine.from_config()."""
        return cls(RuleEngine.from_config(config))

    def classify(
        self,
        patch_path: Path,
        context:    ClassificationContext | None = None,
    ) -> ClassificationResult:
        """
        Classify all pixels in a patch GeoTIFF (Level 2 output).

        Args:
            patch_path: Path to the source feature-stack GeoTIFF.
            context:    Optional ClassificationContext forwarded to each rule.
                        All current rules ignore it; reserved for future
                        seasonal or sensor-specific adaptation.

        Returns:
            ClassificationResult with class_map, confidence_map (Level 2),
            and individual rule results (Level 1 evidence).
            resolved_confidence_map is None until ConflictResolver.resolve()
            is called.

        Raises:
            OSError: Cannot read the patch file.
        """
        band_data    = self._reader.read(patch_path)
        rule_results = self._rule_engine.apply_all(band_data, context=context)

        h, w = band_data.height, band_data.width
        class_map      = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        # Assign each pixel to the rule with the highest evidence score.
        for result in rule_results:
            better = result.confidence > confidence_map
            class_map[better]      = result.class_id
            confidence_map[better] = result.confidence[better]

        # Nodata: pixels where ALL input bands are NaN.
        if band_data.bands:
            nodata_mask = np.ones((h, w), dtype=bool)
            for arr in band_data.bands.values():
                nodata_mask &= np.isnan(arr)
        else:
            nodata_mask = np.zeros((h, w), dtype=bool)

        unclassified_mask = confidence_map == 0.0

        self._logger.debug(
            "Classification complete: unclassified=%d/%d, nodata=%d/%d",
            int(unclassified_mask.sum()), h * w,
            int(nodata_mask.sum()), h * w,
        )
        return ClassificationResult(
            class_map=class_map,
            confidence_map=confidence_map,
            rule_results=rule_results,
            unclassified_mask=unclassified_mask,
            nodata_mask=nodata_mask,
            resolved_confidence_map=None,   # populated by ConflictResolver
        )


# """
# Spectral band reading and classification engine for Module 9.

# Refinement: SpectralClassificationEngine.classify() and
# SpectralBandReader.read() now accept an optional ClassificationContext
# that is forwarded to each rule via RuleEngine.apply_all(). No temporal
# classification logic is added; the parameter exists solely so the
# architecture supports it without future method-signature changes.
# """

# from __future__ import annotations

# import logging
# from pathlib import Path
# from typing import Any

# import numpy as np

# from src.labels.contracts import ClassificationContext, ClassificationResult, SpectralBandData
# from src.labels.rules import RuleEngine

# __all__ = ["SpectralBandReader", "SpectralClassificationEngine"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)


# class SpectralBandReader:
#     """
#     Reads a multi-band patch GeoTIFF into a SpectralBandData object.

#     Unchanged from v1 — reads all bands from rasterio descriptions.
#     """

#     def read(self, patch_path: Path) -> SpectralBandData:
#         """
#         Read all bands from a patch GeoTIFF.

#         Raises:
#             OSError: rasterio is not installed or file cannot be opened.
#         """
#         try:
#             import rasterio
#         except ImportError as exc:
#             raise OSError("rasterio is not installed.") from exc

#         try:
#             with rasterio.open(patch_path) as ds:
#                 h, w      = ds.height, ds.width
#                 crs       = ds.crs.to_string() if ds.crs else ""
#                 transform = ds.transform

#                 bands: dict[str, np.ndarray] = {}
#                 for i in range(1, ds.count + 1):
#                     desc = ds.descriptions[i - 1]
#                     name = desc if desc else f"band_{i}"
#                     data = ds.read(i).astype(np.float32)
#                     if ds.nodata is not None:
#                         data[data == ds.nodata] = np.nan
#                     bands[name] = data

#             band_names = tuple(bands.keys())
#             _LOGGER.debug(
#                 "Read %d band(s) from %s: %s",
#                 len(bands), patch_path.name, band_names,
#             )
#             return SpectralBandData(
#                 bands=bands, height=h, width=w,
#                 crs=crs, transform=transform, band_names=band_names,
#             )
#         except Exception as exc:
#             raise OSError(
#                 f"Failed to read patch GeoTIFF: {patch_path}: {exc}"
#             ) from exc


# class SpectralClassificationEngine:
#     """
#     Classifies pixels in a patch using the RuleEngine.

#     Refinement: classify() now accepts optional ClassificationContext and
#     forwards it to RuleEngine.apply_all(). No temporal logic is implemented.

#     Args:
#         rule_engine: Configured RuleEngine (built via RuleEngine.from_config()
#                      which uses RuleRegistry — no hardcoded rule list).
#     """

#     def __init__(self, rule_engine: RuleEngine) -> None:
#         self._rule_engine = rule_engine
#         self._reader      = SpectralBandReader()
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @classmethod
#     def from_config(cls, config: Any) -> SpectralClassificationEngine:
#         """Build from Config using RuleEngine.from_config() (registry-based)."""
#         return cls(RuleEngine.from_config(config))

#     def classify(
#         self,
#         patch_path: Path,
#         context:    ClassificationContext | None = None,
#     ) -> ClassificationResult:
#         """
#         Classify all pixels in a patch GeoTIFF.

#         Args:
#             patch_path: Path to the source patch GeoTIFF.
#             context:    Optional ClassificationContext forwarded to each rule.
#                         Ignored by all current rules; reserved for future
#                         seasonal/temporal classification.

#         Returns:
#             ClassificationResult with class_map, confidence_map,
#             and individual rule results.

#         Raises:
#             OSError: Cannot read the patch file.
#         """
#         band_data    = self._reader.read(patch_path)
#         rule_results = self._rule_engine.apply_all(band_data, context=context)

#         h, w = band_data.height, band_data.width
#         class_map      = np.zeros((h, w), dtype=np.uint8)
#         confidence_map = np.zeros((h, w), dtype=np.float32)

#         # Assign each pixel to the rule with the highest evidence score.
#         for result in rule_results:
#             better = result.confidence > confidence_map
#             class_map[better]      = result.class_id
#             confidence_map[better] = result.confidence[better]

#         # Nodata: pixels where ALL input bands are NaN.
#         if band_data.bands:
#             nodata_mask = np.ones((h, w), dtype=bool)
#             for arr in band_data.bands.values():
#                 nodata_mask &= np.isnan(arr)
#         else:
#             nodata_mask = np.zeros((h, w), dtype=bool)

#         unclassified_mask = confidence_map == 0.0

#         self._logger.debug(
#             "Classification: unclassified=%d/%d, nodata=%d/%d",
#             int(unclassified_mask.sum()), h * w,
#             int(nodata_mask.sum()), h * w,
#         )
#         return ClassificationResult(
#             class_map=class_map,
#             confidence_map=confidence_map,
#             rule_results=rule_results,
#             unclassified_mask=unclassified_mask,
#             nodata_mask=nodata_mask,
#         )