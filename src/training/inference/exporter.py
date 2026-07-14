"""
Prediction exporter for Module 16.

PredictionExporter writes SamplePrediction outputs to disk.
Supported formats:
    .npy     Numpy array of predicted mask (H, W) uint8.
    .tif     GeoTIFF with CRS and affine transform copied from the source
             patch GeoTIFF (requires rasterio). Predicted mask is band 1;
             probability bands (one per class) follow; confidence is the
             final band.
    .png     8-bit PNG of predicted class IDs (requires PIL/Pillow).

The GeoTIFF writer copies the CRS and affine transform from the original
patch GeoTIFF (patch_path field of SamplePrediction). This reuses the
project's existing geospatial coordinate system without duplicating any
GeoTIFF writing logic from Module 7 — instead we delegate the CRS/affine
reading to rasterio directly.

Design rule: no GeoTIFF writing logic is duplicated from Module 7. This
module reads CRS/affine from the source patch and writes the prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.training.inference.contracts import InferenceConfig, SamplePrediction

__all__ = ["PredictionExporter"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class PredictionExporter:
    """
    Exports SamplePrediction objects to disk in configured formats.

    Args:
        config:      InferenceConfig with output_dir and format flags.
        class_names: Ordered class names for GeoTIFF band descriptions.
    """

    def __init__(
        self,
        config:      InferenceConfig,
        class_names: tuple[str, ...],
    ) -> None:
        self._config      = config
        self._class_names = class_names
        self._out_dir     = Path(config.output_dir).resolve()

    def export(self, prediction: SamplePrediction) -> list[str]:
        """
        Export one prediction in all configured formats.

        Args:
            prediction: SamplePrediction to export.

        Returns:
            List of absolute path strings for all written files.
        """
        self._out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        sid  = _sanitise(prediction.sample_id)

        if self._config.export_numpy:
            p = self._save_numpy(prediction, sid)
            if p:
                paths.append(p)

        if self._config.export_geotiff:
            p = self._save_geotiff(prediction, sid)
            if p:
                paths.append(p)

        if self._config.export_png:
            p = self._save_png(prediction, sid)
            if p:
                paths.append(p)

        return paths

    def export_all(
        self,
        predictions: list[SamplePrediction],
    ) -> list[SamplePrediction]:
        """
        Export all predictions in batch.

        Mutates each SamplePrediction's exported_paths list in place and
        returns the same list (for chaining).
        """
        seen_ids: set[str] = set()

        for index, pred in enumerate(predictions):
            sid = _sanitise(pred.sample_id)

            if sid in seen_ids:
                raise ValueError(
                    "PredictionExporter: duplicate sample_id would overwrite "
                    f"existing artifacts: {pred.sample_id!r} "
                    f"(prediction index {index})."
                )

            seen_ids.add(sid)
            paths = self.export(pred)
            pred.exported_paths.extend(paths)

        return predictions

    # ------------------------------------------------------------------
    # Private writers
    # ------------------------------------------------------------------

    def _save_numpy(self, pred: SamplePrediction, sid: str) -> str | None:
        """Save predicted mask as (H, W) uint8 .npy file."""
        path = self._out_dir / f"{sid}_mask.npy"
        try:
            np.save(str(path), pred.predicted_mask)
            _LOGGER.debug("PredictionExporter: saved numpy -> %s", path.name)
            return str(path)
        except Exception as exc:
            _LOGGER.warning("PredictionExporter: numpy save failed: %s", exc)
            return None

    def _save_geotiff(self, pred: SamplePrediction, sid: str) -> str | None:
        """
        Save prediction as a multi-band GeoTIFF.

        Band layout:
            Band 1:        Predicted class ID mask (uint8)
            Bands 2..C+1:  Per-class probability maps (float32)
            Band C+2:      Confidence map (float32)

        CRS and affine transform are copied from the source patch GeoTIFF.
        When patch_path is empty or rasterio is unavailable, the file is
        written without geospatial reference (plain TIFF).
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            _LOGGER.warning(
                "PredictionExporter: rasterio unavailable; GeoTIFF export skipped."
            )
            return None

        path       = self._out_dir / f"{sid}_prediction.tif"
        C, H, W    = pred.probabilities.shape
        n_bands    = 1 + C + 1   # mask + probs + confidence

        crs       = None
        transform = None

        if pred.patch_path:
            try:
                with rasterio.open(pred.patch_path) as src:
                    crs       = src.crs
                    transform = src.transform
            except Exception as exc:
                _LOGGER.warning(
                    "PredictionExporter: could not read CRS from %s: %s",
                    pred.patch_path, exc,
                )

        profile: dict = {
            "driver":  "GTiff",
            "count":   n_bands,
            "height":  H,
            "width":   W,
            "dtype":   "float32",
            "compress": "deflate",
        }
        if crs is not None:
            profile["crs"] = crs
        if transform is not None:
            profile["transform"] = transform

        try:
            with rasterio.open(str(path), "w", **profile) as dst:
                # Band 1: mask as float32 (class IDs).
                dst.write(pred.predicted_mask.astype(np.float32), 1)
                dst.update_tags(1, band_description="predicted_class_id")

                # Bands 2..C+1: per-class probabilities.
                for c in range(C):
                    dst.write(pred.probabilities[c], c + 2)
                    cls_name = self._class_names[c] if c < len(self._class_names) else f"class_{c}"
                    dst.update_tags(c + 2, band_description=f"probability_{cls_name}")

                # Band C+2: confidence.
                dst.write(pred.confidence, C + 2)
                dst.update_tags(C + 2, band_description="confidence")

                # Write temporal metadata as GeoTIFF tags.
                dst.update_tags(
                    acquisition_date  = pred.acquisition_date,
                    season            = pred.season,
                    hydrological_year = str(pred.hydrological_year),
                    sensor            = pred.sensor,
                    river_name        = pred.river_name,
                    reach_id          = pred.reach_id,
                    basin_id          = pred.basin_id,
                    aoi_id            = pred.aoi_id,
                    sample_id         = pred.sample_id,
                )

            _LOGGER.debug("PredictionExporter: saved GeoTIFF -> %s", path.name)
            return str(path)

        except Exception as exc:
            _LOGGER.warning("PredictionExporter: GeoTIFF save failed: %s", exc)
            return None

    def _save_png(self, pred: SamplePrediction, sid: str) -> str | None:
        """Save predicted class-ID mask as an 8-bit grayscale PNG."""
        try:
            from PIL import Image
        except ImportError:
            _LOGGER.warning(
                "PredictionExporter: Pillow unavailable; PNG export skipped."
            )
            return None

        path = self._out_dir / f"{sid}_mask.png"
        try:
            img = Image.fromarray(pred.predicted_mask, mode="L")
            img.save(str(path))
            _LOGGER.debug("PredictionExporter: saved PNG -> %s", path.name)
            return str(path)
        except Exception as exc:
            _LOGGER.warning("PredictionExporter: PNG save failed: %s", exc)
            return None


def _sanitise(sid: str) -> str:
    """Replace filesystem-unsafe characters in sample_id."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in sid) or "sample"
