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

            visual_paths = self._save_visual_products(
                prediction,
                sid,
            )
            paths.extend(visual_paths)

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


    def _save_visual_products(
        self,
        pred: SamplePrediction,
        sid: str,
    ) -> list[str]:
        """
        Export presentation-ready segmentation products.

        Products:
            *_color.png       class-colored prediction
            *_source_rgb.png  RGB preview of source patch
            *_overlay.png     prediction blended over source
            *_comparison.png  source, prediction, and overlay
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            _LOGGER.warning(
                "PredictionExporter: Pillow unavailable; "
                "visual products skipped."
            )
            return []

        palette = np.asarray(
            [
                [128, 128, 128],  # background
                [0, 119, 190],    # water
                [255, 200, 87],   # sand
                [34, 139, 34],    # vegetation
            ],
            dtype=np.uint8,
        )

        mask = np.asarray(
            pred.predicted_mask,
            dtype=np.int64,
        )

        safe_mask = np.clip(
            mask,
            0,
            len(palette) - 1,
        )

        color = palette[safe_mask]
        color_img = Image.fromarray(
            color,
            mode="RGB",
        )

        written: list[str] = []

        color_path = self._out_dir / f"{sid}_color.png"

        try:
            color_img.save(color_path)
            written.append(str(color_path))
        except Exception as exc:
            _LOGGER.warning(
                "PredictionExporter: colored mask save failed: %s",
                exc,
            )

        source = self._load_source_rgb(
            pred.patch_path,
            mask.shape,
        )

        if source is None:
            _LOGGER.warning(
                "PredictionExporter: source preview unavailable "
                "for sample %s.",
                pred.sample_id,
            )
            return written

        source_img = Image.fromarray(
            source,
            mode="RGB",
        )

        source_path = (
            self._out_dir /
            f"{sid}_source_rgb.png"
        )

        try:
            source_img.save(source_path)
            written.append(str(source_path))
        except Exception as exc:
            _LOGGER.warning(
                "PredictionExporter: source preview save failed: %s",
                exc,
            )

        overlay = (
            source.astype(np.float32) * 0.50
            + color.astype(np.float32) * 0.50
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        overlay_img = Image.fromarray(
            overlay,
            mode="RGB",
        )

        overlay_path = (
            self._out_dir /
            f"{sid}_overlay.png"
        )

        try:
            overlay_img.save(overlay_path)
            written.append(str(overlay_path))
        except Exception as exc:
            _LOGGER.warning(
                "PredictionExporter: overlay save failed: %s",
                exc,
            )

        width, height = source_img.size
        header_height = 42

        comparison = Image.new(
            "RGB",
            (
                width * 3,
                height + header_height,
            ),
            "white",
        )

        comparison.paste(
            source_img,
            (0, header_height),
        )

        comparison.paste(
            color_img,
            (width, header_height),
        )

        comparison.paste(
            overlay_img,
            (width * 2, header_height),
        )

        draw = ImageDraw.Draw(comparison)

        draw.text(
            (10, 12),
            "SOURCE RGB",
            fill="black",
        )

        draw.text(
            (width + 10, 12),
            "PREDICTED CLASSES",
            fill="black",
        )

        draw.text(
            (width * 2 + 10, 12),
            "SEGMENTATION OVERLAY",
            fill="black",
        )

        comparison_path = (
            self._out_dir /
            f"{sid}_comparison.png"
        )

        try:
            comparison.save(comparison_path)
            written.append(str(comparison_path))
        except Exception as exc:
            _LOGGER.warning(
                "PredictionExporter: comparison save failed: %s",
                exc,
            )

        return written

    @staticmethod
    def _load_source_rgb(
        patch_path: str,
        target_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """
        Read an RGB preview from a multispectral source patch.

        The first three source bands are used for the preview. Each band
        receives robust percentile stretching so reflectance data becomes
        visible in an ordinary 8-bit PNG.
        """
        if not patch_path:
            return None

        path = Path(patch_path)

        if not path.is_file():
            return None

        try:
            import rasterio

            with rasterio.open(path) as src:
                count = min(3, src.count)

                if count < 1:
                    return None

                bands = src.read(
                    list(range(1, count + 1))
                ).astype(np.float32)

        except Exception as exc:
            _LOGGER.warning(
                "PredictionExporter: source patch read failed "
                "for %s: %s",
                patch_path,
                exc,
            )
            return None

        if bands.shape[0] == 1:
            bands = np.repeat(
                bands,
                3,
                axis=0,
            )
        elif bands.shape[0] == 2:
            bands = np.concatenate(
                [bands, bands[-1:]],
                axis=0,
            )

        rgb = np.zeros(
            (
                bands.shape[1],
                bands.shape[2],
                3,
            ),
            dtype=np.uint8,
        )

        for channel in range(3):
            band = bands[channel]

            valid = band[
                np.isfinite(band)
            ]

            if valid.size == 0:
                continue

            low, high = np.percentile(
                valid,
                [2.0, 98.0],
            )

            if high <= low:
                high = low + 1.0

            stretched = (
                (band - low)
                / (high - low)
            )

            stretched = np.nan_to_num(
                stretched,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            rgb[:, :, channel] = (
                np.clip(
                    stretched,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(np.uint8)

        expected_h, expected_w = target_shape

        if rgb.shape[:2] != (
            expected_h,
            expected_w,
        ):
            from PIL import Image

            rgb = np.asarray(
                Image.fromarray(rgb).resize(
                    (
                        expected_w,
                        expected_h,
                    ),
                    resample=Image.Resampling.BILINEAR,
                )
            )

        return rgb


def _sanitise(sid: str) -> str:
    """Replace filesystem-unsafe characters in sample_id."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in sid) or "sample"
