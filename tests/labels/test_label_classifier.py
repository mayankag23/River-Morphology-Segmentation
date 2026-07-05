"""
Unit tests for src/labels/classifier.py.

Uses real rasterio I/O via direct rasterio writes for patch fixtures.

Run:
    pytest tests/labels/test_label_classifier.py -v \
        --cov=src/labels/classifier --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.labels.classifier import SpectralBandReader, SpectralClassificationEngine
from tests.conftest import make_valid_config, write_config


def _write_spectral_patch(path: Path, band_values: dict[str, float]) -> Path:
    """Write a synthetic multi-band patch GeoTIFF with named bands."""
    bands  = list(band_values.keys())
    values = list(band_values.values())
    H, W   = 8, 8
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": W, "height": H,
        "count": len(bands), "crs": CRS.from_string("EPSG:4326"),
        "transform": Affine(0.001, 0, 87, 0, -0.001, 26.5),
    }
    with rasterio.open(path, "w", **profile) as ds:
        for i, (name, value) in enumerate(band_values.items(), start=1):
            data = np.full((H, W), value, dtype=np.float32)
            ds.write(data, i)
            ds.set_band_description(i, name)
    return path


def _make_config(tmp_path: Path):
    from src.core.config import Config
    data = make_valid_config()
    data["labels"] = {
        "nodata_value": 255,
        "rules": {
            "water":      {"enabled": True, "mndwi_threshold": 0.2, "ndwi_threshold": 0.0,
                           "awei_nsh_threshold": 0.0, "awei_sh_threshold": 0.0,
                           "ndvi_max_threshold": 0.05, "mndwi_weight": 0.40,
                           "ndwi_weight": 0.20, "awei_nsh_weight": 0.15,
                           "awei_sh_weight": 0.05, "ndvi_weight": 0.20,
                           "confidence_scale": 0.3, "min_confidence": 0.40},
            "sand":       {"enabled": True, "mndwi_max_threshold": 0.0,
                           "bsi_threshold": 0.0, "ndvi_max_threshold": 0.25,
                           "ndwi_max_threshold": 0.0, "mndwi_weight": 0.30,
                           "bsi_weight": 0.35, "ndvi_weight": 0.25,
                           "ndwi_weight": 0.10, "confidence_scale": 0.3, "min_confidence": 0.30},
            "vegetation": {"enabled": True, "ndvi_threshold": 0.2, "savi_threshold": 0.1,
                           "ndmi_threshold": 0.0, "ndvi_weight": 0.50, "savi_weight": 0.30,
                           "ndmi_weight": 0.20, "confidence_scale": 0.25, "min_confidence": 0.35},
            "background": {"enabled": True, "min_confidence": 0.10},
        },
        "conflict_resolution": {"strategy": "highest_confidence",
                                "water_priority": 0, "vegetation_priority": 1,
                                "sand_priority": 2, "background_priority": 3},
    }
    return Config(config_path=write_config(tmp_path, data))


class TestSpectralBandReader:
    def test_reads_named_bands(self, tmp_path: Path) -> None:
        patch = _write_spectral_patch(tmp_path / "p.tif",
                                       {"MNDWI": 0.5, "NDWI": 0.3, "NDVI": -0.1})
        reader = SpectralBandReader()
        result = reader.read(patch)
        assert "MNDWI" in result.bands
        assert "NDWI"  in result.bands
        assert result.height == 8
        assert result.width  == 8

    def test_band_values_correct(self, tmp_path: Path) -> None:
        patch = _write_spectral_patch(tmp_path / "p.tif", {"MNDWI": 0.5})
        reader = SpectralBandReader()
        result = reader.read(patch)
        np.testing.assert_allclose(result.bands["MNDWI"], 0.5, atol=1e-5)

    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        reader = SpectralBandReader()
        with pytest.raises(OSError):
            reader.read(tmp_path / "nonexistent.tif")

    def test_band_names_tuple(self, tmp_path: Path) -> None:
        patch  = _write_spectral_patch(tmp_path / "p.tif", {"A": 0.1, "B": 0.2})
        reader = SpectralBandReader()
        result = reader.read(patch)
        assert set(result.band_names) == {"A", "B"}


class TestSpectralClassificationEngine:
    def test_water_patch_classified_as_water(self, tmp_path: Path) -> None:
        patch = _write_spectral_patch(tmp_path / "p.tif", {
            "MNDWI": 0.7, "NDWI": 0.5, "AWEI_nsh": 0.3,
            "AWEI_sh": 0.2, "NDVI": -0.1, "BSI": -0.2, "SAVI": 0.0, "NDMI": 0.0,
        })
        engine = SpectralClassificationEngine.from_config(_make_config(tmp_path))
        result = engine.classify(patch)
        # Most pixels should be water (class_id=1)
        water_fraction = (result.class_map == 1).mean()
        assert water_fraction > 0.5

    def test_sand_patch_classified_as_sand(self, tmp_path: Path) -> None:
        patch = _write_spectral_patch(tmp_path / "p.tif", {
            "MNDWI": -0.4, "NDWI": -0.3, "NDVI": 0.05,
            "BSI": 0.3, "SAVI": 0.05, "NDMI": -0.1,
        })
        engine = SpectralClassificationEngine.from_config(_make_config(tmp_path))
        result = engine.classify(patch)
        sand_fraction = (result.class_map == 2).mean()
        assert sand_fraction > 0.5

    def test_vegetation_patch_classified_as_vegetation(self, tmp_path: Path) -> None:
        patch = _write_spectral_patch(tmp_path / "p.tif", {
            "NDVI": 0.6, "SAVI": 0.5, "NDMI": 0.3,
            "MNDWI": -0.2, "BSI": -0.1, "NDWI": -0.2,
        })
        engine = SpectralClassificationEngine.from_config(_make_config(tmp_path))
        result = engine.classify(patch)
        veg_fraction = (result.class_map == 3).mean()
        assert veg_fraction > 0.5

    def test_class_map_dtype_uint8(self, tmp_path: Path) -> None:
        patch  = _write_spectral_patch(tmp_path / "p.tif", {"MNDWI": 0.5, "NDWI": 0.3})
        engine = SpectralClassificationEngine.from_config(_make_config(tmp_path))
        result = engine.classify(patch)
        assert result.class_map.dtype == np.uint8

    def test_confidence_map_float32(self, tmp_path: Path) -> None:
        patch  = _write_spectral_patch(tmp_path / "p.tif", {"MNDWI": 0.5})
        engine = SpectralClassificationEngine.from_config(_make_config(tmp_path))
        result = engine.classify(patch)
        assert result.confidence_map.dtype == np.float32

    def test_missing_patch_raises(self, tmp_path: Path) -> None:
        engine = SpectralClassificationEngine.from_config(_make_config(tmp_path))
        with pytest.raises(OSError):
            engine.classify(tmp_path / "nonexistent.tif")