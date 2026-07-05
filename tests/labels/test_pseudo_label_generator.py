"""
Unit tests for src/labels/generator.py.

Uses real rasterio I/O for patch and mask files.

Run:
    pytest tests/labels/test_pseudo_label_generator.py -v \
        --cov=src/labels/generator --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.labels.contracts import PseudoLabelResult
from src.labels.generator import PseudoLabelGenerator
from src.labels.schema import ClassDefinition, ClassSchema
from tests.conftest import make_valid_config, write_config


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water",      (0, 119, 190)),
        ClassDefinition(2, "sand",       (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _write_water_patch(path: Path) -> Path:
    H, W   = 8, 8
    bands  = {"MNDWI": 0.7, "NDWI": 0.5, "AWEI_nsh": 0.4, "AWEI_sh": 0.2,
              "NDVI": -0.1, "BSI": -0.3, "SAVI": 0.0, "NDMI": 0.1}
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": W, "height": H,
        "count": len(bands), "crs": CRS.from_string("EPSG:4326"),
        "transform": Affine(0.001, 0, 87, 0, -0.001, 26.5),
    }
    with rasterio.open(path, "w", **profile) as ds:
        for i, (name, val) in enumerate(bands.items(), start=1):
            ds.write(np.full((H, W), val, dtype=np.float32), i)
            ds.set_band_description(i, name)
    return path


def _make_config(tmp_path: Path):
    from src.core.config import Config
    data = make_valid_config()
    data["model"]["num_classes"] = 4
    data["classes"] = {
        "num_classes": 4,
        "labels": {"background": 0, "water": 1, "sand": 2, "vegetation": 3},
        "names": ["background", "water", "sand", "vegetation"],
        "colors": {"background": [0, 0, 0], "water": [0, 0, 1],
                   "sand": [1, 1, 0], "vegetation": [0, 1, 0]},
    }
    data["labels"] = {
        "nodata_value": 255, "mask_filename_pattern": "{patch_id}_mask.tif",
        "default_label_version": "1.0.0", "default_annotator": "spectral_rule_engine",
        "default_confidence": 0.7, "default_confidence_source": "automatic",
        "min_distinct_classes": 1, "reject_single_class_masks": False,
        "max_nodata_ratio": 0.9, "output_formats": ["csv"],
        "ratios": {}, "bare_sediment_numerator": [], "bare_sediment_denominator": [],
        "rules": {
            "water": {"enabled": True, "mndwi_threshold": 0.2, "ndwi_threshold": 0.0,
                      "awei_nsh_threshold": 0.0, "awei_sh_threshold": 0.0,
                      "ndvi_max_threshold": 0.05, "mndwi_weight": 0.40,
                      "ndwi_weight": 0.20, "awei_nsh_weight": 0.15, "awei_sh_weight": 0.05,
                      "ndvi_weight": 0.20, "confidence_scale": 0.3, "min_confidence": 0.40},
            "sand": {"enabled": True, "mndwi_max_threshold": 0.0, "bsi_threshold": 0.0,
                     "ndvi_max_threshold": 0.25, "ndwi_max_threshold": 0.0,
                     "mndwi_weight": 0.30, "bsi_weight": 0.35, "ndvi_weight": 0.25,
                     "ndwi_weight": 0.10, "confidence_scale": 0.3, "min_confidence": 0.30},
            "vegetation": {"enabled": True, "ndvi_threshold": 0.2, "savi_threshold": 0.1,
                           "ndmi_threshold": 0.0, "ndvi_weight": 0.50, "savi_weight": 0.30,
                           "ndmi_weight": 0.20, "confidence_scale": 0.25, "min_confidence": 0.35},
            "background": {"enabled": True, "min_confidence": 0.10},
        },
        "conflict_resolution": {"strategy": "highest_confidence",
                                "water_priority": 0, "vegetation_priority": 1,
                                "sand_priority": 2, "background_priority": 3},
        "morphology": {"enabled": False},
        "quality": {"min_valid_pixel_ratio": 0.1, "min_quality_score": 0.0,
                    "max_unclassified_ratio": 0.9, "min_class_pixels": 1},
        "confidence": {"min_pixel_confidence": 0.0, "min_mask_confidence": 0.0},
        "generation": {"pseudo_label_version": "1.0.0",
                       "rule_engine_version": "1.0.0",
                       "generation_method": "spectral_rules"},
    }
    return Config(config_path=write_config(tmp_path, data))


class TestPseudoLabelGenerator:
    def test_generates_mask_file(self, tmp_path: Path) -> None:
        patch      = _write_water_patch(tmp_path / "patch.tif")
        mask_path  = tmp_path / "mask.tif"
        gen        = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        result     = gen.generate(patch, "p1", mask_path)
        assert mask_path.exists()
        assert isinstance(result, PseudoLabelResult)

    def test_mask_has_correct_dtype(self, tmp_path: Path) -> None:
        patch  = _write_water_patch(tmp_path / "patch.tif")
        mask   = tmp_path / "mask.tif"
        gen    = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        gen.generate(patch, "p1", mask)
        with rasterio.open(mask) as ds:
            assert ds.dtypes[0] == "uint8"

    def test_mask_same_dimensions_as_patch(self, tmp_path: Path) -> None:
        patch  = _write_water_patch(tmp_path / "patch.tif")
        mask   = tmp_path / "mask.tif"
        gen    = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        gen.generate(patch, "p1", mask)
        with rasterio.open(patch) as src, rasterio.open(mask) as msk:
            assert (src.width, src.height) == (msk.width, msk.height)

    def test_water_patch_produces_water_mask(self, tmp_path: Path) -> None:
        patch  = _write_water_patch(tmp_path / "patch.tif")
        mask   = tmp_path / "mask.tif"
        gen    = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        result = gen.generate(patch, "p1", mask)
        with rasterio.open(mask) as ds:
            class_map     = ds.read(1)
            water_fraction = (class_map == 1).mean()
        assert water_fraction > 0.5, f"Expected water > 50%, got {water_fraction:.0%}"

    def test_result_has_crs(self, tmp_path: Path) -> None:
        patch  = _write_water_patch(tmp_path / "patch.tif")
        mask   = tmp_path / "mask.tif"
        gen    = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        result = gen.generate(patch, "p1", mask)
        assert result.crs == "EPSG:4326"

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        patch  = _write_water_patch(tmp_path / "patch.tif")
        mask   = tmp_path / "mask.tif"
        gen    = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        result = gen.generate(patch, "p1", mask)
        with pytest.raises((AttributeError, TypeError)):
            result.patch_id = "other"  # type: ignore[misc]

    def test_missing_patch_raises_oserror(self, tmp_path: Path) -> None:
        gen = PseudoLabelGenerator.from_config(_make_config(tmp_path), _schema())
        with pytest.raises(OSError):
            gen.generate(tmp_path / "nonexistent.tif", "p1", tmp_path / "mask.tif")