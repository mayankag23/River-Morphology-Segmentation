"""
Shared pytest fixtures and helper functions for Module 2+ test files.

Module 1 tests (test_config.py, frozen) define their own local fixtures
that take precedence over these within that file. These fixtures are
consumed by test_directories.py, test_bootstrap.py, and test_main.py.

Note: make_valid_config() here mirrors the canonical version in
test_config.py to ensure consistent config structure across all tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


def make_valid_config() -> dict[str, Any]:
    """
    Return a complete minimal valid configuration dictionary.

    AOI, date range, and normalization stats are intentionally null,
    matching the valid pre-setup state checked by the Config validator.
    """
    return {
        "project": {
            "name":        "river_morphology",
            "version":     "0.1.0",
            "description": "Test project",
            "author":      "",
        },
        "paths": {
            "data_dir":           "data",
            "raw_dir":            "data/raw",
            "processed_dir":      "data/processed",
            "patches_dir":        "data/patches",
            "patches_images_dir": "data/patches/images",
            "patches_masks_dir":  "data/patches/masks",
            "splits_dir":         "data/splits",
            "checkpoints_dir":    "checkpoints",
            "outputs_dir":        "outputs",
            "geotiffs_dir":       "outputs/geotiffs",
            "shapefiles_dir":     "outputs/shapefiles",
            "visualizations_dir": "outputs/visualizations",
            "logs_dir":           "logs",
            "tensorboard_dir":    "logs/tensorboard",
        },
        "gee": {
            "max_download_size_mb":    32,
            "tile_overlap_pixels":     64,
            "request_timeout_seconds": 300,
        },
        "aoi": {
            "min_lon": None, "min_lat": None,
            "max_lon": None, "max_lat": None,
            "max_width_degrees": 3.0,
        },
        "date_range": {"start": None, "end": None},
        "satellite": {
            "collections": [
                "LANDSAT/LC08/C02/T1_L2",
                "LANDSAT/LC09/C02/T1_L2"
            ],
            "scale_factor":            0.0000275,
            "offset":                  -0.2,
            "sr_min":                  0.0,
            "sr_max":                  1.0,
            "max_cloud_cover_percent": 20,
            "qa_bits": {"fill": 0, "cloud": 3, "cloud_shadow": 4},
            "resolution_meters":       30,
            "output_crs":              "EPSG:4326",
        },
        "spectral_bands": [
            {"name": "B2",      "gee_name": "SR_B2", "description": "Blue"},
            {"name": "B3",      "gee_name": "SR_B3", "description": "Green"},
            {"name": "B4",      "gee_name": "SR_B4", "description": "Red"},
            {"name": "B5",      "gee_name": "SR_B5", "description": "NIR"},
            {"name": "B6",      "gee_name": "SR_B6", "description": "SWIR1"},
            {"name": "B7",      "gee_name": "SR_B7", "description": "SWIR2"},
            {"name": "MNDWI",   "gee_name": None,    "description": "MNDWI"},
            {"name": "NDWI",    "gee_name": None,    "description": "NDWI"},
            {"name": "NDVI",    "gee_name": None,    "description": "NDVI"},
            {"name": "BSI",     "gee_name": None,    "description": "BSI"},
            {"name": "AWEInsh", "gee_name": None,    "description": "AWEInsh"},
        ],
        "num_channels": 11,
        "preprocessing": {
            "clip_percentile_low":  2.0,
            "clip_percentile_high": 98.0,
            "channel_means":        None,
            "channel_stds":         None,
        },
        "label_generation": {
            "water_threshold_mndwi": 0.2,
            "sand_threshold_bsi":    0.1,
            "sand_max_mndwi":        0.2,
        },
        "patch_generation": {
            "patch_size":            512,
            "train_stride":          256,
            "inference_stride":      256,
            "min_valid_pixel_ratio": 0.7,
            "nodata_value":          -9999.0,
        },
        "classes": {
            "num_classes": 3,
            "labels":  {"background": 0, "water": 1, "sand": 2},
            "names":   ["background", "water", "sand"],
            "colors":  {
                "background": [128, 128, 128],
                "water":      [0, 119, 190],
                "sand":       [255, 200, 87],
            },
        },
        "model": {
            "architecture":           "UnetPlusPlus",
            "encoder_name":           "efficientnet-b4",
            "encoder_weights":        "imagenet",
            "in_channels":            11,
            "num_classes":            3,
            "decoder_channels":       [256, 128, 64, 32, 16],
            "decoder_attention_type": None,
            "activation":             None,
        },
        "loss": {
            "dice_weight":  0.5,
            "focal_weight": 0.5,
            "focal_gamma":  2.0,
            "focal_alpha":  None,
            "dice_smooth":  1.0,
            "dice_mode":    "multiclass",
        },
        "optimizer": {
            "name":                "AdamW",
            "learning_rate":       0.0001,
            "weight_decay":        0.0001,
            "betas":               [0.9, 0.999],
            "eps":                 1.0e-8,
            "gradient_clip_value": 1.0,
        },
        "scheduler": {
            "name":       "CosineAnnealingWarmRestarts",
            "T_0":        20,
            "T_mult":     2,
            "eta_min":    1.0e-6,
            "last_epoch": -1,
        },
        "augmentation": {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob":   0.5,
            "random_rotate90_prob": 0.5,
            "shift_scale_rotate": {
                "enabled": True, "shift_limit": 0.05,
                "scale_limit": 0.1, "rotate_limit": 15, "prob": 0.4,
            },
            "elastic_transform": {
                "enabled": True, "alpha": 120, "sigma": 6, "prob": 0.2,
            },
            "grid_distortion":           {"enabled": True, "prob": 0.3},
            "random_brightness_contrast": {
                "enabled": True, "brightness_limit": 0.2,
                "contrast_limit": 0.2, "prob": 0.4,
            },
            "gauss_noise": {
                "enabled": True, "var_limit": [0.005, 0.02], "prob": 0.3,
            },
            "coarse_dropout": {
                "enabled": True, "max_holes": 8, "max_height": 32,
                "max_width": 32, "fill_value": 0, "prob": 0.3,
            },
        },
        "training": {
            "num_epochs":                  150,
            "batch_size":                  8,
            "num_workers":                 4,
            "pin_memory":                  True,
            "gradient_accumulation_steps": 4,
            "use_amp":                     True,
            "early_stopping": {
                "enabled": True, "patience": 20, "min_delta": 0.001,
                "monitor": "val_iou", "mode": "max",
            },
            "checkpoint": {
                "save_best_only": True, "monitor": "val_iou",
                "mode": "max", "filename": "best_model.pth", "save_last": True,
            },
            "val_every_n_epochs": 1,
            "tensorboard": {"enabled": True, "log_every_n_steps": 10},
            "train_val_test_split": [0.70, 0.15, 0.15],
        },
        "inference": {
            "patch_size":           512,
            "stride":               256,
            "gaussian_sigma":       0.5,
            "batch_size":           16,
            "confidence_threshold": 0.5,
            "tta_enabled":          False,
        },
        "postprocessing": {
            "closing": {"enabled": True, "kernel_size": 5, "iterations": 1},
            "opening": {"enabled": True, "kernel_size": 3, "iterations": 1},
            "min_component_size": {"water": 50, "sand": 100},
            "river_buffer_meters": 500,
            "dense_crf": {
                "enabled": False, "num_iterations": 5,
                "pos_w": 3, "pos_xy_std": 3,
                "bi_w": 4, "bi_xy_std": 49, "bi_rgb_std": 5,
            },
        },
        "export": {
            "geotiff": {
                "enabled": True, "compress": "LZW", "tiled": True,
                "tile_size": 256, "overviews": True, "dtype": "uint8",
            },
            "shapefile": {
                "enabled": True, "simplify_tolerance": 0.0, "min_area_m2": 900,
            },
            "area_statistics": {
                "enabled": True, "unit": "km2", "pixel_area_m2": 900,
            },
        },
        "reproducibility": {
            "seed": 42, "deterministic": True, "benchmark": False,
        },
        "device": {"device": "auto", "num_gpus": 1},
        "logging": {
            "config_file":        "config/logging.yaml",
            "level":              "INFO",
            "log_filename":       "river_morphology.log",
            "error_log_filename": "river_morphology_errors.log",
        },
    }


def write_config(directory: Path, data: dict[str, Any]) -> Path:
    """
    Serialize a configuration dictionary to config/config.yaml inside directory.

    Args:
        directory: Root directory in which to create the config/ subdirectory.
        data:      Configuration dictionary to serialize as YAML.

    Returns:
        Absolute path to the written config.yaml file.
    """
    config_dir = directory / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    return config_file


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Write a valid config.yaml to tmp_path and return its absolute path."""
    return write_config(tmp_path, make_valid_config())


@pytest.fixture
def config(valid_config_file: Path):
    """Return an initialized Config object from the valid test config."""
    from src.core.config import Config
    return Config(config_path=valid_config_file)