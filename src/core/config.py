"""
Configuration management for the River Morphology Segmentation System.

Provides two public classes:

    ConfigNode
        A recursive dot-notation namespace that wraps nested dictionaries.
        Enables attribute-style access rather than dictionary key access.
        Example: node.model.encoder_name  instead of  node["model"]["encoder_name"]

    Config
        The master configuration manager. Loads config.yaml, validates all
        parameters with custom exceptions, resolves paths to pathlib.Path
        objects, and initializes the Python logging system from logging.yaml.
        GEE credentials are NOT stored in config.yaml; they are read from
        environment variables via the gee_project_id and gee_service_account_key
        properties at call time.

Usage:

    from src.core.config import Config

    config = Config("config/config.yaml")

    # Dot-notation access
    print(config.model.encoder_name)     # "efficientnet-b4"
    print(config.training.batch_size)    # 8
    print(config.paths.checkpoints_dir) # PosixPath('/abs/path/to/checkpoints')

    # GEE credentials from environment variables
    project_id = config.gee_project_id       # reads GEE_PROJECT_ID env var
    key_path   = config.gee_service_account_key  # reads GEE_SERVICE_ACCOUNT_KEY env var

    # Convenience properties
    if config.has_aoi:
        print(config.aoi.min_lon)
    if not config.has_normalization_stats:
        print("Run normalize.py before training.")

    # Utility methods
    config.save("config/config_backup.yaml")
    config.reload()
    raw = config.to_dict()
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers  # Required for RotatingFileHandler in dictConfig resolution
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.core.exceptions import (
    ConfigurationError,
    GEECredentialError,
    InvalidValueError,
    MissingFieldError,
    TypeMismatchError,
)

__all__ = ["Config", "ConfigNode"]


# ==============================================================================
# ConfigNode
# ==============================================================================

class ConfigNode:
    """
    Recursive dot-notation namespace for configuration values.

    Converts nested dictionaries into an object with attribute access.
    Lists of dictionaries are recursively converted so that list elements
    also support dot-notation access.

    Args:
        data: A dictionary of configuration values. Nested dicts become
              nested ConfigNode instances. Lists of dicts become lists of
              ConfigNode instances.

    Example:
        node = ConfigNode({"model": {"encoder": "efficientnet-b4", "classes": 3}})
        print(node.model.encoder)   # "efficientnet-b4"
        print(node.model.classes)   # 3
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, self._convert(value))

    @staticmethod
    def _convert(value: Any) -> Any:
        """Recursively convert dicts and lists of dicts into ConfigNodes."""
        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [ConfigNode._convert(item) for item in value]
        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate over top-level attribute names."""
        return iter(self.__dict__)

    def __contains__(self, key: str) -> bool:
        """Support 'key in node' membership tests."""
        return key in self.__dict__

    def __repr__(self) -> str:
        keys = sorted(self.__dict__.keys())
        return f"ConfigNode(keys={keys})"

    def to_dict(self) -> dict[str, Any]:
        """
        Recursively convert this ConfigNode back to a plain dictionary.

        Path objects are serialized as strings for YAML compatibility.

        Returns:
            A plain dictionary representation of this node and all children.
        """
        return {
            key: self._to_dict_value(value)
            for key, value in self.__dict__.items()
        }

    @staticmethod
    def _to_dict_value(value: Any) -> Any:
        """Recursively convert a value for to_dict serialization."""
        if isinstance(value, ConfigNode):
            return value.to_dict()
        if isinstance(value, list):
            return [ConfigNode._to_dict_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value


# ==============================================================================
# Config
# ==============================================================================

class Config:
    """
    Master configuration manager for the River Morphology Segmentation System.

    Responsibilities:
        1. Load config.yaml from disk.
        2. Validate all parameters using custom project exceptions.
        3. Resolve relative paths to absolute pathlib.Path objects.
        4. Initialize Python logging from logging.yaml.
        5. Expose GEE credentials as properties reading from environment variables.
        6. Provide dot-notation access to all configuration sections.

    This class intentionally does NOT create any directories. Call
    setup_project_directories(config) (provided in Module 2) before the first
    pipeline run to create all required data and output directories.

    Args:
        config_path:
            Path to the master config.yaml file.
        logging_config_path:
            Optional explicit path to logging.yaml. If None, derived from
            config["logging"]["config_file"] relative to the project root.
            If the derived file does not exist, falls back to basic console logging.
        project_root:
            Optional explicit project root path. If None, derived as the
            grandparent of config_path (e.g., project/config/config.yaml → project/).

    Example:
        config = Config("config/config.yaml")
        print(config.model.encoder_name)       # "efficientnet-b4"
        print(config.paths.checkpoints_dir)    # PosixPath('/path/to/checkpoints')
        print(config.gee_project_id)           # reads GEE_PROJECT_ID env var

    Raises:
        FileNotFoundError:       If config.yaml does not exist.
        MissingFieldError:       If a required configuration section is absent.
        InvalidValueError:       If a parameter value is semantically invalid.
        TypeMismatchError:       If a parameter has an unexpected Python type.
        ConfigurationError:      For other configuration-level failures.
    """

    # All top-level sections required in config.yaml.
    REQUIRED_SECTIONS: tuple[str, ...] = (
        "project",
        "paths",
        "gee",
        "aoi",
        "date_range",
        "satellite",
        "spectral_bands",
        "num_channels",
        "preprocessing",
        "label_generation",
        "patch_generation",
        "classes",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "augmentation",
        "training",
        "inference",
        "postprocessing",
        "export",
        "reproducibility",
        "device",
        "logging",
    )

    # Valid segmentation_models_pytorch architecture class names.
    # The model module resolves the SMP class dynamically from this string.
    SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset({
        "UnetPlusPlus",
        "Unet",
        "DeepLabV3Plus",
        "FPN",
        "PSPNet",
        "Linknet",
        "MAnet",
    })

    def __init__(
        self,
        config_path: str | Path,
        logging_config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        # Resolve constructor arguments to absolute paths immediately.
        self._config_path: Path = Path(config_path).resolve()

        self._project_root: Path = (
            Path(project_root).resolve()
            if project_root is not None
            else self._config_path.parent.parent
        )

        # Initialize internal state before any method calls to avoid
        # AttributeError if __init__ raises partway through.
        self._raw_data: dict[str, Any] = {}
        self._node: ConfigNode | None = None
        self._logging_config_path: Path | None = None

        # Strict ordering: load → validate → resolve paths → setup logging.
        # Validation runs on raw strings before paths are resolved.
        self._load()
        self._validate()
        self._resolve_paths()
        self._resolve_logging_config_path(logging_config_path)
        self._setup_logging()

        self._logger = logging.getLogger(__name__)
        self._logger.info("Configuration loaded from: %s", self._config_path)
        self._logger.info("Project root: %s", self._project_root)
        self._logger.debug("Sections loaded: %s", sorted(self._raw_data.keys()))

    # --------------------------------------------------------------------------
    # Private — Loading
    # --------------------------------------------------------------------------

    def _load(self) -> None:
        """Load and parse the YAML configuration file."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}\n"
                "Ensure config/config.yaml exists in your project root."
            )

        if self._config_path.suffix not in {".yaml", ".yml"}:
            raise ConfigurationError(
                f"Configuration file must have a .yaml or .yml extension. "
                f"Got: {self._config_path}"
            )

        with open(self._config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if data is None:
            raise ConfigurationError(
                f"Configuration file is empty: {self._config_path}"
            )

        if not isinstance(data, dict):
            raise TypeMismatchError(
                field="<root>",
                expected_type="dict (YAML mapping)",
                actual_type=type(data).__name__,
            )

        self._raw_data = data
        self._node = ConfigNode(self._raw_data)

    # --------------------------------------------------------------------------
    # Private — Validation
    # --------------------------------------------------------------------------

    def _validate(self) -> None:
        """Run all validation checks in dependency order. Raises on first failure."""
        self._validate_required_sections()
        self._validate_aoi()
        self._validate_date_range()
        self._validate_spectral_bands()
        self._validate_normalization()
        self._validate_patch_size()
        self._validate_label_generation()
        self._validate_model()
        self._validate_training()
        self._validate_loss_weights()
        self._validate_scheduler()
        self._validate_split_ratios()
        self._validate_class_consistency()

    def _validate_required_sections(self) -> None:
        """Verify all required top-level sections are present."""
        missing = [s for s in self.REQUIRED_SECTIONS if s not in self._raw_data]
        if missing:
            raise MissingFieldError(
                field=", ".join(missing),
                context=(
                    "The above top-level section(s) are absent from config.yaml. "
                    "Compare your config against the template in the repository."
                ),
            )

    def _validate_aoi(self) -> None:
        """
        Validate AOI bounding box coordinates.

        All four coordinates null → valid (coordinates provided at runtime).
        All four coordinates set → validate bounds and width.
        Partial null → error (inconsistent configuration).
        """
        aoi = self._raw_data["aoi"]
        coord_keys = ("min_lon", "min_lat", "max_lon", "max_lat")
        values = {k: aoi.get(k) for k in coord_keys}
        null_keys = [k for k, v in values.items() if v is None]

        # All null: AOI will be provided at runtime — allowed.
        if len(null_keys) == 4:
            return

        # Partial null: configuration is inconsistently set.
        if null_keys:
            raise MissingFieldError(
                field=f"aoi.{{{', '.join(null_keys)}}}",
                context=(
                    "AOI is partially configured. Either set all four coordinates "
                    "(min_lon, min_lat, max_lon, max_lat) or leave all as null.\n"
                    "Partially set AOI coordinates indicate a misconfiguration."
                ),
            )

        # All set: validate ranges.
        min_lon = float(values["min_lon"])
        min_lat = float(values["min_lat"])
        max_lon = float(values["max_lon"])
        max_lat = float(values["max_lat"])

        if not (-180.0 <= min_lon < max_lon <= 180.0):
            raise InvalidValueError(
                field="aoi.[min_lon, max_lon]",
                value=(min_lon, max_lon),
                reason="must satisfy -180 ≤ min_lon < max_lon ≤ 180",
            )

        if not (-90.0 <= min_lat < max_lat <= 90.0):
            raise InvalidValueError(
                field="aoi.[min_lat, max_lat]",
                value=(min_lat, max_lat),
                reason="must satisfy -90 ≤ min_lat < max_lat ≤ 90",
            )

        max_width = float(aoi.get("max_width_degrees", 3.0))
        aoi_width = max_lon - min_lon
        if aoi_width > max_width:
            raise InvalidValueError(
                field="aoi",
                value=aoi_width,
                reason=(
                    f"AOI width ({aoi_width:.4f}°) exceeds max_width_degrees ({max_width}°). "
                    "Large AOIs may span multiple UTM zones causing spatial distortion. "
                    "Split the AOI or increase aoi.max_width_degrees if you understand "
                    "the implications."
                ),
            )

    def _validate_date_range(self) -> None:
        """
        Validate date range format and logical ordering.

        Both null → valid (dates provided at runtime).
        Both set → validate YYYY-MM-DD format and start < end.
        One null, one set → error.
        """
        dr = self._raw_data["date_range"]
        start = dr.get("start")
        end = dr.get("end")

        # Both null: dates will be provided at runtime — allowed.
        if start is None and end is None:
            return

        # One null, one set: inconsistent.
        if (start is None) != (end is None):
            null_field = "date_range.start" if start is None else "date_range.end"
            raise MissingFieldError(
                field=null_field,
                context=(
                    "Both date_range.start and date_range.end must be set together, "
                    "or both left as null."
                ),
            )

        date_fmt = "%Y-%m-%d"

        try:
            start_dt = datetime.strptime(str(start), date_fmt)
        except ValueError:
            raise InvalidValueError(
                field="date_range.start",
                value=start,
                reason="must use YYYY-MM-DD format (e.g., '2023-11-01')",
            )

        try:
            end_dt = datetime.strptime(str(end), date_fmt)
        except ValueError:
            raise InvalidValueError(
                field="date_range.end",
                value=end,
                reason="must use YYYY-MM-DD format (e.g., '2024-02-28')",
            )

        if start_dt >= end_dt:
            raise InvalidValueError(
                field="date_range",
                value=f"start={start}, end={end}",
                reason="date_range.start must be strictly before date_range.end",
            )

    def _validate_spectral_bands(self) -> None:
        """
        Validate spectral band list length, types, and canonical RGB ordering.

        The first three channels must be B2 (Blue), B3 (Green), B4 (Red) in that
        exact order so that the EfficientNet-B4 encoder's pretrained ImageNet weights
        are applied to the correct channels.
        """
        bands = self._raw_data.get("spectral_bands", [])
        num_channels = self._raw_data.get("num_channels", 0)

        if not isinstance(bands, list) or len(bands) == 0:
            raise TypeMismatchError(
                field="spectral_bands",
                expected_type="non-empty list of band descriptor dicts",
                actual_type=type(bands).__name__,
            )

        if len(bands) != num_channels:
            raise InvalidValueError(
                field="spectral_bands / num_channels",
                value=f"len(spectral_bands)={len(bands)}, num_channels={num_channels}",
                reason="len(spectral_bands) must equal num_channels",
            )

        for idx, band in enumerate(bands):
            if not isinstance(band, dict) or "name" not in band:
                raise TypeMismatchError(
                    field=f"spectral_bands[{idx}]",
                    expected_type="dict with at minimum a 'name' key",
                    actual_type=type(band).__name__,
                )

        # Enforce RGB ordering for pretrained weight compatibility.
        expected_rgb = ["B2", "B3", "B4"]
        actual_first_three = [bands[i]["name"] for i in range(min(3, len(bands)))]
        if actual_first_three != expected_rgb:
            raise InvalidValueError(
                field="spectral_bands[0:3]",
                value=actual_first_three,
                reason=(
                    f"First 3 bands must be {expected_rgb} (Blue, Green, Red) "
                    "to match ImageNet RGB input ordering for the EfficientNet-B4 encoder. "
                    "Reorder the spectral_bands list in config.yaml."
                ),
            )

        # Cross-validate with model.in_channels.
        model_in = self._raw_data.get("model", {}).get("in_channels")
        if model_in is not None and int(model_in) != num_channels:
            raise InvalidValueError(
                field="model.in_channels",
                value=model_in,
                reason=f"must equal num_channels ({num_channels})",
            )

    def _validate_normalization(self) -> None:
        """
        Validate per-channel normalization statistics.

        Both null → valid (computed later by normalize.py).
        Both set → must be lists of length num_channels; stds must be non-zero.
        One null, one set → error.
        """
        pp = self._raw_data.get("preprocessing", {})
        means = pp.get("channel_means")
        stds  = pp.get("channel_stds")

        # Both null: not yet computed — allowed.
        if means is None and stds is None:
            return

        # One null, one set: inconsistent.
        if (means is None) != (stds is None):
            null_field = (
                "preprocessing.channel_means"
                if means is None
                else "preprocessing.channel_stds"
            )
            raise MissingFieldError(
                field=null_field,
                context=(
                    "Both channel_means and channel_stds must be set together, "
                    "or both left as null. "
                    "Run src/preprocessing/normalize.py to compute both."
                ),
            )

        num_channels = self._raw_data.get("num_channels", 0)

        for field_name, values in (
            ("preprocessing.channel_means", means),
            ("preprocessing.channel_stds", stds),
        ):
            if not isinstance(values, list):
                raise TypeMismatchError(
                    field=field_name,
                    expected_type=f"list of {num_channels} floats",
                    actual_type=type(values).__name__,
                )
            if len(values) != num_channels:
                raise InvalidValueError(
                    field=field_name,
                    value=f"length={len(values)}",
                    reason=f"must contain exactly {num_channels} values (one per channel)",
                )

        # Zero standard deviation would cause division by zero during normalization.
        zero_std_indices = [
            i for i, s in enumerate(stds) if s is not None and float(s) == 0.0
        ]
        if zero_std_indices:
            raise InvalidValueError(
                field="preprocessing.channel_stds",
                value=f"zero at indices {zero_std_indices}",
                reason=(
                    "Standard deviation must be non-zero for all channels "
                    "to avoid division by zero during normalization."
                ),
            )

    def _validate_patch_size(self) -> None:
        """Validate patch size is a positive power of 2 and consistent with inference."""
        patch_cfg = self._raw_data.get("patch_generation", {})
        patch_size = patch_cfg.get("patch_size")

        if patch_size is None:
            raise MissingFieldError(
                field="patch_generation.patch_size",
                context="Required to define the spatial size of training and inference patches.",
            )

        if not isinstance(patch_size, int) or patch_size <= 0:
            raise TypeMismatchError(
                field="patch_generation.patch_size",
                expected_type="positive int (power of 2, e.g. 256, 512, 1024)",
                actual_type=type(patch_size).__name__,
            )

        if patch_size & (patch_size - 1) != 0:
            raise InvalidValueError(
                field="patch_generation.patch_size",
                value=patch_size,
                reason="must be a power of 2 (e.g., 256, 512, 1024)",
            )

        # Inference patch size must match training patch size.
        inference_patch = self._raw_data.get("inference", {}).get("patch_size")
        if inference_patch is not None and int(inference_patch) != patch_size:
            raise InvalidValueError(
                field="inference.patch_size",
                value=inference_patch,
                reason=(
                    f"must equal patch_generation.patch_size ({patch_size}). "
                    "Patch size must be identical between training and inference."
                ),
            )

    def _validate_label_generation(self) -> None:
        """Validate label generation thresholds for internal consistency."""
        lg = self._raw_data.get("label_generation", {})

        water_thresh = lg.get("water_threshold_mndwi")
        sand_thresh  = lg.get("sand_threshold_bsi")
        sand_max     = lg.get("sand_max_mndwi")

        for field_name, value in (
            ("label_generation.water_threshold_mndwi", water_thresh),
            ("label_generation.sand_threshold_bsi",    sand_thresh),
            ("label_generation.sand_max_mndwi",        sand_max),
        ):
            if value is None:
                raise MissingFieldError(
                    field=field_name,
                    context="Required for automatic label generation.",
                )
            if not isinstance(value, (int, float)):
                raise TypeMismatchError(
                    field=field_name,
                    expected_type="float",
                    actual_type=type(value).__name__,
                )

        # sand_max_mndwi should be ≤ water_threshold_mndwi to prevent class overlap.
        if float(sand_max) > float(water_thresh):
            raise InvalidValueError(
                field="label_generation.sand_max_mndwi",
                value=sand_max,
                reason=(
                    f"must be ≤ water_threshold_mndwi ({water_thresh}). "
                    "If sand_max_mndwi > water_threshold_mndwi, the same pixel "
                    "can qualify as both water and sand."
                ),
            )

    def _validate_model(self) -> None:
        """Validate model architecture name, channel count, and class count."""
        model = self._raw_data.get("model", {})
        arch  = model.get("architecture")

        if arch is not None and arch not in self.SUPPORTED_ARCHITECTURES:
            raise InvalidValueError(
                field="model.architecture",
                value=arch,
                reason=(
                    f"not in supported architectures: "
                    f"{sorted(self.SUPPORTED_ARCHITECTURES)}. "
                    "Update model.architecture in config.yaml."
                ),
            )

        in_channels = model.get("in_channels")
        if in_channels is not None:
            if not isinstance(in_channels, int) or in_channels < 1:
                raise InvalidValueError(
                    field="model.in_channels",
                    value=in_channels,
                    reason="must be a positive integer",
                )

        num_classes = model.get("num_classes")
        if num_classes is not None:
            if not isinstance(num_classes, int) or num_classes < 2:
                raise InvalidValueError(
                    field="model.num_classes",
                    value=num_classes,
                    reason="must be an integer ≥ 2 (minimum: background + one target class)",
                )

    def _validate_training(self) -> None:
        """Validate training hyperparameter types and ranges."""
        training  = self._raw_data.get("training", {})
        optimizer = self._raw_data.get("optimizer", {})

        lr = optimizer.get("learning_rate")
        if lr is not None:
            lr = float(lr)
            if lr <= 0:
                raise InvalidValueError(
                    field="optimizer.learning_rate",
                    value=lr,
                    reason="must be a positive number",
                )
            if lr > 0.1:
                raise InvalidValueError(
                    field="optimizer.learning_rate",
                    value=lr,
                    reason=(
                        "is unusually high for a pretrained encoder. "
                        "Typical range: 1e-5 to 1e-3. "
                        "High learning rates destroy pretrained ImageNet features."
                    ),
                )

        for int_field, key, container in (
            ("training.batch_size",  "batch_size",  training),
            ("training.num_epochs",  "num_epochs",  training),
            ("training.num_workers", "num_workers", training),
        ):
            value = container.get(key)
            if value is not None:
                if not isinstance(value, int):
                    raise TypeMismatchError(
                        field=int_field,
                        expected_type="int",
                        actual_type=type(value).__name__,
                    )
                min_value = 0 if key == "num_workers" else 1
                if value < min_value:
                    raise InvalidValueError(
                        field=int_field,
                        value=value,
                        reason=f"must be >= {min_value}",
                    )

    def _validate_loss_weights(self) -> None:
        """Validate that Dice and Focal loss weights sum to 1.0."""
        loss    = self._raw_data.get("loss", {})
        dice_w  = float(loss.get("dice_weight",  0.5))
        focal_w = float(loss.get("focal_weight", 0.5))
        total   = dice_w + focal_w

        if abs(total - 1.0) > 1e-6:
            raise InvalidValueError(
                field="loss.[dice_weight, focal_weight]",
                value=f"{dice_w} + {focal_w} = {total:.8f}",
                reason="dice_weight + focal_weight must sum to exactly 1.0",
            )

        focal_gamma = loss.get("focal_gamma")
        if focal_gamma is not None:
            if not isinstance(focal_gamma, (int, float)) or float(focal_gamma) < 0:
                raise InvalidValueError(
                    field="loss.focal_gamma",
                    value=focal_gamma,
                    reason="must be a non-negative number",
                )

    def _validate_scheduler(self) -> None:
        """Validate learning rate scheduler parameters."""
        scheduler = self._raw_data.get("scheduler", {})
        t0 = scheduler.get("T_0")

        if t0 is not None:
            if not isinstance(t0, int) or t0 < 1:
                raise InvalidValueError(
                    field="scheduler.T_0",
                    value=t0,
                    reason="must be a positive integer (length of first restart cycle)",
                )

    def _validate_split_ratios(self) -> None:
        """Validate train/val/test split ratios sum to 1.0."""
        splits = self._raw_data.get("training", {}).get("train_val_test_split")
        if splits is None:
            return

        if not isinstance(splits, list) or len(splits) != 3:
            raise TypeMismatchError(
                field="training.train_val_test_split",
                expected_type="list of exactly 3 floats [train, val, test]",
                actual_type=str(splits),
            )

        total = sum(float(s) for s in splits)
        if abs(total - 1.0) > 1e-6:
            raise InvalidValueError(
                field="training.train_val_test_split",
                value=f"{splits} → sum={total:.8f}",
                reason="split ratios must sum to exactly 1.0",
            )

    def _validate_class_consistency(self) -> None:
        """Validate that num_classes is consistent across classes, model, and loss sections."""
        classes     = self._raw_data.get("classes", {})
        num_classes = classes.get("num_classes")

        if num_classes is None:
            return

        names = classes.get("names", [])
        if names and len(names) != num_classes:
            raise InvalidValueError(
                field="classes.names",
                value=f"length={len(names)}",
                reason=f"must equal classes.num_classes ({num_classes})",
            )

        labels = classes.get("labels", {})
        if labels and len(labels) != num_classes:
            raise InvalidValueError(
                field="classes.labels",
                value=f"length={len(labels)}",
                reason=f"must equal classes.num_classes ({num_classes})",
            )

        model_classes = self._raw_data.get("model", {}).get("num_classes")
        if model_classes is not None and int(model_classes) != num_classes:
            raise InvalidValueError(
                field="model.num_classes",
                value=model_classes,
                reason=f"must equal classes.num_classes ({num_classes})",
            )

    # --------------------------------------------------------------------------
    # Private — Path Resolution
    # --------------------------------------------------------------------------

    def _resolve_paths(self) -> None:
        """
        Convert all string values in the 'paths' section to absolute Path objects.

        Each relative path is resolved against self._project_root.
        The project root is also injected as paths.project_root for convenience.
        """
        if "paths" not in self._raw_data:
            return

        resolved: dict[str, Any] = {}
        for key, value in self._raw_data["paths"].items():
            if isinstance(value, str):
                resolved[key] = (self._project_root / value).resolve()
            else:
                resolved[key] = value

        resolved["project_root"] = self._project_root
        self._raw_data["paths"] = resolved

        # Rebuild ConfigNode so resolved Path objects are accessible via dot notation.
        self._node = ConfigNode(self._raw_data)

    # --------------------------------------------------------------------------
    # Private — Logging Setup
    # --------------------------------------------------------------------------

    def _resolve_logging_config_path(
        self, explicit_path: str | Path | None
    ) -> None:
        """Determine the absolute path to logging.yaml."""
        if explicit_path is not None:
            self._logging_config_path = Path(explicit_path).resolve()
            return

        log_cfg_file = self._raw_data.get("logging", {}).get("config_file")
        if log_cfg_file:
            self._logging_config_path = (
                self._project_root / log_cfg_file
            ).resolve()
        else:
            self._logging_config_path = None

    def _setup_logging(self) -> None:
        """
        Initialize Python logging from logging.yaml.

        Behavior:
            - logging.yaml found AND log directory exists:
                Full logging — console + rotating file + rotating error file.
            - logging.yaml found AND log directory does NOT exist:
                Console-only logging. File handlers are stripped from the config.
                A warning is emitted after configuration completes.
            - logging.yaml not found:
                Minimal fallback — basicConfig with console output only.

        This method intentionally does NOT create any directories.
        Run setup_project_directories(config) (Module 2) to create them.
        """
        # Resolve log directory path without creating it.
        logs_dir_raw = self._raw_data.get("paths", {}).get("logs_dir", "logs")
        logs_dir: Path = (
            logs_dir_raw
            if isinstance(logs_dir_raw, Path)
            else (self._project_root / str(logs_dir_raw)).resolve()
        )
        logs_dir_exists = logs_dir.exists()

        log_filename       = self._raw_data.get("logging", {}).get(
            "log_filename", "river_morphology.log"
        )
        error_log_filename = self._raw_data.get("logging", {}).get(
            "error_log_filename", "river_morphology_errors.log"
        )

        if (
            self._logging_config_path is not None
            and self._logging_config_path.exists()
        ):
            with open(self._logging_config_path, "r", encoding="utf-8") as fh:
                log_cfg = yaml.safe_load(fh) or {}

            handlers = log_cfg.get("handlers", {})

            if logs_dir_exists:
                # Override handler filenames with absolute paths.
                if "file" in handlers:
                    handlers["file"]["filename"] = str(logs_dir / log_filename)
                if "error_file" in handlers:
                    handlers["error_file"]["filename"] = str(
                        logs_dir / error_log_filename
                    )
                logging.config.dictConfig(log_cfg)
            else:
                # Strip file handlers; keep only StreamHandler-based handlers.
                console_handlers = {
                    name: cfg
                    for name, cfg in handlers.items()
                    if cfg.get("class") == "logging.StreamHandler"
                }
                log_cfg["handlers"] = console_handlers

                for logger_cfg in log_cfg.get("loggers", {}).values():
                    logger_cfg["handlers"] = [
                        h for h in logger_cfg.get("handlers", [])
                        if h in console_handlers
                    ]

                root = log_cfg.get("root", {})
                if "handlers" in root:
                    root["handlers"] = [
                        h for h in root["handlers"] if h in console_handlers
                    ]

                logging.config.dictConfig(log_cfg)
                logging.warning(
                    "Log directory does not exist: '%s'. "
                    "File logging is disabled. "
                    "Call setup_project_directories(config) to create it.",
                    logs_dir,
                )
        else:
            # Fallback: minimal console-only configuration.
            log_level_str = self._raw_data.get("logging", {}).get("level", "INFO")
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    # --------------------------------------------------------------------------
    # Public — GEE Credential Properties
    # --------------------------------------------------------------------------

    @property
    def gee_project_id(self) -> str:
        """
        Google Earth Engine Cloud project ID.

        Read from the GEE_PROJECT_ID environment variable at call time.
        Not cached — always reflects the current environment state.

        Returns:
            The GEE project ID string.

        Raises:
            GEECredentialError: If GEE_PROJECT_ID is unset or empty.
        """
        value = os.environ.get("GEE_PROJECT_ID", "").strip()
        if not value:
            raise GEECredentialError(
                "GEE_PROJECT_ID environment variable is not set or is empty.\n"
                "Add it to your .env file:\n"
                "    GEE_PROJECT_ID=your-gee-cloud-project-id\n"
                "Or export it in your shell:\n"
                "    export GEE_PROJECT_ID=your-gee-cloud-project-id\n"
                "Find your project ID at: https://console.cloud.google.com"
            )
        return value

    @property
    def gee_service_account_key(self) -> str | None:
        """
        Path to the GEE service account JSON key file.

        Read from the GEE_SERVICE_ACCOUNT_KEY environment variable at call time.
        Returns None if the variable is unset or empty, which triggers
        interactive browser-based authentication (the default for Colab).

        Returns:
            Absolute path string to the key file, or None.
        """
        value = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "").strip()
        return value if value else None

    # --------------------------------------------------------------------------
    # Public — Convenience Properties
    # --------------------------------------------------------------------------

    @property
    def has_aoi(self) -> bool:
        """True if all four AOI coordinates are configured (not null)."""
        aoi = self._raw_data.get("aoi", {})
        return all(
            aoi.get(k) is not None
            for k in ("min_lon", "min_lat", "max_lon", "max_lat")
        )

    @property
    def has_date_range(self) -> bool:
        """True if both date_range.start and date_range.end are set (not null)."""
        dr = self._raw_data.get("date_range", {})
        return dr.get("start") is not None and dr.get("end") is not None

    @property
    def has_normalization_stats(self) -> bool:
        """True if channel_means and channel_stds are computed and set (not null)."""
        pp = self._raw_data.get("preprocessing", {})
        return (
            pp.get("channel_means") is not None
            and pp.get("channel_stds") is not None
        )

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return self._project_root

    @property
    def config_path(self) -> Path:
        """Absolute path to the loaded config.yaml file."""
        return self._config_path

    # --------------------------------------------------------------------------
    # Public — Attribute Access
    # --------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the internal ConfigNode.

        Called only when normal attribute lookup fails, so private attributes
        (starting with '_') are always resolved via __dict__ before this.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            node = object.__getattribute__(self, "_node")
        except AttributeError as exc:
            raise AttributeError(
                "Config is not fully initialized. "
                "This should not happen in normal usage."
            ) from exc

        if node is not None and name in node:
            return getattr(node, name)

        available = sorted(k for k in node) if node else []
        raise AttributeError(
            f"Configuration section '{name}' not found. "
            f"Available sections: {available}"
        )

    def __repr__(self) -> str:
        sections = sorted(k for k in self._node) if self._node else []
        return (
            f"Config(\n"
            f"  path={self._config_path},\n"
            f"  project_root={self._project_root},\n"
            f"  sections={sections}\n"
            f")"
        )

    # --------------------------------------------------------------------------
    # Public — Utility Methods
    # --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return the entire configuration as a plain dictionary.

        All pathlib.Path objects are converted to strings. Suitable for
        serialization, logging, or frameworks that expect plain dicts.

        Returns:
            A deep plain-dict copy of all configuration values.
        """
        if self._node is None:
            return {}
        return self._node.to_dict()

    def save(self, path: str | Path | None = None) -> None:
        """
        Serialize the current configuration to a YAML file.

        Args:
            path: Destination file path. If None, overwrites the original
                  config.yaml that this Config was loaded from.

        Raises:
            ConfigurationError: If Config has not been loaded.
        """
        if self._node is None:
            raise ConfigurationError("Cannot save: Config has not been loaded.")

        save_path = Path(path).resolve() if path is not None else self._config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logging.getLogger(__name__).info("Configuration saved to: %s", save_path)

    def reload(self) -> None:
        """
        Reload configuration from the original config.yaml.

        Re-runs loading, validation, and path resolution. Does NOT
        re-initialize logging (logging is configured once at startup).

        Use when config.yaml has been modified externally and the in-memory
        config needs to reflect the changes without restarting the process.
        """
        self._load()
        self._validate()
        self._resolve_paths()
        logging.getLogger(__name__).info(
            "Configuration reloaded from: %s", self._config_path
        )