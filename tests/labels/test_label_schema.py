"""
Unit tests for src/labels/schema.py.

Run:
    pytest tests/labels/test_label_schema.py -v \
        --cov=src/labels/schema --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import InvalidValueError
from src.labels.schema import ClassDefinition, ClassSchema
from tests.conftest import make_valid_config, write_config


# def _config_with_classes(tmp_path: Path, classes_override: dict | None = None):
#     from src.core.config import Config
#     data = make_valid_config()
#     data["classes"] = classes_override or {
#         "num_classes": 4,
#         "labels": {"background": 0, "water": 1, "sand": 2, "vegetation": 3},
#         "names": ["background", "water", "sand", "vegetation"],
#         "colors": {
#             "background": [128, 128, 128], "water": [0, 119, 190],
#             "sand": [255, 200, 87], "vegetation": [34, 139, 34],
#         },
#     }
#     return Config(config_path=write_config(tmp_path, data))
def _config_with_classes(tmp_path: Path, classes_override: dict | None = None):
    from src.core.config import Config

    data = make_valid_config()

    data["classes"] = classes_override or {
        "num_classes": 4,
        "labels": {
            "background": 0,
            "water": 1,
            "sand": 2,
            "vegetation": 3,
        },
        "names": [
            "background",
            "water",
            "sand",
            "vegetation",
        ],
        "colors": {
            "background": [128, 128, 128],
            "water": [0, 119, 190],
            "sand": [255, 200, 87],
            "vegetation": [34, 139, 34],
        },
    }

    # Keep Config sections consistent
    num_classes = data["classes"]["num_classes"]
    data["model"]["num_classes"] = num_classes

    if "loss" in data:
        data["loss"]["num_classes"] = num_classes

    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def config(tmp_path: Path):
    return _config_with_classes(tmp_path)


class TestClassDefinition:
    def test_frozen(self) -> None:
        d = ClassDefinition(class_id=0, name="background", color=(128, 128, 128))
        with pytest.raises((AttributeError, TypeError)):
            d.class_id = 1  # type: ignore[misc]


class TestClassSchemaFromConfig:
    def test_builds_four_classes(self, config) -> None:
        schema = ClassSchema.from_config(config)
        assert schema.num_classes == 4

    def test_sorted_by_class_id(self, config) -> None:
        schema = ClassSchema.from_config(config)
        ids = [d.class_id for d in schema.classes]
        assert ids == sorted(ids)

    def test_class_names(self, config) -> None:
        schema = ClassSchema.from_config(config)
        assert "vegetation" in schema.class_names

    def test_colors_read_correctly(self, config) -> None:
        schema = ClassSchema.from_config(config)
        assert schema.get_definition(1).color == (0, 119, 190)

    def test_duplicate_class_ids_raise(self, tmp_path: Path) -> None:
        config = _config_with_classes(tmp_path, {
            "num_classes": 2, "labels": {"background": 0, "water": 0},
            "names": ["background", "water"],
            "colors": {"background": [0, 0, 0], "water": [0, 0, 1]},
        })
        with pytest.raises(InvalidValueError, match="unique"):
            ClassSchema.from_config(config)

    def test_malformed_color_raises(self, tmp_path: Path) -> None:
        # config = _config_with_classes(tmp_path, {
        #     "num_classes": 1, "labels": {"background": 0},
        #     "names": ["background"], "colors": {"background": [0, 0]},
        # })
        config = _config_with_classes(tmp_path, {
            "num_classes": 2,
            "labels": {
                "background": 0,
                "water": 1,
            },
            "names": [
                "background",
                "water",
            ],
            "colors": {
            "background": [0, 0],      # intentionally malformed
            "water": [0, 119, 190],
            },
        })     
        
        with pytest.raises(InvalidValueError, match="RGB"):
            ClassSchema.from_config(config)

    def test_missing_colors_section_defaults_gray(self, tmp_path: Path) -> None:
        # config = _config_with_classes(tmp_path, {
        #     "num_classes": 1, "labels": {"background": 0},
        #     "names": ["background"], "colors": {},
        # })
        config = _config_with_classes(tmp_path, {
            "num_classes": 2,
            "labels": {
                "background": 0,
                "water": 1,
            },
            "names": [
                "background",
                "water",
            ],
            "colors": {},          # intentionally empty
        })
        schema = ClassSchema.from_config(config)
        assert schema.get_definition(0).color == (128, 128, 128)

    def test_supports_arbitrary_future_class_names(self, tmp_path: Path) -> None:
        """Architecture must support adding new classes purely via config."""
        config = _config_with_classes(tmp_path, {
            "num_classes": 3,
            "labels": {"wetsand": 0, "drysand": 1, "shadow": 2},
            "names": ["wetsand", "drysand", "shadow"],
            "colors": {"wetsand": [1, 1, 1], "drysand": [2, 2, 2], "shadow": [3, 3, 3]},
        })
        schema = ClassSchema.from_config(config)
        assert schema.has_class_name("shadow") is True


class TestClassSchemaQueries:
    def test_is_valid_class_id(self, config) -> None:
        schema = ClassSchema.from_config(config)
        assert schema.is_valid_class_id(2) is True
        assert schema.is_valid_class_id(99) is False

    def test_has_class_name(self, config) -> None:
        schema = ClassSchema.from_config(config)
        assert schema.has_class_name("water") is True
        assert schema.has_class_name("unknown") is False

    def test_get_name_unknown_raises(self, config) -> None:
        schema = ClassSchema.from_config(config)
        with pytest.raises(InvalidValueError):
            schema.get_name(99)

    def test_get_id_by_name(self, config) -> None:
        schema = ClassSchema.from_config(config)
        assert schema.get_id_by_name("sand") == 2

    def test_get_id_by_name_unknown_raises(self, config) -> None:
        schema = ClassSchema.from_config(config)
        with pytest.raises(InvalidValueError, match="not a configured class name"):
            schema.get_id_by_name("unknown")

    def test_get_definition_unknown_raises(self, config) -> None:
        schema = ClassSchema.from_config(config)
        with pytest.raises(InvalidValueError):
            schema.get_definition(99)