"""
Unit tests for src/training/weights.py.

Run:
    pytest tests/training/test_weights.py -v \
        --cov=src/training/weights --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.exceptions import InvalidValueError
from src.labels.schema import ClassDefinition, ClassSchema
from src.training.weights import ClassWeightStrategy, ClassWeights

pytest.importorskip("torch", reason="torch is required for weight tests")


def _schema() -> ClassSchema:
    return ClassSchema(classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water",      (0, 119, 190)),
        ClassDefinition(2, "sand",       (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    ))


def _train_stats(
    pixel_counts: tuple[int, int, int, int] = (1000, 2000, 500, 500)
) -> MagicMock:
    from src.dataset.statistics import ClassStatistics
    stats = MagicMock()
    definitions = [
        ("background", 0, pixel_counts[0]),
        ("water",      1, pixel_counts[1]),
        ("sand",       2, pixel_counts[2]),
        ("vegetation", 3, pixel_counts[3]),
    ]
    stats.class_statistics = tuple(
        ClassStatistics(
            class_id=cid, class_name=name, pixel_count=count, sample_count=1, percentage=0.0
        )
        for name, cid, count in definitions
    )
    return stats


def _config(tmp_path: Path, strategy: str = "inverse_frequency", manual: list | None = None):
    from src.core.config import Config
    from tests.conftest import make_valid_config, write_config
    data = make_valid_config()
    data["training"] = {
        "class_weights": {
            "strategy": strategy,
            "manual_weights": manual if manual is not None else [],
        }
    }
    return Config(config_path=write_config(tmp_path, data))


class TestClassWeightStrategy:
    def test_from_string_valid(self) -> None:
        assert ClassWeightStrategy.from_string("none") == ClassWeightStrategy.NONE
        assert ClassWeightStrategy.from_string("inverse_frequency") == ClassWeightStrategy.INVERSE_FREQUENCY
        assert ClassWeightStrategy.from_string("manual") == ClassWeightStrategy.MANUAL

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="strategy"):
            ClassWeightStrategy.from_string("unknown")


class TestClassWeights:
    def test_frozen(self, tmp_path: Path) -> None:
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="none"), _train_stats(), _schema()
        )
        with pytest.raises((AttributeError, TypeError)):
            cw.num_classes = 99  # type: ignore[misc]

    def test_none_strategy_returns_uniform_weights(self, tmp_path: Path) -> None:
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="none"), _train_stats(), _schema()
        )
        assert all(w == pytest.approx(1.0) for w in cw.weights)
        assert cw.num_classes == 4

    def test_inverse_frequency_returns_four_weights(self, tmp_path: Path) -> None:
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="inverse_frequency"), _train_stats(), _schema()
        )
        assert len(cw.weights) == 4
        assert cw.strategy == "inverse_frequency"

    def test_inverse_frequency_rare_class_higher_weight(self, tmp_path: Path) -> None:
        """Sand (count=500) should get higher weight than water (count=2000)."""
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="inverse_frequency"),
            _train_stats(pixel_counts=(1000, 2000, 500, 500)),
            _schema(),
        )
        water_w = cw.weights[1]   # class_id=1 water
        sand_w  = cw.weights[2]   # class_id=2 sand
        assert sand_w > water_w

    def test_manual_strategy_correct_weights(self, tmp_path: Path) -> None:
        manual = [1.0, 2.0, 3.0, 4.0]
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="manual", manual=manual), _train_stats(), _schema()
        )
        assert list(cw.weights) == pytest.approx(manual)

    def test_manual_wrong_count_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidValueError, match="manual_weights"):
            ClassWeights.from_config_and_statistics(
                _config(tmp_path, strategy="manual", manual=[1.0, 2.0]),
                _train_stats(),
                _schema(),
            )

    def test_as_tensor_shape(self, tmp_path: Path) -> None:
        import torch
        cw     = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="none"), _train_stats(), _schema()
        )
        tensor = cw.as_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4,)
        assert tensor.dtype == torch.float32

    def test_zero_pixel_counts_falls_back_to_uniform(self, tmp_path: Path) -> None:
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="inverse_frequency"),
            _train_stats(pixel_counts=(0, 0, 0, 0)),
            _schema(),
        )
        assert all(w == pytest.approx(1.0) for w in cw.weights)

    def test_class_names_match_schema(self, tmp_path: Path) -> None:
        cw = ClassWeights.from_config_and_statistics(
            _config(tmp_path, strategy="none"), _train_stats(), _schema()
        )
        assert "water" in cw.class_names
        assert "vegetation" in cw.class_names