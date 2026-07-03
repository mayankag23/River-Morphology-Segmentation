"""
Unit tests for src/training/sampler.py.

Run:
    pytest tests/training/test_sampler.py -v \
        --cov=src/training/sampler --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.exceptions import InvalidValueError
from src.training.sampler import SamplerStrategy, TemporalSampler

pytest.importorskip("torch", reason="torch is required for sampler tests")


def _config(tmp_path: Path, strategy: str = "none"):
    from src.core.config import Config
    from tests.conftest import make_valid_config, write_config
    data = make_valid_config()
    data["training"] = {
        "sampler": {"strategy": strategy, "random_seed": 42}
    }
    return Config(config_path=write_config(tmp_path, data))


def _entry(patch_id: str, season: str = "monsoon", ratio: float = 0.9) -> MagicMock:
    e = MagicMock()
    e.sample_id               = patch_id
    e.season                  = season
    e.year                    = 2023
    e.label_valid_pixel_ratio  = ratio
    return e


class TestSamplerStrategy:
    def test_from_string_valid(self) -> None:
        assert SamplerStrategy.from_string("none") == SamplerStrategy.NONE
        assert SamplerStrategy.from_string("temporal_balanced") == SamplerStrategy.TEMPORAL_BALANCED
        assert SamplerStrategy.from_string("class_balanced") == SamplerStrategy.CLASS_BALANCED

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(InvalidValueError, match="strategy"):
            SamplerStrategy.from_string("unknown_strategy")


class TestTemporalSampler:
    def test_none_strategy_returns_none(self, tmp_path: Path) -> None:
        sampler = TemporalSampler(_config(tmp_path, strategy="none"))
        result  = sampler.build([_entry("p1")])
        assert result is None

    def test_empty_entries_returns_none(self, tmp_path: Path) -> None:
        sampler = TemporalSampler(_config(tmp_path, strategy="temporal_balanced"))
        result  = sampler.build([])
        assert result is None

    def test_temporal_balanced_returns_sampler(self, tmp_path: Path) -> None:
        from torch.utils.data import WeightedRandomSampler
        entries = [
            _entry("p1", season="monsoon"),
            _entry("p2", season="monsoon"),
            _entry("p3", season="winter"),
        ]
        sampler = TemporalSampler(_config(tmp_path, strategy="temporal_balanced"))
        result  = sampler.build(entries)
        assert isinstance(result, WeightedRandomSampler)

    def test_class_balanced_returns_sampler(self, tmp_path: Path) -> None:
        from torch.utils.data import WeightedRandomSampler
        entries = [_entry(f"p{i}", ratio=float(i + 1) / 5) for i in range(5)]
        sampler = TemporalSampler(_config(tmp_path, strategy="class_balanced"))
        result  = sampler.build(entries)
        assert isinstance(result, WeightedRandomSampler)

    def test_temporal_weights_balance_seasons(self, tmp_path: Path) -> None:
        """Equal total weight per season means common season gets lower per-sample weight."""
        entries = [
            _entry("a", season="monsoon"),
            _entry("b", season="monsoon"),
            _entry("c", season="winter"),
        ]
        sampler  = TemporalSampler(_config(tmp_path, strategy="temporal_balanced"))
        weights  = sampler._compute_weights(entries)
        # monsoon has 2 samples -> weight = 1/(2*2); winter has 1 -> weight = 1/(2*1)
        # winter per-sample weight should be higher
        assert weights[2] > weights[0]

    def test_sampler_num_samples_matches_entries(self, tmp_path: Path) -> None:
        entries = [_entry(f"p{i}") for i in range(6)]
        sampler = TemporalSampler(_config(tmp_path, strategy="temporal_balanced"))
        result  = sampler.build(entries)
        assert len(result) == 6