"""
Tests for src/training/normalization.py

Run:
    pytest tests/training/test_training_normalization.py -v \
        --cov=src/training/normalization --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.contracts import NormalizationStatistics, TransformSample
from src.training.normalization import NormalizationTransform


def _stats(n: int = 3) -> NormalizationStatistics:
    return NormalizationStatistics(
        band_names = tuple(f"B{i}" for i in range(n)),
        mean       = tuple(0.5 for _ in range(n)),
        std        = tuple(0.1 for _ in range(n)),
        num_samples = 100,
        source      = "computed",
    )


def _sample(c: int = 3, h: int = 8, w: int = 8) -> TransformSample:
    return TransformSample(
        image     = np.ones((c, h, w), dtype=np.float32) * 0.5,
        mask      = np.zeros((h, w), dtype=np.uint8),
        sample_id  = "p1",
        split      = "train",
    )


class TestNormalizationTransform:
    def test_name(self) -> None:
        assert NormalizationTransform(_stats()).name == "normalization"

    def test_invalid_stats_type_raises(self) -> None:
        with pytest.raises(Exception):
            NormalizationTransform("not_stats")  # type: ignore[arg-type]

    def test_zero_mean_image_produces_zero_output(self) -> None:
        """image == mean -> output should be 0."""
        t = NormalizationTransform(_stats(3))  # mean=0.5, std=0.1
        s = _sample(c=3)  # image == 0.5 everywhere
        t.apply(s)
        np.testing.assert_allclose(s.image, 0.0, atol=1e-5)

    def test_image_dtype_is_float32(self) -> None:
        t = NormalizationTransform(_stats(3))
        s = _sample(c=3)
        t.apply(s)
        assert s.image.dtype == np.float32

    def test_mask_unchanged(self) -> None:
        t  = NormalizationTransform(_stats(3))
        s  = _sample(c=3)
        s.mask[:, :] = 2
        t.apply(s)
        assert (s.mask == 2).all()

    def test_known_normalization(self) -> None:
        """Test exact numerical output for a known input."""
        stats = NormalizationStatistics(
            band_names=("B0",), mean=(1.0,), std=(2.0,),
            num_samples=10, source="computed",
        )
        t = NormalizationTransform(stats)
        s = TransformSample(
            image=np.array([[[3.0]]], dtype=np.float32),  # (1,1,1)
            mask=np.array([[0]], dtype=np.uint8),
            sample_id="p", split="train",
        )
        t.apply(s)
        # (3.0 - 1.0) / 2.0 = 1.0
        np.testing.assert_allclose(s.image[0, 0, 0], 1.0, atol=1e-6)

    def test_denormalize_inverts_normalization(self) -> None:
        t  = NormalizationTransform(_stats(3))
        s  = _sample(c=3)
        original = s.image.copy()
        t.apply(s)
        recovered = t.denormalize(s.image)
        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_band_count_mismatch_warning(self, caplog) -> None:
        """If image has more bands than stats, normalization still proceeds."""
        import logging
        stats = NormalizationStatistics(
            band_names=("B0", "B1"), mean=(0.5, 0.5), std=(0.1, 0.1),
            num_samples=10, source="computed",
        )
        t = NormalizationTransform(stats)
        s = TransformSample(
            image=np.ones((4, 4, 4), dtype=np.float32) * 0.5,
            mask=np.zeros((4, 4), dtype=np.uint8),
            sample_id="p", split="train",
        )
        with caplog.at_level(logging.WARNING):
            t.apply(s)
        assert "WARNING" in caplog.text or len(caplog.records) > 0

    def test_stats_property(self) -> None:
        stats = _stats(4)
        t = NormalizationTransform(stats)
        assert t.stats is stats
