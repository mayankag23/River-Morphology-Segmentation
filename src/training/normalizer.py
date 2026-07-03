"""
Band normalization for the River Morphology training pipeline (Module 11).

DatasetNormalizer computes per-band statistics from the training split
and stores them as an immutable NormalizationStats object. The same stats
are then applied to train, validation, and test splits so that the model
never sees the distribution of the evaluation data during statistics
computation.

Supported strategies:
    PER_BAND_MEAN_STD: subtract mean, divide by std (z-score). Recommended.
    MIN_MAX:           scale to [0, 1] using percentile bounds. Avoids
                       outlier sensitivity at the cost of interpretability.
    NONE:              no normalization; raw float32 values returned.

Statistics are computed by reading mask-valid pixels from each training
patch GeoTIFF via rasterio. NoData pixels (NaN or sentinel value) are
excluded from statistics so that masked pixels do not bias the mean/std.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import InvalidValueError

__all__ = [
    "NormalizationStrategy",
    "NormalizationStats",
    "DatasetNormalizer",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_STATS_FILENAME: str    = "normalization_stats.json"
_EPS: float             = 1e-8


class NormalizationStrategy(str, Enum):
    """Supported normalization strategies."""

    PER_BAND_MEAN_STD = "per_band_mean_std"
    MIN_MAX           = "min_max"
    NONE              = "none"

    @classmethod
    def from_string(cls, value: str) -> NormalizationStrategy:
        """Convert a config string to NormalizationStrategy."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            raise InvalidValueError(
                field="training.normalization.strategy",
                value=value,
                reason=f"must be one of {[s.value for s in cls]}",
            )


@dataclass(frozen=True)
class NormalizationStats:
    """
    Immutable per-band normalization statistics.

    Attributes:
        strategy:      The strategy used to compute these statistics.
        num_bands:      Number of spectral bands.
        band_names:     Ordered tuple of band names.
        mean:           Per-band mean (or min for MIN_MAX). Shape (num_bands,).
        std:            Per-band std  (or range for MIN_MAX). Shape (num_bands,).
        percentile_min: Percentile used for min in MIN_MAX strategy.
        percentile_max: Percentile used for max in MIN_MAX strategy.
    """

    strategy:       str
    num_bands:      int
    band_names:     tuple[str, ...]
    mean:           tuple[float, ...]
    std:            tuple[float, ...]
    percentile_min: int
    percentile_max: int

    def normalize(self, data: Any) -> Any:
        """
        Normalize a float32 numpy array of shape (C, H, W).

        Args:
            data: float32 ndarray (C, H, W). May contain NaN for nodata.

        Returns:
            Normalized float32 ndarray (C, H, W) in the same shape.

        Raises:
            InvalidValueError: data.shape[0] != self.num_bands.
        """
        if data.shape[0] != self.num_bands:
            raise InvalidValueError(
                field="data.shape[0]",
                value=data.shape[0],
                reason=f"must equal num_bands={self.num_bands}",
            )

        if self.strategy == NormalizationStrategy.NONE:
            return data.copy()

        mean_arr = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr  = np.array(self.std,  dtype=np.float32).reshape(-1, 1, 1)
        std_arr  = np.where(std_arr < _EPS, _EPS, std_arr)

        return ((data - mean_arr) / std_arr).astype(np.float32)

    def denormalize(self, data: Any) -> Any:
        """
        Reverse normalization of a float32 numpy array (C, H, W).

        Args:
            data: Normalized float32 ndarray (C, H, W).

        Returns:
            Denormalized float32 ndarray in original scale.
        """
        mean_arr = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr  = np.array(self.std,  dtype=np.float32).reshape(-1, 1, 1)
        std_arr  = np.where(std_arr < _EPS, _EPS, std_arr)
        return (data * std_arr + mean_arr).astype(np.float32)

    def to_dict(self) -> dict:
        """Return a JSON-serializable plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NormalizationStats:
        """Reconstruct from a plain dict (e.g. loaded from JSON)."""
        return cls(
            strategy=data["strategy"],
            num_bands=int(data["num_bands"]),
            band_names=tuple(data["band_names"]),
            mean=tuple(float(v) for v in data["mean"]),
            std=tuple(float(v) for v in data["std"]),
            percentile_min=int(data["percentile_min"]),
            percentile_max=int(data["percentile_max"]),
        )

    def save(self, path: Path) -> Path:
        """Save NormalizationStats to a JSON file."""
        path = Path(path).resolve()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=True)
        return path

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        """Load NormalizationStats from a JSON file."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)


class DatasetNormalizer:
    """
    Computes NormalizationStats from training samples and applies them.

    Statistics are computed by iterating over all training patch GeoTIFFs
    and accumulating per-band sums, sum-of-squares, min, and max using
    Welford's online algorithm for numerical stability.

    Args:
        config: Fully initialized Config object. Reads from
                config.training.normalization.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        train_cfg = getattr(config, "training", None)
        norm_cfg  = getattr(train_cfg, "normalization", None)

        self._strategy      = NormalizationStrategy.from_string(
            str(getattr(norm_cfg, "strategy", "per_band_mean_std"))
        )
        self._percentile_min = int(getattr(norm_cfg, "percentile_min", 2))
        self._percentile_max = int(getattr(norm_cfg, "percentile_max", 98))
        self._nodata_value   = float(
            getattr(getattr(config, "labels", None), "nodata_value", 255)
        )

    def compute(
        self,
        train_entries: list[Any],     # list[DatasetManifestEntry]
        band_names:    tuple[str, ...],
    ) -> NormalizationStats:
        """
        Compute NormalizationStats from training split entries.

        Reads every training patch GeoTIFF. NaN and nodata pixels are
        excluded so they do not bias the computed statistics.

        Args:
            train_entries: DatasetManifestEntry records for the training split.
            band_names:    Ordered band names (from DatasetManifestEntry.num_bands
                           and the manifest header).

        Returns:
            Immutable NormalizationStats.
        """
        if self._strategy == NormalizationStrategy.NONE:
            n_bands = len(band_names)
            return NormalizationStats(
                strategy=self._strategy.value,
                num_bands=n_bands,
                band_names=band_names,
                mean=tuple(0.0 for _ in range(n_bands)),
                std=tuple(1.0 for _ in range(n_bands)),
                percentile_min=self._percentile_min,
                percentile_max=self._percentile_max,
            )

        self._logger.info(
            "Computing normalization stats from %d training samples. "
            "strategy=%s",
            len(train_entries), self._strategy.value,
        )

        if self._strategy == NormalizationStrategy.PER_BAND_MEAN_STD:
            return self._compute_mean_std(train_entries, band_names)
        return self._compute_min_max(train_entries, band_names)

    def save_to_dir(
        self,
        stats:      NormalizationStats,
        output_dir: Path,
    ) -> Path:
        """Save stats to output_dir/normalization_stats.json."""
        path = Path(output_dir).resolve() / _STATS_FILENAME
        return stats.save(path)

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _compute_mean_std(
        self,
        entries:    list[Any],
        band_names: tuple[str, ...],
    ) -> NormalizationStats:
        """
        Compute per-band mean and std using a two-pass accumulation.

        First pass: accumulate sum and count per band.
        Second pass: accumulate sum of squared deviations.
        NaN and nodata sentinel pixels are excluded from both passes.
        """
        n_bands = len(band_names)
        sums    = np.zeros(n_bands, dtype=np.float64)
        counts  = np.zeros(n_bands, dtype=np.int64)

        for entry in entries:
            data = self._read_patch(Path(entry.patch_path))
            if data is None:
                continue
            for b in range(n_bands):
                band = data[b].astype(np.float64)
                mask = ~np.isnan(band) & (band != self._nodata_value)
                sums[b]   += band[mask].sum()
                counts[b] += int(mask.sum())

        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

        sum_sq = np.zeros(n_bands, dtype=np.float64)
        for entry in entries:
            data = self._read_patch(Path(entry.patch_path))
            if data is None:
                continue
            for b in range(n_bands):
                band = data[b].astype(np.float64)
                mask = ~np.isnan(band) & (band != self._nodata_value)
                sum_sq[b] += np.sum((band[mask] - means[b]) ** 2)

        variances = np.where(counts > 1, sum_sq / np.maximum(counts - 1, 1), 1.0)
        stds      = np.sqrt(np.maximum(variances, _EPS ** 2))

        self._logger.info(
            "Normalization stats (mean/std): computed from %d samples.",
            len(entries),
        )
        return NormalizationStats(
            strategy=self._strategy.value,
            num_bands=n_bands,
            band_names=band_names,
            mean=tuple(float(v) for v in means),
            std=tuple(float(v) for v in stds),
            percentile_min=self._percentile_min,
            percentile_max=self._percentile_max,
        )

    def _compute_min_max(
        self,
        entries:    list[Any],
        band_names: tuple[str, ...],
    ) -> NormalizationStats:
        """
        Compute per-band min and max using configured percentiles.

        Uses np.nanpercentile over a sample of pixels to avoid loading
        all data into memory at once. Pixels are accumulated per band and
        percentiles are computed at the end.
        """
        n_bands     = len(band_names)
        band_pixels: list[list[float]] = [[] for _ in range(n_bands)]

        for entry in entries:
            data = self._read_patch(Path(entry.patch_path))
            if data is None:
                continue
            for b in range(n_bands):
                band = data[b].astype(np.float64)
                valid = band[
                    ~np.isnan(band) & (band != self._nodata_value)
                ]
                # Sample at most 10 000 pixels per band per patch to
                # keep memory bounded while getting stable percentiles.
                if len(valid) > 10_000:
                    idx   = np.random.choice(len(valid), 10_000, replace=False)
                    valid = valid[idx]
                band_pixels[b].extend(valid.tolist())

        mins: list[float] = []
        rngs: list[float] = []
        for b in range(n_bands):
            pixels = np.array(band_pixels[b]) if band_pixels[b] else np.zeros(1)
            lo     = float(np.percentile(pixels, self._percentile_min))
            hi     = float(np.percentile(pixels, self._percentile_max))
            mins.append(lo)
            rngs.append(max(hi - lo, float(_EPS)))

        return NormalizationStats(
            strategy=self._strategy.value,
            num_bands=n_bands,
            band_names=band_names,
            mean=tuple(mins),
            std=tuple(rngs),
            percentile_min=self._percentile_min,
            percentile_max=self._percentile_max,
        )

    def _read_patch(self, path: Path) -> np.ndarray | None:
        """Read patch GeoTIFF as float32 (C, H, W). Returns None on failure."""
        try:
            import rasterio
            with rasterio.open(path) as ds:
                return ds.read().astype(np.float32)
        except Exception as exc:
            self._logger.debug("Could not read %s: %s", path, exc)
            return None