"""
Per-band dataset statistics computation for Module 12 normalization.

DatasetStatisticsAccumulator performs a two-pass Welford online algorithm to
compute numerically stable per-band mean and variance from a Dataset without
loading all samples into memory simultaneously.

This is invoked by TransformPipeline when normalization_source = "computed".
It operates only on the training split; validation and test splits are
normalized with the training statistics to avoid data leakage.

Usage
-----
    accumulator = DatasetStatisticsAccumulator(num_bands=12)
    for image, _ in dataloader:          # image: (C, H, W) numpy or tensor
        accumulator.update(image)
    stats = accumulator.finalize(band_names=band_names, num_samples=n)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.training.contracts import NormalizationStatistics

__all__ = ["DatasetStatisticsAccumulator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_MIN_STD_REPLACEMENT: float = 1.0   # Replace zero std with 1.0 to prevent NaN


class DatasetStatisticsAccumulator:
    """
    Online (streaming) per-band mean and std accumulator.

    Uses Welford's algorithm for numerically stable single-pass variance
    computation.  Avoids materializing the entire dataset in memory.

    Args:
        num_bands: Number of spectral bands (C dimension of image arrays).
    """

    def __init__(self, num_bands: int) -> None:
        if num_bands < 1:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="num_bands",
                value=num_bands,
                reason="must be >= 1",
            )
        self._num_bands = int(num_bands)
        self._n:    np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self._mean: np.ndarray = np.zeros(num_bands, dtype=np.float64)
        self._M2:   np.ndarray = np.zeros(num_bands, dtype=np.float64)
        self._min:  np.ndarray = np.full(num_bands,  np.inf,  dtype=np.float64)
        self._max:  np.ndarray = np.full(num_bands, -np.inf, dtype=np.float64)
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def num_bands(self) -> int:
        """Number of spectral bands tracked."""
        return self._num_bands

    @property
    def samples_seen(self) -> int:
        """Number of pixel-patches accumulated so far."""
        return int(self._n[0]) if self._num_bands > 0 else 0

    def update(self, image: Any) -> None:
        """
        Accumulate statistics from one image patch.

        Args:
            image: (C, H, W) array (numpy or torch Tensor).  If C > num_bands,
                   only the first num_bands bands are used.  If C < num_bands,
                   a WARNING is logged and existing bands are updated only.
        """
        # Convert to numpy if needed (torch tensor arrives from DataLoader).
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image, dtype=np.float64)
            except Exception as exc:
                self._logger.warning(
                    "DatasetStatisticsAccumulator.update: cannot convert "
                    "image to numpy -- skipping: %s", exc,
                )
                return

        image = image.astype(np.float64)
        c = min(image.shape[0], self._num_bands)
        if image.shape[0] < self._num_bands:
            self._logger.warning(
                "DatasetStatisticsAccumulator.update: expected %d bands but got "
                "%d -- only first %d bands updated.",
                self._num_bands, image.shape[0], c,
            )

        # Flatten spatial dimensions for each band and apply Welford update.
        for band_idx in range(c):
            values        = image[band_idx].ravel()
            valid         = values[np.isfinite(values)]
            if len(valid) == 0:
                continue
            pixel_count = len(valid)
            # Batch Welford: handle multiple pixels at once.
            old_n          = self._n[band_idx]
            new_n          = old_n + pixel_count
            new_mean       = (old_n * self._mean[band_idx] + valid.sum()) / new_n
            delta_old      = valid - self._mean[band_idx]
            delta_new      = valid - new_mean
            self._M2[band_idx] += float((delta_old * delta_new).sum())
            self._mean[band_idx] = new_mean
            self._n[band_idx]    = new_n
            self._min[band_idx]  = min(self._min[band_idx], float(valid.min()))
            self._max[band_idx]  = max(self._max[band_idx], float(valid.max()))

    def finalize(
        self,
        band_names:  tuple[str, ...] = (),
        num_samples: int = 0,
    ) -> NormalizationStatistics:
        """
        Compute final per-band mean and std from accumulated statistics.

        Zero-std bands (constant value across the entire dataset) have their
        std replaced with 1.0 to prevent division-by-zero in normalization.

        Args:
            band_names:   Optional ordered band name tuple.
            num_samples:  Number of distinct patches used.

        Returns:
            Frozen NormalizationStatistics.

        Raises:
            RuntimeError: No data has been accumulated yet.
        """
        if self._n.sum() == 0:
            raise RuntimeError(
                "DatasetStatisticsAccumulator.finalize() called before any "
                "data was accumulated via update()."
            )

        mean_out  = np.zeros(self._num_bands, dtype=np.float64)
        std_out   = np.ones(self._num_bands, dtype=np.float64)
        min_out   = np.zeros(self._num_bands, dtype=np.float64)
        max_out   = np.zeros(self._num_bands, dtype=np.float64)

        for band_idx in range(self._num_bands):
            n = self._n[band_idx]
            if n < 2:
                mean_out[band_idx] = self._mean[band_idx]
                std_out[band_idx]  = _MIN_STD_REPLACEMENT
            else:
                variance = self._M2[band_idx] / (n - 1)
                mean_out[band_idx] = self._mean[band_idx]
                std_raw = float(np.sqrt(max(variance, 0.0)))
                std_out[band_idx] = std_raw if std_raw > 1e-10 else _MIN_STD_REPLACEMENT

            min_v = self._min[band_idx]
            max_v = self._max[band_idx]
            min_out[band_idx] = min_v if np.isfinite(min_v) else 0.0
            max_out[band_idx] = max_v if np.isfinite(max_v) else 0.0

        self._logger.info(
            "DatasetStatisticsAccumulator: finalized %d bands, "
            "%d patches, mean=[%s], std=[%s]",
            self._num_bands, num_samples,
            ", ".join(f"{v:.4f}" for v in mean_out),
            ", ".join(f"{v:.4f}" for v in std_out),
        )

        return NormalizationStatistics(
            band_names  = band_names[:self._num_bands],
            mean        = tuple(float(v) for v in mean_out),
            std         = tuple(float(v) for v in std_out),
            num_samples = num_samples,
            source      = "computed",
            min_values  = tuple(float(v) for v in min_out),
            max_values  = tuple(float(v) for v in max_out),
        )

    def reset(self) -> None:
        """Reset all accumulated statistics (useful for multi-epoch reuse)."""
        self._n    = np.zeros(self._num_bands, dtype=np.int64)
        self._mean = np.zeros(self._num_bands, dtype=np.float64)
        self._M2   = np.zeros(self._num_bands, dtype=np.float64)
        self._min  = np.full(self._num_bands,  np.inf,  dtype=np.float64)
        self._max  = np.full(self._num_bands, -np.inf, dtype=np.float64)
