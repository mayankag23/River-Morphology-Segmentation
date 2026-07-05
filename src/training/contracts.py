"""
Public data contracts for the Data Transformation and Augmentation Pipeline
(Module 12).

These dataclasses are the stable public interfaces that Module 13 (Model Zoo),
Module 14 (Training Engine), and all future consumers of Module 12 depend on.
They must not be modified without strong justification.

Contract chain:
    TorchDatasetResult (Module 11)
        |
        v
    TransformPipelineResult (Module 12)    <-- public output contract
        |
        v
    Model + Training Engine (Module 13/14)

TransformSample is the per-sample unit passed between pipeline stages.
NormalizationStatistics carries per-band mean and standard deviation.
TransformMetadata records provenance of the applied transform configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "NormalizationStatistics",
    "TransformSample",
    "TransformMetadata",
    "TransformPipelineResult",
]


# ==============================================================================
# NormalizationStatistics
# ==============================================================================

@dataclass(frozen=True)
class NormalizationStatistics:
    """
    Per-band normalization statistics for a multi-spectral dataset.

    All arrays are 1-D with length equal to num_bands.  Values are stored as
    Python tuples (immutable, hashable) rather than numpy arrays so the object
    can be used as a frozen dataclass field without special __hash__ logic.

    Attributes:
        band_names:    Ordered band names.  Empty tuple when unavailable.
        mean:          Per-band mean over the training split.
        std:           Per-band standard deviation over the training split.
                       Values are guaranteed > 0 (replaced with 1.0 when
                       the computed std is zero to prevent division by zero).
        num_samples:   Number of patches used to compute these statistics.
        source:        Human-readable description of the statistics source:
                       "computed" (computed from training data),
                       "supplied" (provided externally via config or caller).
        min_values:    Per-band minimum observed values.  Empty when not
                       computed (e.g. when statistics are externally supplied).
        max_values:    Per-band maximum observed values.  Empty when not
                       computed.
    """

    band_names:  tuple[str, ...]
    mean:        tuple[float, ...]
    std:         tuple[float, ...]
    num_samples: int
    source:      str
    min_values:  tuple[float, ...] = field(default_factory=tuple)
    max_values:  tuple[float, ...] = field(default_factory=tuple)

    @property
    def num_bands(self) -> int:
        """Number of spectral bands covered by these statistics."""
        return len(self.mean)

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mean, std) as (C,) float32 numpy arrays.

        Convenience method for use inside transform implementations.
        """
        return (
            np.array(self.mean, dtype=np.float32),
            np.array(self.std, dtype=np.float32),
        )


# ==============================================================================
# TransformSample
# ==============================================================================

@dataclass
class TransformSample:
    """
    A single training sample as it flows through the transform pipeline.

    NOT frozen: transform stages mutate the image and mask arrays in place
    to avoid unnecessary copies during sequential composition.

    Attributes:
        image:             (C, H, W) float32 numpy array -- spectral bands.
        mask:               (H, W) uint8 numpy array -- class IDs (0..N-1).
        sample_id:          Unique patch identifier (from DatasetSample).
        split:              Dataset split: "train", "validation", or "test".
        acquisition_date:   YYYY-MM-DD representative date.  Empty string if
                            unavailable.
        season:             Resolved season name (e.g. "monsoon").
        hydrological_year:  Resolved hydrological year integer.  0 if unknown.
        sensor:             Sensor name(s) (e.g. "L8,L9").
        river_name:         River name, or empty string.
        reach_id:           River reach identifier, or empty string.
        basin_id:           Drainage basin identifier, or empty string.
        aoi_id:             Area-of-interest identifier, or empty string.
        patch_path:         Absolute path to the source patch GeoTIFF.
        mask_path:          Absolute path to the source mask GeoTIFF.
        metadata:           Free-form dict preserved by all transforms and
                            available to training engine logging.
    """

    image:              np.ndarray          # (C, H, W) float32
    mask:                np.ndarray          # (H, W) uint8
    sample_id:           str
    split:               str
    acquisition_date:    str                 = ""
    season:              str                 = ""
    hydrological_year:   int                 = 0
    sensor:              str                 = ""
    river_name:          str                 = ""
    reach_id:            str                 = ""
    basin_id:            str                 = ""
    aoi_id:              str                 = ""
    patch_path:          str                 = ""
    mask_path:           str                 = ""
    metadata:            dict                = field(default_factory=dict)


# ==============================================================================
# TransformMetadata
# ==============================================================================

@dataclass(frozen=True)
class TransformMetadata:
    """
    Immutable provenance record for a transform configuration.

    Every TransformPipelineResult carries one TransformMetadata instance so
    that any downstream consumer (Module 14, evaluation, inference) can
    reconstruct the exact normalization and augmentation applied to the data.

    Attributes:
        pipeline_version:      Version string from config.training.version.
        random_seed:           Seed used for all random transforms.
        normalization_source:  "computed" or "supplied".
        normalization_stats:   NormalizationStatistics applied to the data.
        train_augmentations:   Ordered names of augmentations in the train
                               pipeline.
        val_augmentations:      Ordered names of augmentations in the
                               validation pipeline (usually empty).
        test_augmentations:     Ordered names of augmentations in the test
                               pipeline (usually empty).
        created_at:             ISO 8601 UTC timestamp of pipeline creation.
        config_hash:            8-char SHA-256 prefix of training config section.
                               None when hashing fails.
    """

    pipeline_version:      str
    random_seed:           int
    normalization_source:   str
    normalization_stats:    NormalizationStatistics
    train_augmentations:    tuple[str, ...]
    val_augmentations:       tuple[str, ...]
    test_augmentations:      tuple[str, ...]
    created_at:              str
    config_hash:             str | None


# ==============================================================================
# TransformPipelineResult
# ==============================================================================

@dataclass(frozen=True)
class TransformPipelineResult:
    """
    Immutable public output contract of TransformPipeline.build().

    Module 13 (Model Zoo) and Module 14 (Training Engine) consume this object.
    Only the public API surface defined here should ever be accessed by
    downstream modules.

    Attributes:
        train_dataset:       PyTorch Dataset yielding (image_tensor, mask_tensor)
                             pairs with full training augmentations applied.
                             Type is Any to avoid importing torch at module level.
        validation_dataset:   PyTorch Dataset for validation (norm only,
                              deterministic).
        test_dataset:         PyTorch Dataset for test (norm only,
                              deterministic).
        normalization_stats:  NormalizationStatistics used for normalization
                              (source = "computed" or "supplied").
        metadata:             TransformMetadata recording full provenance.
        num_train_samples:    Count of samples in the training dataset.
        num_val_samples:      Count of samples in the validation dataset.
        num_test_samples:     Count of samples in the test dataset.
        num_bands:             Number of spectral channels (C dimension).
        num_classes:           Number of segmentation classes.
        patch_size:            Spatial size of each patch as (H, W) tuple.
                               None if samples have mixed sizes.
        is_valid:              True when TransformValidator found no errors.
        validation_issues:     Tuple of human-readable validation issue strings.
        operations_log:        Ordered tuple of operation descriptions.
    """

    train_dataset:       object          # torch.utils.data.Dataset at runtime
    validation_dataset:   object
    test_dataset:         object
    normalization_stats:  NormalizationStatistics
    metadata:             TransformMetadata
    num_train_samples:    int
    num_val_samples:      int
    num_test_samples:     int
    num_bands:             int
    num_classes:           int
    patch_size:            tuple[int, int] | None
    is_valid:              bool
    validation_issues:     tuple[str, ...]
    operations_log:        tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines (no Unicode)."""
        status = "[OK]  " if self.is_valid else "[FAIL]"
        return [
            f"  {status} train_samples:    {self.num_train_samples}",
            f"         val_samples:      {self.num_val_samples}",
            f"         test_samples:     {self.num_test_samples}",
            f"         num_bands:        {self.num_bands}",
            f"         num_classes:      {self.num_classes}",
            f"         patch_size:       {self.patch_size}",
            f"         norm_source:      {self.normalization_stats.source}",
            f"         seed:             {self.metadata.random_seed}",
            f"         valid:            {self.is_valid}",
        ]
