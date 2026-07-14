"""
Data Transformation and Augmentation Pipeline orchestrator (Module 12).

TransformPipeline is the single entry point for Module 12.

    Input:   TorchDatasetResult (from Module 11)
    Output:  TransformPipelineResult (public contract consumed by Modules 13-14)

Responsibilities
-----------------
1. Build the normalization statistics (compute from training data or load
   from config / external supply).
2. Build the training / validation / test transform pipelines using
   TransformRegistry.create_pipeline().
3. Wrap the Module 11 PyTorch datasets with the appropriate transforms.
4. Validate the assembled result with TransformValidator.
5. Record provenance in TransformMetadata.
6. Return an immutable TransformPipelineResult.

TransformPipeline does not define or implement any transform logic.  It is a
pure orchestrator following the Single Responsibility Principle.

Architecture note
------------------
The pipeline wraps Module 11's datasets in AugmentedDataset, which intercepts
__getitem__() calls and applies the configured ComposedTransform to each
sample before it is returned to the DataLoader.

No GPU operations occur here.  All transforms are CPU-side numpy operations.
Conversion to torch Tensors happens inside AugmentedDataset.__getitem__() to
be compatible with multi-worker DataLoader processes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

from src.training.contracts import (
    NormalizationStatistics,
    TransformMetadata,
    TransformPipelineResult,
    TransformSample,
)
from src.training.registry import TransformRegistry
from src.training.statistics import DatasetStatisticsAccumulator
from src.training.transform import ComposedTransform, IdentityTransform
from src.training.validator import TransformValidator

__all__ = ["TransformPipeline", "AugmentedDataset"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# AugmentedDataset
# ==============================================================================

class AugmentedDataset:
    """
    Wraps a Module 11 PyTorch Dataset and applies a ComposedTransform.

    Implements the PyTorch Dataset interface (__len__, __getitem__) so it can
    be passed directly to a DataLoader.  The transform is applied lazily
    per sample, compatible with multiple DataLoader workers.

    Args:
        base_dataset:    Module 11 RiverMorphologyDataset (or any object
                         implementing __len__ and __getitem__).
        transform:       ComposedTransform to apply per sample.
        split:           "train", "validation", or "test".
    """

    def __init__(
        self,
        base_dataset: Any,
        transform:     ComposedTransform,
        split:         str,
    ) -> None:
        self._dataset   = base_dataset
        self._transform = transform
        self._split     = split
        self._logger: logging.Logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Return (image_tensor, mask_tensor) for one sample.

        Pipeline:
            1. Get raw item from base_dataset -> (image_tensor, mask_tensor, *meta).
            2. Convert to numpy TransformSample.
            3. Apply ComposedTransform.
            4. Convert back to torch float32 tensor (image) and long tensor (mask).
        """
        try:
            item = self._dataset[index]
        except Exception as exc:
            from src.core.exceptions import InvalidValueError
            raise InvalidValueError(
                field="index",
                value=index,
                reason=f"base_dataset[{index}] raised: {exc}",
            ) from exc

        # # Unpack; Module 11 may return (image, mask) or (image, mask, meta_dict).
        # image_tensor = item[0]
        # mask_tensor  = item[1]
        # meta         = item[2] if len(item) > 2 else {}

        # # Convert to numpy.
        # image_np = _to_numpy_float32(image_tensor)
        # mask_np  = _to_numpy_uint8(mask_tensor)

        # sample = TransformSample(
        #     image            = image_np,
        #     mask              = mask_np,
        #     sample_id         = str(meta.get("sample_id", str(index))),
        #     split             = self._split,
        #     acquisition_date  = str(meta.get("acquisition_date", "")),
        #     season            = str(meta.get("season", "")),
        #     hydrological_year = int(meta.get("hydrological_year", 0)),
        #     sensor            = str(meta.get("sensor", "")),
        #     river_name        = str(meta.get("river_name", "")),
        #     reach_id          = str(meta.get("reach_id", "")),
        #     basin_id          = str(meta.get("basin_id", "")),
        #     aoi_id            = str(meta.get("aoi_id", "")),
        #     patch_path        = str(meta.get("patch_path", "")),
        #     mask_path         = str(meta.get("mask_path", "")),
        #     metadata          = dict(meta),
        # )


        # Unpack; Module 11 returns (image, mask, SampleMetadata)
        image_tensor = item[0]
        mask_tensor  = item[1]
        meta         = item[2] if len(item) > 2 else None

        # Convert to numpy.
        image_np = _to_numpy_float32(image_tensor)
        mask_np  = _to_numpy_uint8(mask_tensor)

        # sample = TransformSample(
        #     image=image_np,
        #     mask=mask_np,
        #     sample_id=meta.sample_id if meta else str(index),
        #     split=meta.split if meta else self._split,
        #     acquisition_date=meta.acquisition_date if meta else "",
        #     season=meta.season if meta else "",
        #     hydrological_year=meta.hydrological_year if meta else 0,
        #     sensor=meta.sensor if meta else "",
        #     river_name=meta.river_name if meta else "",
        #     reach_id=meta.reach_id if meta else "",
        #     basin_id=meta.basin_id if meta else "",
        #     aoi_id=meta.aoi_id if meta else "",
        #     patch_path=meta.patch_path if meta else "",
        #     mask_path=meta.mask_path if meta else "",
        #     metadata=asdict(meta) if meta else {},
        # )
        # Normalize metadata to one dictionary representation.
        if meta is None:
            meta_dict: dict[str, Any] = {}
        elif isinstance(meta, dict):
            meta_dict = dict(meta)
        elif is_dataclass(meta):
            meta_dict = asdict(meta)
        else:
            # # Support compatible metadata objects without requiring one concrete class.
            # meta_dict = {
            #     field: getattr(meta, field)
            #     for field in (
            #         "sample_id",
            #         "split",
            #         "acquisition_date",
            #         "season",
            #         "hydrological_year",
            #         "sensor",
            #         "river_name",
            #         "reach_id",
            #         "basin_id",
            #         "aoi_id",
            #         "patch_path",
            #         "mask_path",
            #     )
            #     if hasattr(meta, field)
            # }
            raise TypeError(
                "AugmentedDataset expected metadata to be a dictionary, "
                f"dataclass, or None, but received {type(meta).__name__}."
            )
 
        sample = TransformSample(
            image=image_np,
            mask=mask_np,
            sample_id=str(meta_dict.get("sample_id", index)),
            split=str(meta_dict.get("split", self._split)),
            acquisition_date=str(meta_dict.get("acquisition_date", "")),
            season=str(meta_dict.get("season", "")),
            hydrological_year=int(meta_dict.get("hydrological_year", 0)),
            sensor=str(meta_dict.get("sensor", "")),
            river_name=str(meta_dict.get("river_name", "")),
            reach_id=str(meta_dict.get("reach_id", "")),
            basin_id=str(meta_dict.get("basin_id", "")),
            aoi_id=str(meta_dict.get("aoi_id", "")),
            patch_path=str(meta_dict.get("patch_path", "")),
            mask_path=str(meta_dict.get("mask_path", "")),
            metadata=meta_dict,
        )

        sample = self._transform.apply(sample)

        image_out = _to_torch_float32(sample.image)
        mask_out  = _to_torch_long(sample.mask)

        return image_out, mask_out, sample.metadata

    @property
    def transform(self) -> ComposedTransform:
        """The ComposedTransform applied to each sample."""
        return self._transform

    @property
    def split(self) -> str:
        """Dataset split name."""
        return self._split

    @property
    def base_dataset(self) -> Any:
        """Underlying Module 11 dataset."""
        return self._dataset


# ==============================================================================
# TransformPipeline (orchestrator)
# ==============================================================================

class TransformPipeline:
    """
    Orchestrates normalization, augmentation, and dataset wrapping.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Any) -> None:
        self._config  = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        train_cfg = getattr(config, "training", None)

        # Reproducibility.
        self._seed = int(getattr(train_cfg, "random_seed", 42))

        # Normalization configuration.
        norm_cfg = getattr(train_cfg, "normalization", None)
        self._norm_source = str(
            getattr(norm_cfg, "source", "computed")
        ).lower()  # "computed" | "supplied"

        # Class schema.
        from src.labels.schema import ClassSchema
        self._class_schema = ClassSchema.from_config(config)
        self._valid_class_ids = {d.class_id for d in self._class_schema.classes}
        self._nodata_value = int(getattr(train_cfg, "nodata_value", 255))
        self._num_classes  = self._class_schema.num_classes

        # Pipeline versioning.
        self._pipeline_version = str(
            getattr(train_cfg, "pipeline_version", "1.0.0")
        )

        # Validation.
        self._validator = TransformValidator(
            valid_class_ids = self._valid_class_ids,
            check_metadata  = bool(getattr(train_cfg, "validate_metadata", True)),
            nodata_class_id = self._nodata_value,
        )

        self._logger.debug(
            "TransformPipeline initialized. seed=%d, norm_source=%s, "
            "classes=%d, nodata=%d",
            self._seed, self._norm_source,
            self._num_classes, self._nodata_value,
        )

    def build(
        self,
        torch_dataset_result: Any,
        external_stats:        NormalizationStatistics | None = None,
    ) -> TransformPipelineResult:
        """
        Build the complete transform pipeline.

        Args:
            torch_dataset_result:  TorchDatasetResult from Module 11.
                                   Must expose train_dataset, validation_dataset,
                                   test_dataset attributes.
            external_stats:         Optional pre-computed NormalizationStatistics.
                                   When provided, overrides the norm_source
                                   config setting and uses these stats directly.

        Returns:
            Frozen TransformPipelineResult.
        """
        self._logger.info(
            "TransformPipeline.build: seeding numpy with seed=%d", self._seed
        )
        np.random.seed(self._seed)
        random.seed(self._seed)

        operations: list[str] = []

        # Step 1: Determine normalization statistics.
        stats = self._resolve_statistics(torch_dataset_result, external_stats)
        operations.append(
            f"normalization: source={stats.source}, bands={stats.num_bands}, "
            f"samples={stats.num_samples}"
        )

        # Step 2: Detect num_bands and patch_size from the training dataset.
        num_bands, patch_size = self._detect_dimensions(
            torch_dataset_result.train_dataset
        )
        operations.append(
            f"dimensions: bands={num_bands}, patch_size={patch_size}"
        )

        # Step 3: Build transform pipelines.
        train_pipeline = TransformRegistry.create_pipeline(
            config               = self._config,
            normalization_stats  = stats,
            augmentation_only    = False,
        )
        val_pipeline = TransformRegistry.create_pipeline(
            config               = self._config,
            normalization_stats  = stats,
            augmentation_only    = True,    # normalization only, no augmentation
        )
        # Validation pipeline: norm + identity (no random augmentation).
        from src.training.normalization import NormalizationTransform
        from src.training.transform import ComposedTransform, IdentityTransform
        val_norm_pipeline  = ComposedTransform([NormalizationTransform(stats), IdentityTransform()])
        test_norm_pipeline = ComposedTransform([NormalizationTransform(stats), IdentityTransform()])

        operations.append(
            f"train_pipeline: [{', '.join(train_pipeline.transform_names)}]"
        )
        operations.append(
            f"val_pipeline: [{', '.join(val_norm_pipeline.transform_names)}]"
        )

        # Step 4: Wrap datasets with AugmentedDataset.
        train_ds = AugmentedDataset(
            torch_dataset_result.train_dataset, train_pipeline, "train"
        )
        val_ds   = AugmentedDataset(
            torch_dataset_result.validation_dataset, val_norm_pipeline, "validation"
        )
        test_ds  = AugmentedDataset(
            torch_dataset_result.test_dataset, test_norm_pipeline, "test"
        )

        n_train = len(train_ds)
        n_val   = len(val_ds)
        n_test  = len(test_ds)
        operations.append(
            f"datasets: train={n_train}, val={n_val}, test={n_test}"
        )

        # Step 5: Build metadata.
        metadata = self._build_metadata(
            stats, train_pipeline, val_norm_pipeline, test_norm_pipeline
        )

        # Step 6: Validate result structure.
        result = TransformPipelineResult(
            train_dataset       = train_ds,
            validation_dataset   = val_ds,
            test_dataset         = test_ds,
            normalization_stats  = stats,
            metadata             = metadata,
            num_train_samples    = n_train,
            num_val_samples      = n_val,
            num_test_samples     = n_test,
            num_bands             = num_bands,
            num_classes           = self._num_classes,
            patch_size            = patch_size,
            is_valid              = True,           # updated below
            validation_issues     = (),
            operations_log        = tuple(operations),
        )

        validation = self._validator.validate_pipeline_result(
            result, self._num_classes, num_bands
        )
        if not validation.is_valid:
            self._logger.warning(
                "TransformPipeline.build: %d validation issue(s): %s",
                len(validation.issues), validation.issues,
            )

        # Re-build with validation results (frozen dataclass -- reconstruct).
        import dataclasses
        result = dataclasses.replace(
            result,
            is_valid          = validation.is_valid,
            validation_issues = tuple(validation.issues),
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_statistics(
        self,
        torch_result:   Any,
        external_stats:  NormalizationStatistics | None,
    ) -> NormalizationStatistics:
        """
        Determine which normalization statistics to use.

        Priority:
            1. external_stats (caller-provided, overrides everything).
            2. config-supplied statistics (norm_source = "supplied").
            3. Computed from training data (norm_source = "computed").
        """
        if external_stats is not None:
            self._logger.info(
                "Normalization: using externally supplied statistics "
                "(%d bands).", external_stats.num_bands
            )
            return external_stats

        if self._norm_source == "supplied":
            return self._load_supplied_statistics()

        # Computed from training data.
        return self._compute_statistics(torch_result.train_dataset)

    def _load_supplied_statistics(self) -> NormalizationStatistics:
        """
        Load normalization statistics from config.training.normalization.

        Config structure:
            training:
              normalization:
                source: supplied
                mean: [0.1, 0.2, ...]    # per-band, length == num_bands
                std:  [0.05, 0.08, ...]
                band_names: ["Blue", "Green", ...]
        """
        train_cfg = getattr(self._config, "training", None)
        norm_cfg  = getattr(train_cfg, "normalization", None)

        raw_mean  = list(getattr(norm_cfg, "mean", []))
        raw_std   = list(getattr(norm_cfg, "std",  []))
        raw_names = list(getattr(norm_cfg, "band_names", []))

        if not raw_mean or not raw_std:
            raise ValueError(
                "normalization.source is 'supplied' but "
                "config.training.normalization.mean and .std are not set."
            )
        if len(raw_mean) != len(raw_std):
            raise ValueError(
                f"normalization.mean ({len(raw_mean)} values) and "
                f"normalization.std ({len(raw_std)} values) must have the same length."
            )

        std_safe = tuple(
            max(float(s), 1e-8) for s in raw_std
        )

        return NormalizationStatistics(
            band_names   = tuple(str(n) for n in raw_names)[:len(raw_mean)],
            mean         = tuple(float(m) for m in raw_mean),
            std          = std_safe,
            num_samples  = 0,
            source       = "supplied",
        )

    def _compute_statistics(
        self,
        train_dataset: Any,
    ) -> NormalizationStatistics:
        """
        Compute per-band mean and std from the training dataset.

        Iterates over all training samples using a streaming accumulator
        to avoid loading the entire dataset into memory.
        """
        n = len(train_dataset)
        if n == 0:
            raise ValueError(
                "Cannot compute normalization statistics: training dataset is empty."
            )

        # Determine num_bands from first sample.
        first_item = train_dataset[0]
        first_image = _to_numpy_float32(first_item[0])
        num_bands = first_image.shape[0]

        accumulator = DatasetStatisticsAccumulator(num_bands=num_bands)
        accumulator.update(first_image)

        self._logger.info(
            "Computing normalization statistics from %d training samples "
            "(%d bands)...", n, num_bands,
        )

        for i in range(1, n):
            item  = train_dataset[i]
            image = _to_numpy_float32(item[0])
            accumulator.update(image)

        stats = accumulator.finalize(num_samples=n)
        self._logger.info("Normalization statistics computed.")
        return stats

    def _detect_dimensions(
        self,
        dataset: Any,
    ) -> tuple[int, tuple[int, int] | None]:
        """
        Detect num_bands and patch_size (H, W) from the first dataset item.

        Returns (num_bands, patch_size) or (0, None) on failure.
        """
        try:
            if len(dataset) == 0:
                return 0, None
            item  = dataset[0]
            image = _to_numpy_float32(item[0])
            if image.ndim == 3:
                c, h, w = image.shape
                return c, (h, w)
        except Exception as exc:
            self._logger.warning(
                "TransformPipeline._detect_dimensions failed: %s", exc
            )
        return 0, None

    def _build_metadata(
        self,
        stats:               NormalizationStatistics,
        train_pipeline:      ComposedTransform,
        val_pipeline:        ComposedTransform,
        test_pipeline:       ComposedTransform,
    ) -> TransformMetadata:
        """Build TransformMetadata for provenance tracking."""
        timestamp = datetime.now(timezone.utc).isoformat()

        config_hash: str | None = None
        try:
            train_cfg = getattr(self._config, "training", None)
            config_str = str(vars(train_cfg) if train_cfg else "")
            config_hash = hashlib.sha256(
                config_str.encode("utf-8")
            ).hexdigest()[:8]
        except Exception:
            pass

        return TransformMetadata(
            pipeline_version     = self._pipeline_version,
            random_seed          = self._seed,
            normalization_source  = stats.source,
            normalization_stats   = stats,
            train_augmentations  = train_pipeline.transform_names,
            val_augmentations    = val_pipeline.transform_names,
            test_augmentations   = test_pipeline.transform_names,
            created_at           = timestamp,
            config_hash          = config_hash,
        )


# ==============================================================================
# Private tensor conversion helpers
# ==============================================================================

def _to_numpy_float32(tensor: Any) -> np.ndarray:
    """Convert a torch Tensor or numpy array to (C, H, W) float32 ndarray."""
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    try:
        return tensor.numpy().astype(np.float32)
    except AttributeError:
        return np.array(tensor, dtype=np.float32)


def _to_numpy_uint8(tensor: Any) -> np.ndarray:
    """Convert a torch Tensor or numpy array to (H, W) uint8 ndarray."""
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.uint8)
    try:
        return tensor.numpy().astype(np.uint8)
    except AttributeError:
        return np.array(tensor, dtype=np.uint8)


def _to_torch_float32(array: np.ndarray) -> Any:
    """Convert a numpy float32 array to a float32 torch Tensor."""
    try:
        import torch
        return torch.from_numpy(np.ascontiguousarray(array)).float()
    except ImportError:
        return array


def _to_torch_long(array: np.ndarray) -> Any:
    """Convert a numpy uint8 array to a long (int64) torch Tensor for CE loss."""
    try:
        import torch
        return torch.from_numpy(np.ascontiguousarray(array.astype(np.int64))).long()
    except ImportError:
        return array
