"""
DataLoader factory for the River Morphology training pipeline (Module 11).

DataLoaderFactory orchestrates all five training components:
    DatasetNormalizer    -> NormalizationStats (from training split)
    AugmentationPipeline -> Transform instances (AlbumentationsTransform
                           for training, IdentityTransform for eval)
    RiverMorphologyDataset -> per-split Dataset
    TemporalSampler      -> WeightedRandomSampler (optional)
    ClassWeights         -> loss weight tensor

Returns a frozen DataLoaderBundle with all three DataLoaders,
NormalizationStats, ClassWeights, and the path to the persisted
normalization_stats.json.

Input:  TrainingDatasetResult (Module 10)
Output: DataLoaderBundle (immutable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.dataset.manifest import DatasetManifest
from src.training.dataset import RiverMorphologyDataset
from src.training.normalizer import DatasetNormalizer, NormalizationStats
from src.training.sampler import TemporalSampler
from src.training.transforms import AugmentationConfig, AugmentationPipeline
from src.training.weights import ClassWeights

if TYPE_CHECKING:
    from src.dataset.assembler import TrainingDatasetResult

__all__ = ["DataLoaderConfig", "DataLoaderBundle", "DataLoaderFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# DataLoaderConfig
# ==============================================================================

@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Immutable DataLoader construction parameters.

    Attributes:
        batch_size:           Number of samples per batch.
        num_workers:           Number of worker processes.
        pin_memory:            Pin memory buffers for faster GPU transfer.
        prefetch_factor:        Batches to prefetch per worker.
        persistent_workers:     Keep workers alive across epochs.
        train_shuffle:           Shuffle training samples each epoch.
                                Overridden to False when a custom sampler is
                                provided (sampler implies shuffle=False).
    """

    batch_size:          int   = 8
    num_workers:          int   = 4
    pin_memory:           bool  = True
    prefetch_factor:       int   = 2
    persistent_workers:    bool  = True
    train_shuffle:          bool  = True

    @classmethod
    def from_config(cls, config: Any) -> DataLoaderConfig:
        """Build DataLoaderConfig from config.training.dataloader."""
        train_cfg = getattr(config, "training", None)
        dl_cfg    = getattr(train_cfg, "dataloader", None)
        if dl_cfg is None:
            return cls()
        return cls(
            batch_size         = int(getattr(dl_cfg, "batch_size",         8)),
            num_workers         = int(getattr(dl_cfg, "num_workers",         4)),
            pin_memory          = bool(getattr(dl_cfg, "pin_memory",          True)),
            prefetch_factor      = int(getattr(dl_cfg, "prefetch_factor",      2)),
            persistent_workers   = bool(getattr(dl_cfg, "persistent_workers",   True)),
            train_shuffle        = bool(getattr(dl_cfg, "train_shuffle",        True)),
        )


# ==============================================================================
# DataLoaderBundle
# ==============================================================================

@dataclass(frozen=True)
class DataLoaderBundle:
    """
    Immutable bundle of all DataLoader objects for one training run.

    Attributes:
        train_loader:          DataLoader for the training split.
        val_loader:             DataLoader for the validation split.
        test_loader:            DataLoader for the test split.
        norm_stats:             NormalizationStats from training split.
        class_weights:           Per-class loss weights.
        norm_stats_path:          Path to persisted normalization_stats.json.
        train_dataset_size:       Number of training samples.
        val_dataset_size:          Number of validation samples.
        test_dataset_size:          Number of test samples.
        num_bands:               Number of spectral bands per patch.
        num_classes:              Number of segmentation classes.
        split_strategy:            Strategy used to create the splits.
        aug_config:                Active AugmentationConfig.
        dl_config:                 Active DataLoaderConfig.
    """

    train_loader:         Any   # torch.utils.data.DataLoader
    val_loader:            Any   # torch.utils.data.DataLoader
    test_loader:           Any   # torch.utils.data.DataLoader
    norm_stats:            NormalizationStats
    class_weights:          ClassWeights
    norm_stats_path:         Path
    train_dataset_size:      int
    val_dataset_size:         int
    test_dataset_size:         int
    num_bands:               int
    num_classes:              int
    split_strategy:           str
    aug_config:               AugmentationConfig
    dl_config:                DataLoaderConfig
    train_dataset:            Any
    validation_dataset:       Any
    test_dataset:             Any

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        return [
            f"  train:        {self.train_dataset_size} samples",
            f"  validation:   {self.val_dataset_size} samples",
            f"  test:         {self.test_dataset_size} samples",
            f"  num_bands:    {self.num_bands}",
            f"  num_classes:  {self.num_classes}",
            f"  batch_size:   {self.dl_config.batch_size}",
            f"  strategy:     {self.split_strategy}",
            f"  augmentation: {self.aug_config.enabled}",
        ]


# ==============================================================================
# DataLoaderFactory
# ==============================================================================

class DataLoaderFactory:
    """
    Assembles all training components into a DataLoaderBundle.

    Orchestrates DatasetNormalizer, AugmentationPipeline,
    RiverMorphologyDataset, TemporalSampler, and ClassWeights.
    Contains no pixel I/O logic of its own.

    Args:
        config:       Fully initialized Config object.
        class_schema: ClassSchema from Module 9 (defines class taxonomy).
    """

    def __init__(self, config: Config, class_schema: Any) -> None:
        self._config       = config
        self._class_schema = class_schema
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._aug_config = AugmentationConfig.from_config(config)
        self._dl_config  = DataLoaderConfig.from_config(config)

        labels_cfg         = getattr(config, "labels", None)
        self._nodata_value = int(getattr(labels_cfg, "nodata_value", 255))

        self._normalizer      = DatasetNormalizer(config)
        self._aug_pipeline    = AugmentationPipeline(self._aug_config)
        self._sampler_builder = TemporalSampler(config)

        self._logger.debug("DataLoaderFactory initialized.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        training_result: TrainingDatasetResult,
        output_dir:      Path,
        read_masks_for_weights: bool = True,
    ) -> DataLoaderBundle:
        """
        Build a DataLoaderBundle from a TrainingDatasetResult.

        Args:
            training_result:          From DatasetAssembler.assemble().
            output_dir:                Directory to persist normalization_stats.json.
            read_masks_for_weights:    Unused in this method; class weights come
                                      from training_result.train_statistics which
                                      was computed during assembly.

        Returns:
            Frozen DataLoaderBundle with all three DataLoaders.

        Raises:
            InvalidValueError: Training split is empty.
            ImportError:        torch is not installed.
        """
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ImportError(
                "torch is not installed. Install with: pip install torch>=2.0"
            ) from exc

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = training_result.manifest

        train_entries = list(manifest.entries_for_split("train"))
        val_entries   = list(manifest.entries_for_split("validation"))
        test_entries  = list(manifest.entries_for_split("test"))

        if not train_entries:
            raise InvalidValueError(
                field="training split entries",
                value=0,
                reason="training split is empty; cannot build DataLoader",
            )

        # Determine band names and class names.
        num_bands  = int(train_entries[0].num_bands) if train_entries else 0
        band_names = self._resolve_band_names(training_result, num_bands)
        class_names = self._class_schema.class_names
        num_classes = self._class_schema.num_classes

        # Step 1: Compute normalization stats from training split only.
        norm_stats = self._normalizer.compute(train_entries, band_names)
        norm_stats_path = self._normalizer.save_to_dir(norm_stats, output_dir)
        self._logger.info("Normalization stats saved: %s", norm_stats_path.name)

        # Step 2: Build Transform instances via AugmentationPipeline.
        #         Training -> AlbumentationsTransform (or IdentityTransform when disabled).
        #         Eval     -> IdentityTransform always.
        train_transform = self._aug_pipeline.build_train_transform()
        eval_transform  = self._aug_pipeline.build_eval_transform()

        # Step 3: Build per-split Datasets.
        #         band_names / class_names / num_classes are passed so that
        #         the read-only Dataset properties return meaningful values.
        train_ds = RiverMorphologyDataset(
            entries       = train_entries,
            norm_stats    = norm_stats,
            transform     = train_transform,
            nodata_value  = self._nodata_value,
            ignore_index  = self._nodata_value,
            split         = "train",
            band_names    = band_names,
            class_names   = class_names,
            num_classes   = num_classes,
        )
        val_ds = RiverMorphologyDataset(
            entries       = val_entries,
            norm_stats    = norm_stats,
            transform     = eval_transform,
            nodata_value  = self._nodata_value,
            ignore_index  = self._nodata_value,
            split         = "validation",
            band_names    = band_names,
            class_names   = class_names,
            num_classes   = num_classes,
        )
        test_ds = RiverMorphologyDataset(
            entries       = test_entries,
            norm_stats    = norm_stats,
            transform     = eval_transform,
            nodata_value  = self._nodata_value,
            ignore_index  = self._nodata_value,
            split         = "test",
            band_names    = band_names,
            class_names   = class_names,
            num_classes   = num_classes,
        )

        # Step 4: Build optional sampler.
        sampler      = self._sampler_builder.build(train_entries)
        train_shuffle = self._dl_config.train_shuffle and sampler is None

        # Step 5: Build DataLoaders.
        persistent = (
            self._dl_config.persistent_workers
            and self._dl_config.num_workers > 0
        )
        prefetch = (
            self._dl_config.prefetch_factor
            if self._dl_config.num_workers > 0 else None
        )

        train_loader = DataLoader(
            dataset            = train_ds,
            batch_size         = self._dl_config.batch_size,
            shuffle            = train_shuffle,
            num_workers        = self._dl_config.num_workers,
            pin_memory         = self._dl_config.pin_memory,
            prefetch_factor    = prefetch,
            persistent_workers  = persistent,
            sampler            = sampler,
            collate_fn         = self._collate_fn,
        )
        val_loader = DataLoader(
            dataset            = val_ds,
            batch_size         = self._dl_config.batch_size,
            shuffle            = False,
            num_workers        = self._dl_config.num_workers,
            pin_memory         = self._dl_config.pin_memory,
            prefetch_factor    = prefetch,
            persistent_workers  = persistent,
            collate_fn         = self._collate_fn,
        )
        test_loader = DataLoader(
            dataset            = test_ds,
            batch_size         = self._dl_config.batch_size,
            shuffle            = False,
            num_workers        = self._dl_config.num_workers,
            pin_memory         = self._dl_config.pin_memory,
            prefetch_factor    = prefetch,
            persistent_workers  = persistent,
            collate_fn         = self._collate_fn,
        )

        # Step 6: Class weights from training statistics.
        class_weights = ClassWeights.from_config_and_statistics(
            config           = self._config,
            train_statistics = training_result.train_statistics,
            class_schema     = self._class_schema,
        )

        bundle = DataLoaderBundle(
            train_loader       = train_loader,
            val_loader          = val_loader,
            test_loader         = test_loader,
            train_dataset=train_ds,
            validation_dataset=val_ds,
            test_dataset=test_ds,
            norm_stats          = norm_stats,
            class_weights       = class_weights,
            norm_stats_path     = norm_stats_path,
            train_dataset_size  = len(train_ds),
            val_dataset_size     = len(val_ds),
            test_dataset_size    = len(test_ds),
            num_bands           = num_bands,
            num_classes         = num_classes,
            split_strategy      = training_result.split_strategy,
            aug_config          = self._aug_config,
            dl_config           = self._dl_config,
        )

        for line in bundle.summary_lines():
            self._logger.info(line)

        return bundle

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _collate_fn(
        batch: list[tuple[Any, Any, Any]],
    ) -> tuple[Any, Any, list[Any]]:
        """
        Stack image/mask tensors; collect SampleMetadata as a plain list.

        torch's default collate cannot handle dataclass objects in the third
        batch element. This collate function stacks the first two elements and
        keeps metadata as a list so training loops can access per-sample
        provenance.

        Returns:
            (stacked_images, stacked_masks, [SampleMetadata, ...])
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is not installed.") from exc

        images    = torch.stack([b[0] for b in batch])
        masks     = torch.stack([b[1] for b in batch])
        metadatas = [b[2] for b in batch]
        return images, masks, metadatas

    def _resolve_band_names(
        self,
        training_result: Any,
        num_bands: int,
    ) -> tuple[str, ...]:
        """
        Resolve band names from the training result.

        Falls back to generic "band_N" names when no other source is available.
        """
        try:
            from src.gee.harmonization import COMMON_BAND_NAMES
            if num_bands >= len(COMMON_BAND_NAMES):
                return COMMON_BAND_NAMES + tuple(
                    f"index_{i}"
                    for i in range(num_bands - len(COMMON_BAND_NAMES))
                )
        except Exception:
            pass
        return tuple(f"band_{i}" for i in range(num_bands))