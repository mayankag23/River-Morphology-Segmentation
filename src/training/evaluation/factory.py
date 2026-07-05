"""
Evaluation factory for Module 15.

EvaluationFactory assembles the DataLoader, device, and Evaluator from
an EvaluationConfig, TrainingResult, and TransformPipelineResult.
EvaluationEngine calls this internally; users never call it directly.
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.evaluation.contracts import EvaluationConfig
from src.training.evaluation.evaluator import Evaluator

__all__ = ["EvaluationFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class EvaluationFactory:
    """Assembles Evaluator and DataLoader from config and data contracts."""

    @classmethod
    def build(
        cls,
        config:      EvaluationConfig,
        data_result: Any,
        num_classes: int,
        class_names: tuple[str, ...],
    ) -> tuple[Evaluator, Any]:
        """
        Build an Evaluator and the appropriate DataLoader.

        Args:
            config:      EvaluationConfig.
            data_result: TransformPipelineResult from Module 12.
            num_classes: Number of segmentation classes.
            class_names: Ordered class names.

        Returns:
            (Evaluator, DataLoader) tuple.
        """
        device   = cls._resolve_device(config.device)
        dataset  = cls._select_dataset(config.split, data_result)
        loader   = cls._build_loader(config, dataset, device)
        evaluator = Evaluator(
            config      = config,
            num_classes = num_classes,
            class_names = class_names,
            device      = device,
        )
        return evaluator, loader

    @staticmethod
    def _resolve_device(device_str: str) -> Any:
        """Resolve device string to torch.device, falling back to CPU."""
        try:
            import torch
            if device_str.startswith("cuda") and not torch.cuda.is_available():
                _LOGGER.warning(
                    "EvaluationFactory: CUDA requested but unavailable; using CPU."
                )
                return torch.device("cpu")
            return torch.device(device_str)
        except ImportError:
            return "cpu"

    @staticmethod
    def _select_dataset(split: str, data_result: Any) -> Any:
        """Select the correct dataset from TransformPipelineResult."""
        split = split.lower().strip()
        if split == "test":
            return data_result.test_dataset
        if split in ("val", "validation"):
            return data_result.validation_dataset
        if split == "train":
            return data_result.train_dataset
        _LOGGER.warning(
            "EvaluationFactory: unknown split '%s'; defaulting to test_dataset.", split
        )
        return data_result.test_dataset

    @staticmethod
    def _build_loader(
        config:  EvaluationConfig,
        dataset: Any,
        device:  Any,
    ) -> Any:
        """Build a deterministic (shuffle=False) DataLoader."""
        try:
            import torch
            from torch.utils.data import DataLoader
            device_type = getattr(device, "type", str(device))
            return DataLoader(
                dataset,
                batch_size  = config.batch_size,
                shuffle     = False,
                num_workers = config.num_workers,
                pin_memory  = config.pin_memory and device_type == "cuda",
                drop_last   = False,
            )
        except ImportError:
            # Fallback for torch-free test environments: wrap in a simple list.
            return _SimpleLoader(dataset, config.batch_size)


class _SimpleLoader:
    """Minimal DataLoader stub for environments without torch DataLoader."""

    def __init__(self, dataset: Any, batch_size: int) -> None:
        self._dataset    = dataset
        self._batch_size = max(1, batch_size)

    def __iter__(self):
        import torch
        n     = len(self._dataset)
        for start in range(0, n, self._batch_size):
            end    = min(start + self._batch_size, n)
            images = []
            masks  = []
            for i in range(start, end):
                item = self._dataset[i]
                images.append(item[0])
                masks.append(item[1])
            yield torch.stack(images), torch.stack(masks)

    def __len__(self) -> int:
        import math
        return math.ceil(len(self._dataset) / self._batch_size)
