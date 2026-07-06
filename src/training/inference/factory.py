"""
Inference factory for Module 16.

InferenceFactory assembles CheckpointLoader, Predictor, PostprocessorPipeline,
PredictionExporter, and DataLoader from an InferenceConfig.
InferenceEngine calls this internally.
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.inference.confidence import ConfidenceRegistry
from src.training.inference.contracts import InferenceConfig
from src.training.inference.exporter import PredictionExporter
from src.training.inference.loader import CheckpointLoader
from src.training.inference.postprocessing import PostprocessorPipeline
from src.training.inference.predictor import Predictor

__all__ = ["InferenceFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class InferenceFactory:
    """Assembles all inference components from config."""

    @classmethod
    def build(
        cls,
        config:      InferenceConfig,
        model:       Any,
        class_names: tuple[str, ...],
    ) -> dict:
        """
        Build the complete inference context.

        Args:
            config:       InferenceConfig.
            model:        torch.nn.Module (weights already restored).
            class_names:  Ordered class names.

        Returns:
            Dict with keys: device, predictor, postprocessor,
            exporter, model.
        """
        # Step 1: Resolve and seed device.
        device = cls._resolve_device(config)

        # Step 2: Seed for determinism.
        if config.deterministic:
            cls._seed(config.seed, deterministic=True)
        else:
            cls._seed(config.seed, deterministic=False)

        # Step 3: Move model to device and set eval mode.
        model.to(device)
        model.eval()

        # Step 4: Build confidence strategy.
        confidence_strategy = ConfidenceRegistry.build(config.confidence_strategy)

        # Step 5: Build predictor.
        predictor = Predictor(
            config              = config,
            model               = model,
            confidence_strategy = confidence_strategy,
            device              = device,
            class_names         = class_names,
        )

        # Step 6: Build post-processing pipeline.
        postprocessor = PostprocessorPipeline.build_from_config(config) \
            if config.postprocess else PostprocessorPipeline([])

        # Step 7: Build exporter.
        exporter = PredictionExporter(
            config      = config,
            class_names = class_names,
        )

        return {
            "device":        device,
            "predictor":     predictor,
            "postprocessor": postprocessor,
            "exporter":      exporter,
            "model":         model,
        }

    @classmethod
    def build_dataloader(
        cls,
        config:      InferenceConfig,
        dataset:     Any,
        device:      Any,
    ) -> Any:
        """
        Build a deterministic (shuffle=False) DataLoader for inference.

        Args:
            config:  InferenceConfig.
            dataset: torch.utils.data.Dataset.
            device:  torch.device.

        Returns:
            DataLoader.
        """
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
            return _SimpleBatchLoader(dataset, config.batch_size)

    @staticmethod
    def _resolve_device(config: InferenceConfig) -> Any:
        """Resolve device string, falling back to CPU when CUDA unavailable."""
        try:
            import torch
            if config.device.startswith("cuda"):
                if torch.cuda.is_available():
                    return torch.device(config.device)
                _LOGGER.warning(
                    "InferenceFactory: CUDA requested but unavailable; using CPU."
                )
                return torch.device("cpu")
            return torch.device(config.device)
        except ImportError:
            return "cpu"

    @staticmethod
    def _seed(seed: int, deterministic: bool) -> None:
        """Seed torch/numpy/python for deterministic inference."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if deterministic:
                torch.use_deterministic_algorithms(True)
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark     = False
                except AttributeError:
                    pass
        except ImportError:
            pass


class _SimpleBatchLoader:
    """Minimal DataLoader stub when torch DataLoader is unavailable."""

    def __init__(self, dataset: Any, batch_size: int) -> None:
        self._dataset    = dataset
        self._batch_size = max(1, batch_size)

    def __iter__(self):
        import torch
        n = len(self._dataset)
        for start in range(0, n, self._batch_size):
            end    = min(start + self._batch_size, n)
            images = []
            for i in range(start, end):
                item = self._dataset[i]
                images.append(item[0])
            yield torch.stack(images), None, [{}] * len(images)

    def __len__(self) -> int:
        import math
        return math.ceil(len(self._dataset) / self._batch_size)
