"""
Core inference predictor for Module 16.

Predictor runs model forward passes in eval+no_grad mode, converts logits
to probabilities (softmax or sigmoid), and applies the confidence strategy.
It supports single-image, batch, and dataset-level inference.

AMP (float16) inference is supported on CUDA. CPU falls back to float32.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.training.inference.confidence import ConfidenceRegistry, ConfidenceStrategy
from src.training.inference.contracts import InferenceConfig, SamplePrediction

__all__ = ["Predictor"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


def _to_dict(value: Any) -> dict[str, Any]:
    """
    Convert supported metadata objects to a dictionary.

    Dictionaries are returned unchanged. Dataclasses are converted with
    asdict(). Compatible duck-typed metadata objects are read from their
    known public attributes.
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)

    fields = (
        "sample_id",
        "acquisition_date",
        "season",
        "hydrological_year",
        "sensor",
        "aoi_id",
        "river_name",
        "reach_id",
        "basin_id",
        "patch_path",
        "mask_path",
        "scene_id",
        "year",
        "month",
    )

    return {
        field: getattr(value, field)
        for field in fields
        if hasattr(value, field)
    }


class Predictor:
    """
    Runs model inference and converts logits to SamplePrediction objects.

    Args:
        config:             InferenceConfig.
        model:              torch.nn.Module in eval mode on the target device.
        confidence_strategy: ConfidenceStrategy instance.
        device:             torch.device.
        class_names:        Ordered class names.
    """

    def __init__(
        self,
        config:              InferenceConfig,
        model:               Any,
        confidence_strategy: ConfidenceStrategy,
        device:              Any,
        class_names:         tuple[str, ...],
    ) -> None:
        self._config    = config
        self._model     = model
        self._strategy  = confidence_strategy
        self._device    = device
        self._cls_names = class_names
        self._logger    = logging.getLogger(__name__)

    def predict_batch(
        self,
        images:   Any,
        metadata: list[dict] | None = None,
        sample_offset: int = 0,
    ) -> list[SamplePrediction]:
        """
        Run inference on a batch of images.

        Args:
            images:   (B, C, H, W) float32 torch.Tensor on CPU or device.
            metadata: Optional list of per-sample metadata dicts (length B).

        Returns:
            List of SamplePrediction, one per sample in the batch.
        """
        import torch
        import torch.nn.functional as F

        self._model.eval()
        images = images.to(self._device, non_blocking=True)

        use_amp = (
            self._config.mixed_precision
            and getattr(self._device, "type", str(self._device)) == "cuda"
        )

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self._model(images)
            else:
                logits = self._model(images)

        # Convert to float32 CPU numpy for all post-processing.
        logits_np = logits.float().cpu().numpy()   # (B, C, H, W)

        # Compute probabilities.
        mode = self._config.probability_mode.lower().strip()
        if mode == "sigmoid":
            prob_tensor = torch.sigmoid(torch.from_numpy(logits_np))
        else:
            prob_tensor = torch.softmax(torch.from_numpy(logits_np), dim=1)
        probs_np = prob_tensor.numpy()   # (B, C, H, W)

        # Argmax → predicted class IDs.
        preds_np = probs_np.argmax(axis=1).astype(np.uint8)   # (B, H, W)

        results: list[SamplePrediction] = []
        B = logits_np.shape[0]
        for i in range(B):
            conf_map = self._strategy.compute(probs_np[i])   # (H, W)
            meta     = (metadata[i] if metadata else {}) or {}

            sample = SamplePrediction(
                sample_id          = str(
                    meta.get("sample_id")
                    or f"sample_{sample_offset + i}"
                ),
                predicted_mask     = preds_np[i],
                probabilities      = probs_np[i],
                confidence         = conf_map,
                logits             = logits_np[i] if self._config.export_numpy else None,
                acquisition_date   = str(meta.get("acquisition_date",   "")),
                season             = str(meta.get("season",             "")),
                hydrological_year  = int(meta.get("hydrological_year",  0)),
                sensor             = str(meta.get("sensor",             "")),
                river_name         = str(meta.get("river_name",         "")),
                reach_id           = str(meta.get("reach_id",           "")),
                basin_id           = str(meta.get("basin_id",           "")),
                aoi_id             = str(meta.get("aoi_id",             "")),
                patch_path         = str(meta.get("patch_path",         "")),
                mask_path          = str(meta.get("mask_path",          "")),
                scene_id           = str(meta.get("scene_id",           "")),
                year               = int(meta.get("year",               0)),
                month              = int(meta.get("month",              0)),
                metadata           = dict(meta),
            )
            results.append(sample)

        return results

    def predict_single(
        self,
        image:    np.ndarray,
        metadata: dict | None = None,
    ) -> SamplePrediction:
        """
        Run inference on a single (C, H, W) numpy image.

        Args:
            image:    (C, H, W) float32 numpy array.
            metadata: Optional metadata dict.

        Returns:
            SamplePrediction.
        """
        import torch
        tensor = torch.from_numpy(image[np.newaxis]).float()   # (1, C, H, W)
        return self.predict_batch(tensor, [metadata or {}])[0]

    def predict_dataset(
        self,
        dataloader: Any,
    ) -> list[SamplePrediction]:
        """
        Run inference over an entire DataLoader.

        Supports batches containing:
            (images, masks)
            (images, masks, list[dict])
            (images, masks, dict[str, batched values])

        PyTorch default_collate converts per-sample metadata dictionaries
        into a dictionary whose values are batched. This method converts
        that representation back into one dictionary per sample.
        """
        import torch

        all_predictions: list[SamplePrediction] = []
        self._model.eval()

        sample_offset = 0

        for batch in dataloader:
            images = batch[0]
            batch_size = int(images.shape[0])

            metadata: list[dict[str, Any]] | None = None

            if len(batch) >= 3:
                candidate = batch[2]

                if isinstance(candidate, dict):
                    metadata = []

                    for sample_index in range(batch_size):
                        sample_metadata: dict[str, Any] = {}

                        for key, values in candidate.items():
                            try:
                                value = values[sample_index]
                            except (
                                IndexError,
                                KeyError,
                                TypeError,
                            ):
                                value = values

                            if hasattr(value, "item"):
                                try:
                                    value = value.item()
                                except (
                                    RuntimeError,
                                    TypeError,
                                    ValueError,
                                ):
                                    pass

                            sample_metadata[str(key)] = value

                        metadata.append(sample_metadata)

                elif (
                    isinstance(candidate, (list, tuple))
                    and len(candidate) == batch_size
                    and all(
                        isinstance(item, dict)
                        for item in candidate
                    )
                ):
                    metadata = [
                        dict(item)
                        for item in candidate
                    ]

            if metadata is None:
                metadata = [
                    {
                        "sample_id": (
                            f"sample_{sample_offset + sample_index}"
                        )
                    }
                    for sample_index in range(batch_size)
                ]
            else:
                for sample_index, sample_metadata in enumerate(metadata):
                    sample_metadata.setdefault(
                        "sample_id",
                        f"sample_{sample_offset + sample_index}",
                    )

            with torch.no_grad():
                batch_predictions = self.predict_batch(
                    images,
                    metadata=metadata,
                )

            all_predictions.extend(batch_predictions)
            sample_offset += batch_size

        return all_predictions
