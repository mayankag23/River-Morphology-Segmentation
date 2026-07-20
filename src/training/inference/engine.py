"""
InferenceEngine -- the single public interface for Module 16.

Usage
-----
    from src.training.inference import InferenceEngine, InferenceConfig

    config = InferenceConfig(checkpoint_strategy="best",
                             checkpoint_dir="checkpoints",
                             device="cpu", export_numpy=True)
    engine = InferenceEngine(config)
    result = engine.predict(training_result, data_result,
                            class_names=("background","water","sand","vegetation"))

    for pred in result.predictions:
        print(pred.sample_id, pred.confidence.mean())

InferenceEngine orchestrates:
    CheckpointLoader  -> resolves path, loads weights, extracts metadata
    InferenceFactory  -> assembles Predictor, PostprocessorPipeline, Exporter
    Predictor         -> logits -> probabilities -> SamplePrediction
    PostprocessorPipeline -> optional mask cleanup
    PredictionExporter -> numpy / GeoTIFF / PNG export
    InferenceValidator -> pre-flight + per-batch output validation
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pathlib import Path
import numpy as np

from src.training.inference.contracts import (
    InferenceConfig,
    InferenceResult,
    SamplePrediction,
)
from src.training.inference.factory import InferenceFactory
from src.training.inference.loader import CheckpointLoader
from src.training.inference.validator import InferenceValidator

__all__ = ["InferenceEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Orchestrates a complete inference run.

    Args:
        config: InferenceConfig or project Config object.
    """

    def __init__(self, config: Any) -> None:
        # if isinstance(config, InferenceConfig):
        #     self._config = config
        # else:
        #     self._config = InferenceConfig.from_config(config)
        # self._validator = InferenceValidator()
        # self._logger    = _LOGGER
        # Keep the original project config
        self._project_config = config

        if isinstance(config, InferenceConfig):
            self._config = config
        else:
            self._config = InferenceConfig.from_config(config)

        self._validator = InferenceValidator()
        self._logger = _LOGGER

    def predict(
        self,
        training_result:  Any,
        data_result:      Any | None = None,
        class_names:      tuple[str, ...] | None = None,
        num_classes:      int | None = None,
        model:            Any | None = None,
    ) -> InferenceResult:
        """
        Run a complete inference pass.

        Args:
            training_result: TrainingResult from Module 14 (carries .model
                             and .architecture), or any object with a .model
                             attribute.  When model is provided directly,
                             training_result is used only for provenance.
            data_result:     TransformPipelineResult from Module 12 (for
                             dataset inference).  None = no dataset inference;
                             the returned InferenceResult will have 0 predictions.
            class_names:     Ordered class names.  When None, derived from
                             num_classes or checkpoint metadata.
            num_classes:     Number of classes.  When None, taken from
                             checkpoint metadata or data_result.
            model:           Optional override model (e.g. for TorchScript).
                             When None, training_result.model is used.

        Returns:
            Frozen InferenceResult.
        """
        ops: list[str] = []
        t0             = time.perf_counter()

        # Step 1: Load checkpoint and restore model weights.
        loader     = CheckpointLoader(self._config)
        ckpt_path  = loader.resolve_path()
        payload    = loader.load(ckpt_path)
        ckpt_meta  = loader.extract_metadata(ckpt_path, payload)
        ops.append(
            f"checkpoint: {ckpt_path.name} "
            f"(epoch={ckpt_meta.epoch}, arch={ckpt_meta.architecture})"
        )

        # Step 2: Resolve model.
        resolved_model = model or getattr(training_result, "model", None)
        if resolved_model is None:
            raise ValueError(
                "InferenceEngine: no model available. "
                "Provide training_result.model or the model kwarg."
            )
        loader.restore_model(resolved_model, payload)
        ops.append("model weights restored")

        # Step 3: Resolve class metadata.
        resolved_nc, resolved_names = self._resolve_classes(
            ckpt_meta, data_result, num_classes, class_names
        )
        ops.append(
            f"classes: {resolved_nc} ({', '.join(resolved_names)})"
        )

        # Step 4: Pre-flight validation.
        validation = self._validator.validate_config(self._config, ckpt_meta)
        for issue in validation.issues:
            self._logger.warning("InferenceEngine pre-flight: %s", issue)
        ops.append(f"validation: {len(validation.issues)} issue(s)")

        # Step 5: Build inference context.
        context = InferenceFactory.build(
            config      = self._config,
            model       = resolved_model,
            class_names = resolved_names,
        )
        device_used = str(context["device"])
        ops.append(f"device: {device_used}")

        # Step 6: Run inference.
        all_predictions: list[SamplePrediction] = []

        if data_result is not None:
            dataset = self._select_dataset(data_result)
            dataloader = InferenceFactory.build_dataloader(
                self._config, dataset, context["device"]
            )
            raw_preds = context["predictor"].predict_dataset(dataloader)

            # Step 7: Apply post-processing to each prediction.
            postprocessor = context["postprocessor"]
            for pred in raw_preds:
                pred.predicted_mask = postprocessor.apply(pred.predicted_mask)
                all_predictions.append(pred)

            ops.append(f"inference: {len(all_predictions)} samples")
        else:
            ops.append("inference: skipped (no data_result)")

        # Step 8: Export.
        exporter = context["exporter"]
        exporter.export_all(all_predictions)
        ops.append(
            f"export: numpy={self._config.export_numpy}, "
            f"geotiff={self._config.export_geotiff}, "
            f"png={self._config.export_png}"
        )

        # Step 9: Compute aggregate statistics.
        total_s      = time.perf_counter() - t0
        per_sample_ms = (total_s / max(1, len(all_predictions))) * 1000.0
        mean_conf    = float(
            np.mean([p.confidence.mean() for p in all_predictions])
            if all_predictions else 0.0
        )
        class_pixel_counts = self._count_class_pixels(all_predictions, resolved_names)
        ops.append(f"total_time: {total_s:.2f}s")

        # Step 10: Assemble InferenceResult.
        result = InferenceResult(
            predictions         = tuple(all_predictions),
            num_samples         = len(all_predictions),
            architecture        = ckpt_meta.architecture,
            num_classes         = resolved_nc,
            class_names         = resolved_names,
            checkpoint_meta     = ckpt_meta,
            inference_config    = self._config,
            device_used         = device_used,
            total_inference_s   = total_s,
            per_sample_ms       = per_sample_ms,
            operations_log      = tuple(ops),
            mean_confidence     = mean_conf,
            class_pixel_counts  = class_pixel_counts,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_classes(
        ckpt_meta:    Any,
        data_result:  Any,
        num_classes:  int | None,
        class_names:  tuple[str, ...] | None,
    ) -> tuple[int, tuple[str, ...]]:
        """Resolve num_classes and class_names from all available sources."""
        nc = (
            num_classes
            or getattr(data_result, "num_classes", None)
            or ckpt_meta.num_classes
            or 4
        )
        if class_names:
            return nc, tuple(class_names)[:nc]
        names_from_data = getattr(data_result, "class_names", None)
        if names_from_data:
            return nc, tuple(names_from_data)[:nc]
        return nc, tuple(f"class_{i}" for i in range(nc))

    @staticmethod
    def _select_dataset(data_result: Any) -> Any:
        """Select test_dataset from TransformPipelineResult."""
        return getattr(data_result, "test_dataset", data_result)

    @staticmethod
    def _count_class_pixels(
        predictions: list[SamplePrediction],
        class_names: tuple[str, ...],
    ) -> dict[str, int]:
        """Count total predicted pixels per class across all samples."""
        counts: dict[str, int] = {n: 0 for n in class_names}
        for pred in predictions:
            mask = pred.predicted_mask
            for i, name in enumerate(class_names):
                counts[name] += int((mask == i).sum())
        return counts
    
    def predict_aoi(
        self,
        raster_path: str,
        output_dir: str,
    ):
        """
        Run inference on a complete GeoTIFF AOI.
        """

        from src.deployment.predictor import AOIPredictor

        # predictor = AOIPredictor(
        #     config=self._config,
        #     model=resolved_model,
        # )
        predictor = AOIPredictor(self._project_config)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictor.save_prediction(
            raster_path=raster_path,
            output_png=output_dir / "prediction.png",
        )

        return output_dir    
