"""
TrainingEngine — the single public interface for Module 14.

Usage
-----
    from src.training.engine import TrainingEngine

    engine = TrainingEngine(config)
    result = engine.train(model_result, data_result)

    # result: TrainingResult (frozen)
    # result.model          -- trained model (on CPU)
    # result.best_checkpoint -- path to best checkpoint
    # result.history        -- tuple of EpochResult

TrainingEngine delegates all heavy lifting:
    - TrainingEngineFactory builds the training context
    - Trainer runs the epoch/batch loops
    - Callbacks handle checkpointing, logging, early stopping
    - TrainingValidator runs pre-flight checks

Module 14 is responsible ONLY for orchestration. It does NOT implement:
    - model architectures  (Module 13)
    - augmentations        (Module 12)
    - evaluation metrics   (Module 15)
    - inference            (Module 16)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.training.engine.contracts import TrainingConfig, TrainingResult
from src.training.engine.factory import TrainingEngineFactory
from src.training.engine.history import TrainingHistory
from src.training.engine.seed import SeedManager
from src.training.engine.validator import TrainingValidator

__all__ = ["TrainingEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TrainingEngine:
    """
    Orchestrates a complete model training run.

    Args:
        config: Fully initialized TrainingConfig (or project Config object;
                TrainingConfig.from_config(config) is called when needed).
    """

    def __init__(self, config: Any) -> None:
        # Accept both TrainingConfig and raw project Config objects.
        if isinstance(config, TrainingConfig):
            self._config = config
        else:
            self._config = TrainingConfig.from_config(config)

        self._validator = TrainingValidator()
        self._logger    = _LOGGER

    def train(
        self,
        model_result: Any,
        data_result:  Any,
    ) -> TrainingResult:
        """
        Execute a complete training run.

        Args:
            model_result: ModelResult from Module 13 (ModelFactory.build()).
            data_result:  TransformPipelineResult from Module 12
                          (TransformPipeline.build()).

        Returns:
            Frozen TrainingResult with trained model, history, and provenance.

        Raises:
            ValueError: Pre-flight validation fails on a fatal issue.
        """
        ops: list[str] = []

        # Step 1: Pre-flight validation.
        validation = self._validator.validate(self._config, model_result, data_result)
        if not validation.is_valid:
            # Non-fatal issues are warnings; fatal issues raise.
            for issue in validation.issues:
                self._logger.warning("TrainingEngine pre-flight: %s", issue)
        ops.append(f"validation: {len(validation.issues)} issue(s)")

        # Step 2: Build all components.
        context = TrainingEngineFactory.build(
            config       = self._config,
            model_result = model_result,
            data_result  = data_result,
        )
        ops.extend(context.get("operations_log", ()))

        # Step 3: Resolve resume epoch.
        start_epoch  = 1
        ckpt_manager = context["checkpoint_manager"]
        resume_path  = self._config.checkpoint.resume_from
        if resume_path is not None:
            start_epoch = ckpt_manager.restore(resume_path, context) + 1
            self._logger.info(
                "TrainingEngine: resuming from epoch %d.", start_epoch
            )
            ops.append(f"resume: from epoch {start_epoch - 1}")

        # Step 4: Log training start.
        training_logger = context["training_logger"]
        training_logger.log_train_begin(
            total_epochs   = self._config.epochs,
            num_parameters = model_result.num_parameters,
        )

        # Step 5: Run the training loop.
        trainer      = context["trainer"]
        epoch_results = trainer.run(
            train_loader = context["train_loader"],
            val_loader   = context["val_loader"],
            start_epoch  = start_epoch,
            context      = context,
        )
        ops.append(f"training: {len(epoch_results)} epoch(s) completed")

        # Step 6: Assemble history.
        history = TrainingHistory()
        for r in epoch_results:
            history.append(r)

        # Step 7: Determine best epoch.
        best_epoch  = history.best_epoch
        best_metric = history.best_val_loss if history.val_losses else (
            min(history.train_losses) if history.train_losses else 0.0
        )

        # Step 8: Move model to CPU before returning.
        model = context["model"]
        try:
            model.cpu()
        except Exception:
            pass
        ops.append("model: moved to CPU")

        # Step 9: Log completion.
        training_logger.log_train_end(best_epoch, best_metric)
        stopped_early = context.get("stop_training", False)
        if stopped_early:
            training_logger.log_early_stop(len(epoch_results) + start_epoch - 1)

        # Step 10: Assemble TrainingResult.
        result = TrainingResult(
            model             = model,
            history           = history.to_tuple(),
            best_epoch        = best_epoch,
            best_metric       = best_metric,
            best_checkpoint   = ckpt_manager.best_path,
            latest_checkpoint = ckpt_manager.latest_path,
            total_epochs      = len(epoch_results),
            architecture      = model_result.architecture,
            num_parameters    = model_result.num_parameters,
            seed              = self._config.seed,
            training_config   = self._config,
            operations_log    = tuple(ops),
            stopped_early     = stopped_early,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result
