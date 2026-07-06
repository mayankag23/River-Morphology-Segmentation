"""
Report factory for Module 19.

ReportFactory assembles ExperimentManager, ReportGenerator, ReportExporter,
ArtifactManager, and ManifestManager from ReportingConfig and upstream results.
"""

from __future__ import annotations

import logging
from typing import Any

from src.reporting.artifact import ArtifactManager
from src.reporting.contracts import ReportingConfig
from src.reporting.experiment import ExperimentManager
from src.reporting.exporter import ReportExporter
from src.reporting.manifest import ManifestManager

__all__ = ["ReportFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ReportFactory:
    """Assembles all reporting components from config."""

    @classmethod
    def build(
        cls,
        config:            ReportingConfig,
        inference_result:  Any | None = None,
        evaluation_result: Any | None = None,
        total_duration_s:  float = 0.0,
    ) -> dict:
        """
        Build the complete reporting context.

        Args:
            config:            ReportingConfig.
            inference_result:  InferenceResult from Module 16 (for experiment metadata).
            evaluation_result: EvaluationResult from Module 15 (for experiment metadata).
            total_duration_s:  Combined duration of upstream modules.

        Returns:
            Dict with keys: experiment_manager, experiment, report_generator,
            exporter, artifact_manager, manifest_manager, config.
        """
        from src.reporting.report import ReportGenerator

        # Step 1: ExperimentManager builds ExperimentMetadata.
        exp_mgr    = ExperimentManager(config)
        experiment = exp_mgr.build(
            inference_result  = inference_result,
            evaluation_result = evaluation_result,
            total_duration_s  = total_duration_s,
        )

        # Step 2: ReportGenerator for content assembly.
        report_gen = ReportGenerator(config, experiment)

        # Step 3: ReportExporter for file writing.
        exporter   = ReportExporter(config)

        # Step 4: ArtifactManager.
        artifact_mgr = ArtifactManager(experiment_id=experiment.experiment_id)

        # Step 5: ManifestManager.
        manifest_mgr = ManifestManager(config)

        _LOGGER.debug(
            "ReportFactory: built context for experiment '%s'.",
            experiment.experiment_id,
        )

        return {
            "experiment_manager": exp_mgr,
            "experiment":         experiment,
            "report_generator":   report_gen,
            "exporter":           exporter,
            "artifact_manager":   artifact_mgr,
            "manifest_manager":   manifest_mgr,
            "config":             config,
        }