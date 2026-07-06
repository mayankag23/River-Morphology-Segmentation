"""
ReportEngine -- the single public interface for Module 19.

Usage
-----
    from src.reporting import ReportEngine, ReportingConfig

    config = ReportingConfig(
        output_dir      = "reports/exp_001",
        report_name     = "kosi_river_report",
        export_markdown = True,
        export_json     = True,
        export_csv      = True,
        project_name    = "Kosi River Morphology",
        author          = "Research Team",
    )
    engine = ReportEngine(config)
    result = engine.generate(
        evaluation_result    = eval_result,
        inference_result     = inf_result,
        morphology_result    = morph_result,
        visualization_result = viz_result,
    )

    print(result.experiment.experiment_id)
    print(result.report_files)

ReportEngine orchestrates:
    ReportValidator     -> pre-flight checks
    ReportFactory       -> builds all components
    ExperimentManager   -> assembles ExperimentMetadata
    ArtifactManager     -> registers all artifact files
    ReportGenerator     -> assembles Markdown / JSON / CSV content
    ReportExporter      -> writes report files to disk
    ManifestManager     -> builds and writes ReportManifest
    ReportResult        -> immutable public output
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.reporting.contracts import ReportResult, ReportingConfig
from src.reporting.factory import ReportFactory
from src.reporting.report import (
    _extract_evaluation_summary,
    _extract_inference_summary,
    _extract_morphology_summary,
    _extract_visualization_summary,
)
from src.reporting.validator import ReportValidator

__all__ = ["ReportEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ReportEngine:
    """
    Orchestrates a complete report generation run.

    Args:
        config: ReportingConfig or project Config object.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, ReportingConfig):
            self._config = config
        else:
            self._config = ReportingConfig.from_config(config)
        self._validator = ReportValidator()

    def generate(
        self,
        evaluation_result:    Any | None = None,
        inference_result:     Any | None = None,
        morphology_result:    Any | None = None,
        visualization_result: Any | None = None,
        total_duration_s:     float = 0.0,
    ) -> ReportResult:
        """
        Generate a complete report from upstream results.

        Args:
            evaluation_result:    EvaluationResult from Module 15.
            inference_result:     InferenceResult from Module 16.
            morphology_result:    RiverMorphologyResult from Module 17.
            visualization_result: VisualizationResult from Module 18.
            total_duration_s:     Total pipeline duration for provenance.

        Returns:
            Frozen ReportResult with all report files and provenance.
        """
        ops: list[str] = []
        t0             = time.perf_counter()

        # Step 1: Pre-flight validation.
        validation = self._validator.validate(
            self._config,
            evaluation_result    = evaluation_result,
            inference_result     = inference_result,
            morphology_result    = morphology_result,
            visualization_result = visualization_result,
        )
        for issue in validation.issues:
            _LOGGER.warning("ReportEngine pre-flight: %s", issue)
        ops.append(f"validation: {len(validation.issues)} issue(s)")

        # Step 2: Build all components.
        context      = ReportFactory.build(
            config            = self._config,
            inference_result  = inference_result,
            evaluation_result = evaluation_result,
            total_duration_s  = total_duration_s,
        )
        experiment   = context["experiment"]
        report_gen   = context["report_generator"]
        exporter     = context["exporter"]
        artifact_mgr = context["artifact_manager"]
        manifest_mgr = context["manifest_manager"]
        ops.append(f"experiment: {experiment.experiment_id}")

        # Step 3: Register upstream artifacts.
        if visualization_result is not None:
            artifact_mgr.register_from_visualization_result(visualization_result)
        if inference_result is not None:
            artifact_mgr.register_from_inference_result(inference_result)
        ops.append(f"artifacts registered: {artifact_mgr.num_artifacts()}")

        # Step 4: Generate report content.
        markdown_content = report_gen.build_markdown(
            evaluation_result    = evaluation_result,
            inference_result     = inference_result,
            morphology_result    = morphology_result,
            visualization_result = visualization_result,
        )
        json_summary = report_gen.build_json_summary(
            evaluation_result    = evaluation_result,
            inference_result     = inference_result,
            morphology_result    = morphology_result,
            visualization_result = visualization_result,
        )
        csv_rows = report_gen.build_csv_rows(
            evaluation_result = evaluation_result,
            morphology_result = morphology_result,
        )
        ops.append("content: built markdown, json, csv")

        # Step 5: Export report files.
        report_file_paths: list[str] = []

        md_path = exporter.export_markdown(markdown_content)
        if md_path:
            report_file_paths.append(md_path)
            artifact_mgr.register(md_path, "markdown", "Markdown report", "Module 19")

        json_path = exporter.export_json(json_summary)
        if json_path:
            report_file_paths.append(json_path)
            artifact_mgr.register(json_path, "json", "JSON metrics summary", "Module 19")

        csv_path = exporter.export_csv(csv_rows)
        if csv_path:
            report_file_paths.append(csv_path)
            artifact_mgr.register(csv_path, "csv", "Per-class metrics CSV", "Module 19")

        pdf_path = exporter.export_pdf(markdown_content)
        if pdf_path:
            report_file_paths.append(pdf_path)
            artifact_mgr.register(pdf_path, "pdf", "PDF report", "Module 19")

        ops.append(f"exported: {len(report_file_paths)} report files")

        # Step 6: Build and write manifest.
        manifest = manifest_mgr.build(
            experiment_id = experiment.experiment_id,
            artifacts     = artifact_mgr.all_records(),
            report_files  = tuple(report_file_paths),
        )
        manifest_path = manifest_mgr.write(manifest, self._config.output_dir)
        if manifest_path:
            artifact_mgr.register(manifest_path, "manifest",
                                   "Report manifest", "Module 19")
        ops.append(f"manifest: {manifest_path}")

        # Step 7: Extract input summaries for the ReportResult.
        eval_summary  = (_extract_evaluation_summary(evaluation_result)
                         if evaluation_result else {})
        inf_summary   = (_extract_inference_summary(inference_result)
                         if inference_result else {})
        morph_summary = (_extract_morphology_summary(morphology_result)
                         if morphology_result else {})
        viz_summary   = (_extract_visualization_summary(visualization_result)
                         if visualization_result else {})

        elapsed = time.perf_counter() - t0
        ops.append(f"total_time: {elapsed:.3f}s")

        # Step 8: Assemble ReportResult.
        result = ReportResult(
            experiment             = experiment,
            manifest               = manifest,
            report_files           = tuple(report_file_paths),
            num_report_files       = len(report_file_paths),
            num_artifacts          = artifact_mgr.num_artifacts(),
            report_version         = self._config.report_version,
            operations_log         = tuple(ops),
            generation_time_s      = elapsed,
            evaluation_summary     = eval_summary,
            inference_summary      = inf_summary,
            morphology_summary     = morph_summary,
            visualization_summary  = viz_summary,
        )

        for line in result.summary_lines():
            _LOGGER.info(line)

        return result