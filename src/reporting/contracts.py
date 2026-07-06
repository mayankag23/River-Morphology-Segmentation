"""
Public data contracts for the Reporting & Experiment Management Framework (Module 19).

Contract chain:
    EvaluationResult      (Module 15) ──┐
    InferenceResult       (Module 16) ──┤──> ReportEngine.generate() ──> ReportResult
    RiverMorphologyResult (Module 17) ──┤
    VisualizationResult   (Module 18) ──┘

ReportResult is the immutable public output, consumable by external tools,
dashboards, CI/CD pipelines, and archive systems.

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- All fields have stable JSON-serializable types.
- No module-level imports of upstream types (only used inside methods).
- ReportResult carries complete provenance: all input result summaries,
  all generated artifact paths, and the experiment configuration snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ReportingConfig",
    "ArtifactRecord",
    "ExperimentMetadata",
    "ReportManifest",
    "ReportResult",
]

_REPORT_VERSION: str = "1.0"


# ==============================================================================
# ReportingConfig
# ==============================================================================

@dataclass(frozen=True)
class ReportingConfig:
    """
    Immutable reporting configuration.

    Attributes:
        output_dir:         Root directory for all report artifacts.
        report_name:        Base filename prefix for all reports.
        export_markdown:    Generate a .md summary report.
        export_json:        Generate a .json metrics summary.
        export_csv:         Generate a .csv per-class metrics table.
        export_pdf:         Generate a .pdf report (requires weasyprint or reportlab).
        include_figures:    Copy or reference VisualizationResult figures in report.
        include_config:     Include the full configuration snapshot in the report.
        include_evaluation: Include EvaluationResult metrics section.
        include_inference:  Include InferenceResult section.
        include_morphology: Include RiverMorphologyResult section.
        include_visualization: Include VisualizationResult figure list.
        project_name:       Human-readable project name.
        project_version:    Project/model version string.
        author:             Report author name or team.
        institution:        Institution or organisation name.
        report_version:     Report schema version.
        git_commit:         Git commit hash (populated by ExperimentManager).
        experiment_id:      Unique experiment identifier.
    """

    output_dir:             str   = "reports"
    report_name:            str   = "river_morphology_report"
    export_markdown:        bool  = True
    export_json:            bool  = True
    export_csv:             bool  = True
    export_pdf:             bool  = False
    include_figures:        bool  = True
    include_config:         bool  = True
    include_evaluation:     bool  = True
    include_inference:      bool  = True
    include_morphology:     bool  = True
    include_visualization:  bool  = True
    project_name:           str   = "River Morphology Segmentation"
    project_version:        str   = "1.0.0"
    author:                 str   = ""
    institution:            str   = ""
    report_version:         str   = _REPORT_VERSION
    git_commit:             str   = ""
    experiment_id:          str   = ""

    @classmethod
    def from_config(cls, config: Any) -> ReportingConfig:
        """Build ReportingConfig from config.reporting."""
        rcfg = getattr(config, "reporting", None)
        if rcfg is None:
            return cls()
        return cls(
            output_dir            = str(getattr(rcfg,  "output_dir",            "reports")),
            report_name           = str(getattr(rcfg,  "report_name",           "river_morphology_report")),
            export_markdown       = bool(getattr(rcfg, "export_markdown",       True)),
            export_json           = bool(getattr(rcfg, "export_json",           True)),
            export_csv            = bool(getattr(rcfg, "export_csv",            True)),
            export_pdf            = bool(getattr(rcfg, "export_pdf",            False)),
            include_figures       = bool(getattr(rcfg, "include_figures",       True)),
            include_config        = bool(getattr(rcfg, "include_config",        True)),
            include_evaluation    = bool(getattr(rcfg, "include_evaluation",    True)),
            include_inference     = bool(getattr(rcfg, "include_inference",     True)),
            include_morphology    = bool(getattr(rcfg, "include_morphology",    True)),
            include_visualization = bool(getattr(rcfg, "include_visualization", True)),
            project_name          = str(getattr(rcfg,  "project_name",          "River Morphology Segmentation")),
            project_version       = str(getattr(rcfg,  "project_version",       "1.0.0")),
            author                = str(getattr(rcfg,  "author",                "")),
            institution           = str(getattr(rcfg,  "institution",           "")),
            report_version        = str(getattr(rcfg,  "report_version",        _REPORT_VERSION)),
            git_commit            = str(getattr(rcfg,  "git_commit",            "")),
            experiment_id         = str(getattr(rcfg,  "experiment_id",         "")),
        )


# ==============================================================================
# ArtifactRecord
# ==============================================================================

@dataclass(frozen=True)
class ArtifactRecord:
    """
    Immutable record for one managed artifact file.

    Attributes:
        artifact_id:   Unique identifier for this artifact.
        artifact_type: Category: "report", "figure", "checkpoint", "prediction",
                       "manifest", "csv", "json", "markdown", "pdf".
        path:          Absolute file path.
        size_bytes:    File size in bytes. -1 when not yet written.
        description:   Human-readable description.
        source_module: Module that produced this artifact (e.g. "Module 16").
        metadata:      Free-form additional metadata dict.
    """

    artifact_id:    str
    artifact_type:  str
    path:           str
    size_bytes:     int   = -1
    description:    str   = ""
    source_module:  str   = ""
    metadata:       dict  = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "artifact_id":   self.artifact_id,
            "artifact_type": self.artifact_type,
            "path":          self.path,
            "size_bytes":    self.size_bytes,
            "description":   self.description,
            "source_module": self.source_module,
            "metadata":      self.metadata,
        }


# ==============================================================================
# ExperimentMetadata
# ==============================================================================

@dataclass(frozen=True)
class ExperimentMetadata:
    """
    Immutable experiment provenance record.

    Attributes:
        experiment_id:      Unique experiment identifier (UUID or human label).
        run_timestamp:      ISO-8601 UTC timestamp when the run started.
        architecture:       Model architecture name.
        checkpoint_epoch:   Epoch of the loaded checkpoint.
        checkpoint_path:    Path to the loaded checkpoint.
        num_classes:        Number of segmentation classes.
        class_names:        Ordered class names.
        git_commit:         Git commit hash at run time. "" when not available.
        project_name:       Human-readable project name.
        project_version:    Project version string.
        author:             Author or team name.
        institution:        Institution name.
        total_duration_s:   Total wall-clock seconds from inference to report.
        config_snapshot:    Dict snapshot of key configuration values.
    """

    experiment_id:    str
    run_timestamp:    str
    architecture:     str
    checkpoint_epoch: int
    checkpoint_path:  str
    num_classes:      int
    class_names:      tuple[str, ...]
    git_commit:       str
    project_name:     str
    project_version:  str
    author:           str
    institution:      str
    total_duration_s: float
    config_snapshot:  dict

    def as_dict(self) -> dict:
        return {
            "experiment_id":    self.experiment_id,
            "run_timestamp":    self.run_timestamp,
            "architecture":     self.architecture,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_path":  self.checkpoint_path,
            "num_classes":      self.num_classes,
            "class_names":      list(self.class_names),
            "git_commit":       self.git_commit,
            "project_name":     self.project_name,
            "project_version":  self.project_version,
            "author":           self.author,
            "institution":      self.institution,
            "total_duration_s": round(self.total_duration_s, 3),
            "config_snapshot":  self.config_snapshot,
        }


# ==============================================================================
# ReportManifest
# ==============================================================================

@dataclass(frozen=True)
class ReportManifest:
    """
    Immutable manifest listing every artifact produced in this report run.

    Attributes:
        manifest_version: Schema version for the manifest format.
        experiment_id:    Experiment identifier this manifest belongs to.
        report_timestamp: ISO-8601 UTC timestamp of report generation.
        artifacts:        Tuple of all ArtifactRecord objects.
        num_artifacts:    Total artifact count.
        report_files:     Paths to the generated report files (md/json/csv/pdf).
        total_size_bytes: Sum of all artifact file sizes.
    """

    manifest_version: str
    experiment_id:    str
    report_timestamp: str
    artifacts:        tuple[ArtifactRecord, ...]
    num_artifacts:    int
    report_files:     tuple[str, ...]
    total_size_bytes: int

    def as_dict(self) -> dict:
        return {
            "manifest_version": self.manifest_version,
            "experiment_id":    self.experiment_id,
            "report_timestamp": self.report_timestamp,
            "num_artifacts":    self.num_artifacts,
            "total_size_bytes": self.total_size_bytes,
            "report_files":     list(self.report_files),
            "artifacts":        [a.as_dict() for a in self.artifacts],
        }


# ==============================================================================
# ReportResult
# ==============================================================================

@dataclass(frozen=True)
class ReportResult:
    """
    Immutable public output of ReportEngine.generate().

    Attributes:
        experiment:        ExperimentMetadata for this run.
        manifest:          ReportManifest listing all artifacts.
        report_files:      Absolute paths of generated report files.
        num_report_files:  Number of report files written.
        num_artifacts:     Total artifact count in the manifest.
        report_version:    Report schema version.
        operations_log:    Ordered log of report generation steps.
        generation_time_s: Wall-clock seconds for report generation.

        # Input summaries (extracted from upstream results for standalone use).
        evaluation_summary:   Dict of key evaluation metrics.
        inference_summary:    Dict of key inference statistics.
        morphology_summary:   Dict of key morphology statistics.
        visualization_summary: Dict of key visualization statistics.
    """

    experiment:            ExperimentMetadata
    manifest:              ReportManifest
    report_files:          tuple[str, ...]
    num_report_files:      int
    num_artifacts:         int
    report_version:        str
    operations_log:        tuple[str, ...]
    generation_time_s:     float
    evaluation_summary:    dict
    inference_summary:     dict
    morphology_summary:    dict
    visualization_summary: dict

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        return [
            f"  experiment_id:      {self.experiment.experiment_id}",
            f"  architecture:       {self.experiment.architecture}",
            f"  num_artifacts:      {self.num_artifacts}",
            f"  num_report_files:   {self.num_report_files}",
            f"  report_version:     {self.report_version}",
            f"  generation_time_s:  {self.generation_time_s:.2f}",
        ]

    def as_dict(self) -> dict:
        """Return a fully JSON-serializable dict."""
        return {
            "report_version":       self.report_version,
            "generation_time_s":    round(self.generation_time_s, 3),
            "num_report_files":     self.num_report_files,
            "num_artifacts":        self.num_artifacts,
            "report_files":         list(self.report_files),
            "operations_log":       list(self.operations_log),
            "experiment":           self.experiment.as_dict(),
            "manifest":             self.manifest.as_dict(),
            "evaluation_summary":   self.evaluation_summary,
            "inference_summary":    self.inference_summary,
            "morphology_summary":   self.morphology_summary,
            "visualization_summary": self.visualization_summary,
        }