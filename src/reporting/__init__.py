"""
src/reporting -- Reporting & Experiment Management Framework (Module 19).

Single public entry point:
    ReportEngine.generate(...) -> ReportResult

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
        evaluation_result    = eval_result,    # Module 15
        inference_result     = inf_result,     # Module 16
        morphology_result    = morph_result,   # Module 17
        visualization_result = viz_result,     # Module 18
    )

    print(result.experiment.experiment_id)
    for path in result.report_files:
        print(path)
"""

# Primary entry point
from src.reporting.engine import ReportEngine

# Public contracts
from src.reporting.contracts import (
    ArtifactRecord,
    ExperimentMetadata,
    ReportManifest,
    ReportingConfig,
    ReportResult,
)

# Components (for advanced / testing use)
from src.reporting.artifact import ArtifactManager
from src.reporting.experiment import ExperimentManager
from src.reporting.exporter import ReportExporter
from src.reporting.factory import ReportFactory
from src.reporting.manifest import ManifestManager
from src.reporting.report import ReportGenerator
from src.reporting.validator import ReportValidator, ReportValidationResult

__all__ = [
    # Primary
    "ReportEngine",
    # Contracts
    "ReportingConfig",
    "ArtifactRecord",
    "ExperimentMetadata",
    "ReportManifest",
    "ReportResult",
    # Components
    "ArtifactManager",
    "ExperimentManager",
    "ReportExporter",
    "ReportFactory",
    "ManifestManager",
    "ReportGenerator",
    "ReportValidator",
    "ReportValidationResult",
]