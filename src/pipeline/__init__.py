"""
src/pipeline -- End-to-End Pipeline Orchestration & CLI Framework (Module 20).

Single public entry point (programmatic):
    PipelineOrchestrator(config, pipeline_config).run() -> PipelineResult

CLI entry point:
    python main.py [--mode MODE] [--aoi-ids ID...] [--dry-run] [--help]

Usage (programmatic)
--------------------
    from src.pipeline import PipelineOrchestrator, PipelineConfig

    pipeline_config = PipelineConfig(mode="inference", dry_run=False)
    orchestrator    = PipelineOrchestrator(config, pipeline_config)
    result          = orchestrator.run()

    print(result.success, result.num_failed)
    for sr in result.failed_stages():
        print(sr.stage, sr.error)
"""

from src.pipeline.contracts import (
    AOIConfig,
    PipelineConfig,
    PipelineResult,
    StageResult,
    VALID_MODES,
)
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.validator import PipelineValidator, PipelineValidationResult
from src.pipeline.factory import PipelineFactory
from src.pipeline.runner import StageRunner
from src.pipeline.cli import build_parser, parse_args, run_cli

__all__ = [
    # Primary
    "PipelineOrchestrator",
    # Contracts
    "PipelineConfig",
    "AOIConfig",
    "StageResult",
    "PipelineResult",
    "VALID_MODES",
    # Support
    "PipelineValidator",
    "PipelineValidationResult",
    "PipelineFactory",
    "StageRunner",
    # CLI
    "build_parser",
    "parse_args",
    "run_cli",
]
