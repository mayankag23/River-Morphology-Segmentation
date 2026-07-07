"""
CLI interface for Module 20.

Provides a clean argparse-based CLI that:
- loads config.yaml
- accepts CLI overrides for all key pipeline parameters
- delegates to PipelineOrchestrator
- exits with code 0 on success, 1 on failure

Usage examples
--------------
    # Full pipeline, single AOI from config.yaml:
    python main.py

    # Inference-only, specific checkpoint:
    python main.py --mode inference --resume-from checkpoints/best.pt

    # Dry-run validation:
    python main.py --dry-run

    # Multiple AOIs:
    python main.py --mode full --aoi-ids kosi brahmaputra

    # Custom output directory:
    python main.py --mode reporting --output-dir /data/runs/exp42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.pipeline.contracts import PipelineConfig, PipelineResult, VALID_MODES

__all__ = ["build_parser", "parse_args", "run_cli"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog        = "river-pipeline",
        description = (
            "River Morphology Segmentation — End-to-End Pipeline\n\n"
            "Connects Modules 1-19 into a single executable application.\n"
            "Configuration is driven by config.yaml with optional CLI overrides."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # Core execution control.
    parser.add_argument(
        "--config", "-c",
        default = "config.yaml",
        metavar = "PATH",
        help    = "Path to the config.yaml file (default: config.yaml).",
    )
    parser.add_argument(
        "--mode", "-m",
        choices = list(VALID_MODES),
        default = None,
        help    = (
            "Pipeline mode to run. Overrides config.pipeline.mode. "
            f"Options: {', '.join(VALID_MODES)} (default: from config)."
        ),
    )
    parser.add_argument(
        "--aoi-ids", "-a",
        nargs   = "+",
        default = None,
        metavar = "AOI_ID",
        help    = "One or more AOI identifiers to process. Overrides config.pipeline.aoi_ids.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default = None,
        metavar = "DIR",
        help    = "Root output directory. Overrides config.pipeline.output_dir.",
    )
    parser.add_argument(
        "--resume-from", "-r",
        default = None,
        metavar = "PATH",
        help    = "Checkpoint path to resume from. Overrides config.pipeline.resume_from.",
    )
    parser.add_argument(
        "--run-id",
        default = None,
        metavar = "ID",
        help    = "Explicit run identifier. Auto-generated when not set.",
    )
    parser.add_argument(
        "--device",
        default = None,
        metavar = "DEVICE",
        help    = "Compute device: cpu, cuda, cuda:0, etc. Overrides config.device.",
    )

    # Flags.
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        default = False,
        help    = "Validate configuration and print execution plan without running stages.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action  = "store_true",
        default = False,
        help    = "Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--json-summary",
        action  = "store_true",
        default = False,
        help    = "Print JSON summary of PipelineResult to stdout on completion.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def load_config(config_path: str) -> Any:
    """
    Load and return the project Config object from a YAML file.

    Falls back to a SimpleNamespace stub when the project Config loader
    is not installed (e.g. in test environments).
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Run from the project root, or pass --config /path/to/config.yaml"
        )
    try:
        # Attempt to use the project's own Config loader (Module 2).
        from src.core.config import Config
        return Config(str(path))
    except ImportError:
        pass

    # Fallback: parse YAML and wrap in a SimpleNamespace tree.
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from None

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _dict_to_namespace(raw or {})


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert a dict to a SimpleNamespace for attribute access."""
    import types
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def _configure_logging(verbose: bool, config: Any) -> None:
    """Configure Python logging from config and --verbose flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    # Silence very verbose third-party loggers.
    for noisy in ("urllib3", "botocore", "boto3", "google", "earthengine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_cli(argv: list[str] | None = None) -> int:
    """
    Parse arguments, load config, run the pipeline, and return an exit code.

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args(argv)

    # ---- Logging ----
    _configure_logging(args.verbose, None)

    # ---- Load config ----
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ImportError) as exc:
        _LOGGER.error("%s", exc)
        return 1

    # ---- Build CLI overrides dict ----
    overrides: dict = {}
    if args.mode         is not None: overrides["mode"]        = args.mode
    if args.output_dir   is not None: overrides["output_dir"]  = args.output_dir
    if args.resume_from  is not None: overrides["resume_from"] = args.resume_from
    if args.run_id       is not None: overrides["run_id"]      = args.run_id
    if args.device       is not None: overrides["device"]      = args.device
    if args.aoi_ids      is not None: overrides["aoi_ids"]     = args.aoi_ids
    if args.dry_run:                  overrides["dry_run"]      = True

    # ---- Build PipelineConfig ----
    pipeline_config = PipelineConfig.from_config(config, overrides)

    _LOGGER.info(
        "river-pipeline | mode=%s | run_id=%s | dry_run=%s | aois=%s",
        pipeline_config.mode,
        pipeline_config.run_id or "(auto)",
        pipeline_config.dry_run,
        pipeline_config.aoi_ids,
    )

    # ---- Run ----
    try:
        from src.pipeline.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config, pipeline_config)
        result: PipelineResult = orchestrator.run()
    except Exception as exc:
        _LOGGER.exception("Pipeline aborted with unexpected error: %s", exc)
        return 1

    # ---- Print summary ----
    for line in result.summary_lines():
        _LOGGER.info(line)

    if args.json_summary:
        print(json.dumps(result.as_dict(), indent=2))

    if result.num_failed > 0:
        _LOGGER.error(
            "%d stage(s) failed. Check logs above for details.",
            result.num_failed,
        )
        for sr in result.failed_stages():
            _LOGGER.error("  FAILED: [%s] %s — %s", sr.aoi_id, sr.stage, sr.error)

    return 0 if result.success else 1
