"""
Public data contracts for the Pipeline Orchestration Framework (Module 20).

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- PipelineResult is the single immutable deliverable of the CLI run.
- Stage names match the module they invoke (e.g. "training", "evaluation").
- AOIConfig carries per-AOI overrides without changing the global config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VALID_MODES",
    "PipelineConfig",
    "AOIConfig",
    "StageResult",
    "PipelineResult",
]

# Canonical mode names supported by the pipeline.
VALID_MODES: tuple[str, ...] = (
    "training",
    "evaluation",
    "inference",
    "analysis",
    "visualization",
    "reporting",
    "full",
)


# ==============================================================================
# PipelineConfig
# ==============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable pipeline-level configuration.

    Attributes:
        mode:        Pipeline execution mode. One of VALID_MODES.
        run_id:      Unique run identifier. Auto-generated when empty.
        dry_run:     When True, validate config and print plan without executing.
        output_dir:  Root output directory for this run.
        aoi_ids:     AOI identifiers to process (single or multiple).
        device:      Compute device string.
        seed:        Global random seed.
        resume_from: Checkpoint path to resume from. "" = fresh start.
    """

    mode:        str               = "full"
    run_id:      str               = ""
    dry_run:     bool              = False
    output_dir:  str               = "outputs"
    aoi_ids:     tuple[str, ...]   = ()
    device:      str               = "cpu"
    seed:        int               = 42
    resume_from: str               = ""

    @classmethod
    def from_config(cls, config: Any, cli_overrides: dict | None = None) -> PipelineConfig:
        """
        Build PipelineConfig from the project config object and optional CLI overrides.

        Priority: CLI overrides > config.pipeline > defaults.
        """
        overrides = dict(cli_overrides or {})
        pcfg      = getattr(config, "pipeline", None)

        mode        = str(overrides.get("mode",
                          getattr(pcfg, "mode",        "full")        if pcfg else "full"))
        run_id      = str(overrides.get("run_id",
                          getattr(pcfg, "run_id",      "")            if pcfg else ""))
        dry_run     = bool(overrides.get("dry_run",
                           getattr(pcfg, "dry_run",    False)         if pcfg else False))
        output_dir  = str(overrides.get("output_dir",
                          getattr(pcfg, "output_dir",  "outputs")     if pcfg else "outputs"))
        device      = str(overrides.get("device",
                          getattr(config, "device",    "cpu")         if config else "cpu"))
        seed        = int(overrides.get("seed",
                          getattr(getattr(config, "reproducibility", None), "seed", 42)))
        resume_from = str(overrides.get("resume_from",
                          getattr(pcfg, "resume_from", "")            if pcfg else ""))

        # AOI IDs: CLI --aoi-ids list > config.pipeline.aoi_ids > single aoi.id
        raw_aoi_ids = overrides.get("aoi_ids", None)
        if raw_aoi_ids is None:
            raw_aoi_ids = getattr(pcfg, "aoi_ids", None) if pcfg else None
        if raw_aoi_ids is None:
            # Fall back to single AOI from config.aoi
            aoi_section = getattr(config, "aoi", None)
            aoi_id      = str(getattr(aoi_section, "id", "default")) if aoi_section else "default"
            raw_aoi_ids = [aoi_id]
        aoi_ids = tuple(str(a) for a in raw_aoi_ids)

        return cls(
            mode        = mode,
            run_id      = run_id,
            dry_run     = dry_run,
            output_dir  = output_dir,
            aoi_ids     = aoi_ids,
            device      = device,
            seed        = seed,
            resume_from = resume_from,
        )


# ==============================================================================
# AOIConfig
# ==============================================================================

@dataclass(frozen=True)
class AOIConfig:
    """
    Immutable per-AOI specification.

    Attributes:
        aoi_id:    Unique AOI identifier (used to namespace outputs).
        min_lon:   Western boundary (decimal degrees).
        min_lat:   Southern boundary (decimal degrees).
        max_lon:   Eastern boundary (decimal degrees).
        max_lat:   Northern boundary (decimal degrees).
        start_date: YYYY-MM-DD start date for imagery compositing.
        end_date:   YYYY-MM-DD end date for imagery compositing.
        output_dir: Per-AOI output directory (resolved by orchestrator).
    """

    aoi_id:     str
    min_lon:    float | None = None
    min_lat:    float | None = None
    max_lon:    float | None = None
    max_lat:    float | None = None
    start_date: str          = ""
    end_date:   str          = ""
    output_dir: str          = ""

    @property
    def is_complete(self) -> bool:
        """True when all four coordinate bounds are set."""
        return all(v is not None for v in (self.min_lon, self.min_lat,
                                           self.max_lon, self.max_lat))

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """Return (min_lon, min_lat, max_lon, max_lat) or None."""
        if not self.is_complete:
            return None
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)  # type: ignore[return-value]

    def as_dict(self) -> dict:
        return {
            "aoi_id":     self.aoi_id,
            "min_lon":    self.min_lon,
            "min_lat":    self.min_lat,
            "max_lon":    self.max_lon,
            "max_lat":    self.max_lat,
            "start_date": self.start_date,
            "end_date":   self.end_date,
            "output_dir": self.output_dir,
        }


# ==============================================================================
# StageResult
# ==============================================================================

@dataclass(frozen=True)
class StageResult:
    """
    Immutable result of one pipeline stage execution.

    Attributes:
        stage:        Stage name (e.g. "training", "evaluation").
        aoi_id:       AOI this stage was executed for.
        success:      True when the stage completed without error.
        skipped:      True when the stage was deliberately skipped.
        duration_s:   Wall-clock execution time in seconds.
        artifacts:    Tuple of file paths produced.
        error:        Error message if failed; "" otherwise.
        notes:        Additional log notes.
    """

    stage:      str
    aoi_id:     str
    success:    bool
    skipped:    bool              = False
    duration_s: float             = 0.0
    artifacts:  tuple[str, ...]   = ()
    error:      str               = ""
    notes:      str               = ""

    def as_dict(self) -> dict:
        return {
            "stage":      self.stage,
            "aoi_id":     self.aoi_id,
            "success":    self.success,
            "skipped":    self.skipped,
            "duration_s": round(self.duration_s, 3),
            "artifacts":  list(self.artifacts),
            "error":      self.error,
            "notes":      self.notes,
        }


# ==============================================================================
# PipelineResult
# ==============================================================================

@dataclass(frozen=True)
class PipelineResult:
    """
    Immutable public output of a complete pipeline run.

    Attributes:
        run_id:           Unique run identifier.
        mode:             Pipeline mode executed.
        aoi_ids:          Tuple of AOI IDs processed.
        stage_results:    Tuple of StageResult, one per stage per AOI.
        success:          True when all non-skipped stages succeeded.
        total_duration_s: Total wall-clock time in seconds.
        output_dirs:      Dict aoi_id -> output directory path.
        warnings:         Validation warnings encountered.
        operations_log:   Ordered log of orchestration steps.
        dry_run:          True when this was a dry-run (no stages executed).
        num_stages:       Total stages executed (not skipped).
        num_failed:       Number of failed stages.
    """

    run_id:           str
    mode:             str
    aoi_ids:          tuple[str, ...]
    stage_results:    tuple[StageResult, ...]
    success:          bool
    total_duration_s: float
    output_dirs:      dict[str, str]
    warnings:         tuple[str, ...]
    operations_log:   tuple[str, ...]
    dry_run:          bool
    num_stages:       int
    num_failed:       int

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        status = "SUCCESS" if self.success else "FAILED"
        return [
            f"  run_id:           {self.run_id}",
            f"  mode:             {self.mode}",
            f"  status:           {status}",
            f"  aoi_ids:          {', '.join(self.aoi_ids)}",
            f"  num_stages:       {self.num_stages}",
            f"  num_failed:       {self.num_failed}",
            f"  dry_run:          {self.dry_run}",
            f"  total_duration_s: {self.total_duration_s:.2f}",
        ]

    def as_dict(self) -> dict:
        """Return a JSON-serializable summary dict."""
        return {
            "run_id":           self.run_id,
            "mode":             self.mode,
            "aoi_ids":          list(self.aoi_ids),
            "success":          self.success,
            "dry_run":          self.dry_run,
            "total_duration_s": round(self.total_duration_s, 3),
            "num_stages":       self.num_stages,
            "num_failed":       self.num_failed,
            "output_dirs":      self.output_dirs,
            "warnings":         list(self.warnings),
            "operations_log":   list(self.operations_log),
            "stage_results":    [s.as_dict() for s in self.stage_results],
        }

    def failed_stages(self) -> list[StageResult]:
        """Return all failed StageResults."""
        return [s for s in self.stage_results if not s.success and not s.skipped]

    def stages_for_aoi(self, aoi_id: str) -> list[StageResult]:
        """Return all StageResults for a specific AOI."""
        return [s for s in self.stage_results if s.aoi_id == aoi_id]
