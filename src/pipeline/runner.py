"""
Stage runner for Module 20.

StageRunner executes exactly one pipeline stage (e.g. "training", "evaluation"),
captures its duration, output artifacts, and any exceptions, and returns a
frozen StageResult. It never implements stage logic — it only wraps callables
provided by the PipelineOrchestrator.

Design rules
------------
- Each stage is a zero-argument callable returning an optional artifacts tuple.
- Exceptions are caught and recorded; the pipeline decides whether to abort.
- In dry_run mode the callable is never invoked; a skipped StageResult is returned.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable

from src.pipeline.contracts import StageResult

__all__ = ["StageRunner"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class StageRunner:
    """
    Executes a single pipeline stage and returns a StageResult.

    Args:
        dry_run: When True, skip all stage execution.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run

    def run(
        self,
        stage:        str,
        aoi_id:       str,
        fn:           Callable[[], Any],
        skip_when:    bool = False,
        skip_reason:  str  = "",
    ) -> StageResult:
        """
        Execute one pipeline stage.

        Args:
            stage:       Stage name (e.g. "training").
            aoi_id:      AOI being processed.
            fn:          Zero-argument callable implementing the stage.
                         It may return a list/tuple of artifact paths, or None.
            skip_when:   When True, skip this stage (e.g. disabled in config).
            skip_reason: Human-readable reason for skipping.

        Returns:
            Frozen StageResult.
        """
        # Dry-run or explicitly skipped.
        if self._dry_run or skip_when:
            reason = "dry-run" if self._dry_run else skip_reason
            _LOGGER.info("SKIP  [%s] %s — %s", aoi_id, stage, reason)
            return StageResult(
                stage    = stage,
                aoi_id   = aoi_id,
                success  = True,
                skipped  = True,
                notes    = reason,
            )

        _LOGGER.info("START [%s] %s", aoi_id, stage)
        t0 = time.perf_counter()

        try:
            raw = fn()
            elapsed = time.perf_counter() - t0

            # Normalize artifacts return value.
            if isinstance(raw, (list, tuple)):
                artifacts = tuple(str(p) for p in raw if p)
            else:
                artifacts = ()

            _LOGGER.info(
                "DONE  [%s] %s in %.2fs (artifacts: %d)",
                aoi_id, stage, elapsed, len(artifacts),
            )
            return StageResult(
                stage      = stage,
                aoi_id     = aoi_id,
                success    = True,
                duration_s = elapsed,
                artifacts  = artifacts,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            tb      = traceback.format_exc()
            _LOGGER.error(
                "FAIL  [%s] %s after %.2fs: %s\n%s",
                aoi_id, stage, elapsed, exc, tb,
            )
            return StageResult(
                stage      = stage,
                aoi_id     = aoi_id,
                success    = False,
                duration_s = elapsed,
                error      = f"{type(exc).__name__}: {exc}",
                notes      = tb,
            )
