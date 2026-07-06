"""
Experiment management for Module 19.

ExperimentManager assembles ExperimentMetadata from upstream results and
the ReportingConfig. It captures git commit hash, timestamps, architecture,
checkpoint information, and a configuration snapshot.

Design rules
------------
- Git commit is read from the environment or subprocess; falls back to "".
- All fields are derived at build-time from immutable upstream contracts.
- No files are written; ExperimentManager only assembles metadata.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

from src.reporting.contracts import ExperimentMetadata, ReportingConfig

__all__ = ["ExperimentManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Assembles ExperimentMetadata from upstream results and config.

    Args:
        config: ReportingConfig.
    """

    def __init__(self, config: ReportingConfig) -> None:
        self._config = config

    def build(
        self,
        inference_result:  Any | None = None,
        evaluation_result: Any | None = None,
        total_duration_s:  float = 0.0,
    ) -> ExperimentMetadata:
        """
        Build ExperimentMetadata from available upstream results.

        Args:
            inference_result:  InferenceResult from Module 16.
            evaluation_result: EvaluationResult from Module 15.
            total_duration_s:  Combined duration of upstream modules.

        Returns:
            Frozen ExperimentMetadata.
        """
        experiment_id = self._config.experiment_id or _generate_id()
        timestamp     = datetime.now(timezone.utc).isoformat()
        git_commit    = self._config.git_commit or _read_git_commit()

        # Architecture and checkpoint from InferenceResult.
        architecture    = ""
        checkpoint_epoch = 0
        checkpoint_path  = ""
        num_classes      = 0
        class_names: tuple[str, ...] = ()

        if inference_result is not None:
            architecture  = str(getattr(inference_result, "architecture", ""))
            class_names   = tuple(getattr(inference_result, "class_names", ()))
            num_classes   = int(getattr(inference_result, "num_classes",   0))
            ckpt_meta     = getattr(inference_result, "checkpoint_meta", None)
            if ckpt_meta is not None:
                checkpoint_epoch = int(getattr(ckpt_meta, "epoch", 0))
                checkpoint_path  = str(getattr(ckpt_meta, "checkpoint_path", ""))

        if not architecture and evaluation_result is not None:
            architecture = str(getattr(evaluation_result, "architecture", ""))

        config_snapshot = self._build_config_snapshot(inference_result, evaluation_result)

        return ExperimentMetadata(
            experiment_id    = experiment_id,
            run_timestamp    = timestamp,
            architecture     = architecture,
            checkpoint_epoch = checkpoint_epoch,
            checkpoint_path  = checkpoint_path,
            num_classes      = num_classes,
            class_names      = class_names,
            git_commit       = git_commit,
            project_name     = self._config.project_name,
            project_version  = self._config.project_version,
            author           = self._config.author,
            institution      = self._config.institution,
            total_duration_s = total_duration_s,
            config_snapshot  = config_snapshot,
        )

    def _build_config_snapshot(
        self,
        inference_result:  Any | None,
        evaluation_result: Any | None,
    ) -> dict:
        """Build a JSON-serializable config snapshot from available context."""
        snapshot: dict = {
            "project_name":    self._config.project_name,
            "project_version": self._config.project_version,
            "report_version":  self._config.report_version,
            "export_markdown": self._config.export_markdown,
            "export_json":     self._config.export_json,
            "export_csv":      self._config.export_csv,
            "export_pdf":      self._config.export_pdf,
        }
        if inference_result is not None:
            inf_cfg = getattr(inference_result, "inference_config", None)
            if inf_cfg is not None:
                snapshot["inference"] = {
                    "device":              getattr(inf_cfg, "device",              ""),
                    "batch_size":          getattr(inf_cfg, "batch_size",          0),
                    "probability_mode":    getattr(inf_cfg, "probability_mode",    ""),
                    "confidence_strategy": getattr(inf_cfg, "confidence_strategy", ""),
                    "checkpoint_strategy": getattr(inf_cfg, "checkpoint_strategy", ""),
                }
        if evaluation_result is not None:
            snapshot["evaluation"] = {
                "split":        getattr(evaluation_result, "split",        ""),
                "ignore_index": getattr(evaluation_result, "ignore_index", 255),
            }
        return snapshot


def _generate_id() -> str:
    """Generate a short UUID-based experiment ID."""
    return f"exp_{uuid.uuid4().hex[:8]}"


def _read_git_commit() -> str:
    """Read the current git commit hash. Returns '' on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""