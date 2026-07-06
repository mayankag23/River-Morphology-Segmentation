"""
Artifact management for Module 19.

ArtifactManager registers every file produced by Modules 15-18 (and by the
report generator itself) into an ordered catalog of ArtifactRecord objects.
It is intentionally stateful during report generation; the final snapshot is
frozen into ReportManifest once generation completes.

Design rules
------------
- File sizes are read from disk when the file exists.
- Files that do not yet exist get size_bytes = -1.
- No file is read or written here; ArtifactManager only tracks paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.reporting.contracts import ArtifactRecord

__all__ = ["ArtifactManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    Registers and inventories all artifact files for one report run.

    Usage:
        mgr = ArtifactManager(experiment_id="exp_001")
        mgr.register(path="/out/eval.json", artifact_type="json",
                     description="Evaluation metrics", source_module="Module 15")
        records = mgr.all_records()
    """

    def __init__(self, experiment_id: str = "") -> None:
        self._experiment_id = experiment_id
        self._records: list[ArtifactRecord] = []
        self._counter: int = 0

    def register(
        self,
        path:          str,
        artifact_type: str,
        description:   str   = "",
        source_module: str   = "",
        metadata:      dict  | None = None,
    ) -> ArtifactRecord:
        """
        Register one artifact file.

        Args:
            path:          Absolute or relative path to the artifact file.
            artifact_type: Category string: "report", "figure", "json", "csv",
                           "markdown", "pdf", "checkpoint", "prediction", "manifest".
            description:   Human-readable description.
            source_module: Module that produced this artifact.
            metadata:      Optional additional metadata dict.

        Returns:
            Frozen ArtifactRecord.
        """
        self._counter += 1
        artifact_id = f"{self._experiment_id}_{artifact_type}_{self._counter:04d}"
        size        = _file_size(path)

        record = ArtifactRecord(
            artifact_id    = artifact_id,
            artifact_type  = artifact_type,
            path           = str(path),
            size_bytes     = size,
            description    = description,
            source_module  = source_module,
            metadata       = dict(metadata or {}),
        )
        self._records.append(record)
        _LOGGER.debug("ArtifactManager: registered %s -> %s", artifact_type, path)
        return record

    def register_from_visualization_result(self, viz_result: Any) -> None:
        """
        Register all exported figure paths from a VisualizationResult.

        Args:
            viz_result: VisualizationResult from Module 18.
        """
        for spec in getattr(viz_result, "figures", ()):
            for path in getattr(spec, "export_paths", []):
                ext = Path(path).suffix.lstrip(".").lower()
                self.register(
                    path          = path,
                    artifact_type = "figure",
                    description   = f"Figure: {getattr(spec, 'figure_id', '')}",
                    source_module = "Module 18",
                    metadata      = {
                        "figure_type":      getattr(spec, "figure_type", ""),
                        "figure_id":        getattr(spec, "figure_id",   ""),
                        "sample_id":        getattr(spec, "sample_id",   ""),
                        "acquisition_date": getattr(spec, "acquisition_date", ""),
                    },
                )

    def register_from_inference_result(self, inf_result: Any) -> None:
        """
        Register all exported prediction paths from an InferenceResult.

        Args:
            inf_result: InferenceResult from Module 16.
        """
        for pred in getattr(inf_result, "predictions", ()):
            for path in getattr(pred, "exported_paths", []):
                self.register(
                    path          = path,
                    artifact_type = "prediction",
                    description   = f"Prediction: {getattr(pred, 'sample_id', '')}",
                    source_module = "Module 16",
                )

    def all_records(self) -> tuple[ArtifactRecord, ...]:
        """Return all registered ArtifactRecords as an immutable tuple."""
        return tuple(self._records)

    def by_type(self, artifact_type: str) -> list[ArtifactRecord]:
        """Return all records of a specific artifact_type."""
        return [r for r in self._records if r.artifact_type == artifact_type]

    def total_size_bytes(self) -> int:
        """Sum of all known file sizes (excludes size_bytes == -1)."""
        return sum(r.size_bytes for r in self._records if r.size_bytes >= 0)

    def num_artifacts(self) -> int:
        """Total number of registered artifacts."""
        return len(self._records)


def _file_size(path: str) -> int:
    """Return file size in bytes, or -1 when the file does not exist."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return -1