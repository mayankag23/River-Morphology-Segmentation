"""
Manifest management for Module 19.

ManifestManager builds a ReportManifest from all registered ArtifactRecords
and writes it to disk as a JSON file. The manifest is machine-readable and
serves as the authoritative index of every file produced in a report run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reporting.contracts import ArtifactRecord, ReportManifest, ReportingConfig

__all__ = ["ManifestManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_MANIFEST_VERSION: str = "1.0"


class ManifestManager:
    """
    Builds and writes the ReportManifest.

    Args:
        config: ReportingConfig.
    """

    def __init__(self, config: ReportingConfig) -> None:
        self._config = config

    def build(
        self,
        experiment_id: str,
        artifacts:     tuple[ArtifactRecord, ...],
        report_files:  tuple[str, ...],
    ) -> ReportManifest:
        """
        Build an immutable ReportManifest.

        Args:
            experiment_id: Experiment identifier.
            artifacts:     All registered ArtifactRecord objects.
            report_files:  Paths to the generated report files.

        Returns:
            Frozen ReportManifest.
        """
        timestamp    = datetime.now(timezone.utc).isoformat()
        total_size   = sum(a.size_bytes for a in artifacts if a.size_bytes >= 0)

        return ReportManifest(
            manifest_version = _MANIFEST_VERSION,
            experiment_id    = experiment_id,
            report_timestamp = timestamp,
            artifacts        = artifacts,
            num_artifacts    = len(artifacts),
            report_files     = report_files,
            total_size_bytes = total_size,
        )

    def write(self, manifest: ReportManifest, output_dir: str) -> str:
        """
        Write the manifest as a JSON file.

        Args:
            manifest:   ReportManifest to serialise.
            output_dir: Directory to write the manifest file.

        Returns:
            Absolute path of the written manifest file.
        """
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{manifest.experiment_id}_manifest.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(manifest.as_dict(), fh, indent=2, ensure_ascii=True)
            _LOGGER.info("ManifestManager: wrote manifest -> %s", path.name)
        except Exception as exc:
            _LOGGER.warning("ManifestManager: failed to write manifest: %s", exc)
        return str(path)