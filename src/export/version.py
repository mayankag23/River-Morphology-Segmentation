"""
Dataset versioning for the River Morphology export pipeline.

VersionInfo records the lineage of a dataset package so that a consumer
of the data can determine exactly which pipeline version, feature schema,
and Landsat collection produced it.

version.json is written to the dataset root directory and updated on
every export run. It contains the same information regardless of how
many scenes are in the dataset.

DatasetVersionManager.generate() reads version strings from the Config
object (config.export.*) so they can be bumped in a single place.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.config import Config

__all__ = [
    "VersionInfo",
    "DatasetVersionManager",
    "VERSION_FILE_NAME",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

VERSION_FILE_NAME: str = "version.json"

# Fallback version constants when config.export is not present.
_DEFAULT_DATASET_VERSION:        str = "1.0.0"
_DEFAULT_PIPELINE_VERSION:       str = "1.0.0"
_DEFAULT_FEATURE_SCHEMA_VERSION: str = "1.0.0"
_DEFAULT_LANDSAT_COLLECTION:     str = "Landsat Collection 2 Level-2"


# ==============================================================================
# VersionInfo
# ==============================================================================

@dataclass(frozen=True)
class VersionInfo:
    """
    Immutable dataset version and lineage record.

    Attributes:
        dataset_version:        User-facing dataset release version.
        pipeline_version:       River Morphology pipeline version.
        feature_schema_version: Spectral feature schema version.
        landsat_collection:     GEE Landsat collection description.
        created_at:             ISO 8601 UTC creation timestamp.
        git_commit:             Short git commit hash, or None if unavailable.
        config_hash:            8-character SHA256 of config dict, or None.
    """

    dataset_version:        str
    pipeline_version:       str
    feature_schema_version: str
    landsat_collection:     str
    created_at:             str
    git_commit:             str | None
    config_hash:            str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plain dict."""
        return dataclasses.asdict(self)


# ==============================================================================
# DatasetVersionManager
# ==============================================================================

class DatasetVersionManager:
    """
    Generates and persists VersionInfo.

    Reads version strings from config.export.* with safe getattr fallbacks.
    Git commit and config hash are optional; failures are logged as debug
    messages rather than warnings to avoid noise in environments without git.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

    def generate(self) -> VersionInfo:
        """
        Build a VersionInfo from the current config and environment.

        Returns:
            Populated, immutable VersionInfo.
        """
        export_cfg = getattr(self._config, "export", None)

        dataset_version        = str(getattr(export_cfg, "dataset_version",        _DEFAULT_DATASET_VERSION))
        pipeline_version       = str(getattr(export_cfg, "pipeline_version",       _DEFAULT_PIPELINE_VERSION))
        feature_schema_version = str(getattr(export_cfg, "feature_schema_version", _DEFAULT_FEATURE_SCHEMA_VERSION))
        landsat_collection     = str(getattr(export_cfg, "landsat_collection",     _DEFAULT_LANDSAT_COLLECTION))

        git_commit  = self._read_git_commit()
        config_hash = self._hash_config()

        info = VersionInfo(
            dataset_version=dataset_version,
            pipeline_version=pipeline_version,
            feature_schema_version=feature_schema_version,
            landsat_collection=landsat_collection,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            config_hash=config_hash,
        )

        self._logger.debug(
            "VersionInfo generated: pipeline=%s, schema=%s",
            pipeline_version, feature_schema_version,
        )
        return info

    def save(self, version_info: VersionInfo, dataset_root: Path) -> Path:
        """
        Write VersionInfo to version.json in dataset_root.

        Args:
            version_info: VersionInfo to persist.
            dataset_root: Root dataset directory. File is written at
                          dataset_root/version.json.

        Returns:
            Absolute path to the written version.json.
        """
        dataset_root = Path(dataset_root).resolve()
        path         = dataset_root / VERSION_FILE_NAME

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(version_info.to_dict(), fh, indent=2, ensure_ascii=True)

        self._logger.info("Version info written: %s", path.name)
        return path

    @staticmethod
    def load(dataset_root: Path) -> VersionInfo:
        """
        Load VersionInfo from version.json in dataset_root.

        Args:
            dataset_root: Root dataset directory.

        Returns:
            Reconstructed VersionInfo.

        Raises:
            FileNotFoundError: version.json does not exist.
            ValueError:        JSON structure is invalid.
        """
        path = Path(dataset_root).resolve() / VERSION_FILE_NAME
        if not path.exists():
            raise FileNotFoundError(f"version.json not found in: {dataset_root}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return VersionInfo(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"version.json has unexpected structure in {dataset_root}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_git_commit(self) -> str | None:
        """
        Return the short git commit hash of the current HEAD, or None.

        Uses subprocess to run 'git rev-parse --short HEAD'. Never raises;
        returns None if git is unavailable, the directory is not a repo,
        or any other error occurs.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            commit = result.stdout.strip()
            return commit if commit else None
        except Exception as exc:
            self._logger.debug("Could not read git commit: %s", exc)
            return None

    def _hash_config(self) -> str | None:
        """
        Return an 8-character SHA256 hash of the config dict, or None.

        Produces a short fingerprint to detect config differences between
        exports. Never raises; returns None on any error.
        """
        try:
            config_dict = self._config.to_dict()
            serialized  = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
        except Exception as exc:
            self._logger.debug("Could not hash config: %s", exc)
            return None