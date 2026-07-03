"""
Dataset version management for the Dataset Assembly pipeline (Module 10).

Produces version.json describing the assembled dataset's lineage. Follows
the same pattern as src.export.version.DatasetVersionManager (Module 7)
but stores assembly-specific fields (split strategy, sample counts).
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

__all__ = ["DatasetVersionInfo", "DatasetVersionManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_VERSION_FILENAME: str  = "version.json"

_DEFAULT_DATASET_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class DatasetVersionInfo:
    """
    Immutable dataset version and lineage record for Module 10.

    Attributes:
        dataset_version:          User-facing dataset release version.
        assembly_timestamp:        ISO 8601 UTC timestamp of assembly.
        total_samples:             Total samples included in the dataset.
        train_samples:             Training split sample count.
        validation_samples:         Validation split sample count.
        test_samples:              Test split sample count.
        excluded_samples:           Excluded sample count.
        split_strategy:             Split strategy used ("random", "temporal",
                                   "spatial").
        train_ratio:                Configured training split ratio.
        validation_ratio:            Configured validation split ratio.
        test_ratio:                  Configured test split ratio.
        random_seed:                 Random seed used for reproducibility.
        source_scenes:               Number of distinct source scenes.
        git_commit:                  Short git commit hash, or None.
        config_hash:                 8-character SHA256 of config, or None.
    """

    dataset_version:    str
    assembly_timestamp:  str
    total_samples:       int
    train_samples:       int
    validation_samples:   int
    test_samples:         int
    excluded_samples:     int
    split_strategy:       str
    train_ratio:          float
    validation_ratio:      float
    test_ratio:            float
    random_seed:           int
    source_scenes:         int
    git_commit:            str | None
    config_hash:           str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plain dict."""
        return dataclasses.asdict(self)


class DatasetVersionManager:
    """
    Generates and persists DatasetVersionInfo as version.json.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

    def generate(
        self,
        total_samples:     int,
        train_samples:     int,
        validation_samples: int,
        test_samples:       int,
        excluded_samples:   int,
        split_strategy:     str,
        source_scenes:      int,
    ) -> DatasetVersionInfo:
        """
        Build a DatasetVersionInfo from the current config and assembly results.

        Args:
            total_samples:       Total samples in dataset.
            train_samples:       Training split count.
            validation_samples:   Validation split count.
            test_samples:         Test split count.
            excluded_samples:     Excluded count.
            split_strategy:       Strategy used for splitting.
            source_scenes:        Number of distinct source scenes.

        Returns:
            Populated, immutable DatasetVersionInfo.
        """
        dataset_cfg = getattr(self._config, "dataset", None)
        split_cfg   = getattr(dataset_cfg, "split", None)

        dataset_version = str(
            getattr(dataset_cfg, "dataset_version", _DEFAULT_DATASET_VERSION)
        )
        train_ratio     = float(getattr(split_cfg, "train_ratio",  0.70))
        val_ratio       = float(getattr(split_cfg, "val_ratio",    0.15))
        test_ratio      = float(getattr(split_cfg, "test_ratio",   0.15))
        random_seed     = int(getattr(split_cfg,   "random_seed",  42))

        info = DatasetVersionInfo(
            dataset_version=dataset_version,
            assembly_timestamp=datetime.now(timezone.utc).isoformat(),
            total_samples=total_samples,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            excluded_samples=excluded_samples,
            split_strategy=split_strategy,
            train_ratio=train_ratio,
            validation_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            source_scenes=source_scenes,
            git_commit=self._read_git_commit(),
            config_hash=self._hash_config(),
        )
        return info

    def save(self, version_info: DatasetVersionInfo, output_dir: Path) -> Path:
        """
        Write DatasetVersionInfo to version.json in output_dir.

        Returns:
            Absolute path to the written file.
        """
        output_dir = Path(output_dir).resolve()
        path       = output_dir / _VERSION_FILENAME
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(version_info.to_dict(), fh, indent=2, ensure_ascii=True)
        self._logger.info("Dataset version info written: %s", path.name)
        return path

    @staticmethod
    def load(output_dir: Path) -> DatasetVersionInfo:
        """
        Load DatasetVersionInfo from version.json in output_dir.

        Raises:
            FileNotFoundError: version.json does not exist.
            ValueError:        JSON structure is invalid.
        """
        path = Path(output_dir).resolve() / _VERSION_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"version.json not found in: {output_dir}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return DatasetVersionInfo(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"version.json has unexpected structure: {exc}") from exc

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _read_git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return result.stdout.strip() or None
        except Exception as exc:
            self._logger.debug("Could not read git commit: %s", exc)
            return None

    def _hash_config(self) -> str | None:
        try:
            config_dict = self._config.to_dict()
            serialized  = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
        except Exception as exc:
            self._logger.debug("Could not hash config: %s", exc)
            return None