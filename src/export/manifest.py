"""
Dataset manifest management for the River Morphology export pipeline.

A dataset manifest is a flat table with one row per exported scene.
It is persisted in both CSV (for spreadsheet tools and pandas) and
JSON (for programmatic use).

Data contract types:
    ManifestEntry       -- Frozen single-scene record.
    DatasetManifest     -- Frozen manifest snapshot (entries + file paths).

Manager class:
    DatasetManifestManager  -- Mutable manager; call save() to get DatasetManifest.

DatasetManifest is always obtained from DatasetManifestManager.save() or
the static from_csv() / from_json() class methods. It is never constructed
directly by external code.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.core.exceptions import InvalidValueError

__all__ = [
    "ManifestEntry",
    "DatasetManifest",
    "DatasetManifestManager",
    "MANIFEST_SCHEMA_VERSION",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION: str = "1.0"

_MANIFEST_CSV_FILENAME:  str = "manifest.csv"
_MANIFEST_JSON_FILENAME: str = "manifest.json"


# ==============================================================================
# ManifestEntry
# ==============================================================================

@dataclass(frozen=True)
class ManifestEntry:
    """
    Immutable record for one exported scene in the dataset manifest.

    All fields use str, int, float, or bool for lossless CSV round-trips.
    Paths are stored as strings (absolute or relative at caller's choice).

    Attributes:
        scene_id:             Unique scene identifier.
        image_path:           Path to the scene GeoTIFF (image.tif).
        metadata_path:        Path to the scene metadata JSON.
        export_timestamp:     ISO 8601 UTC timestamp of the export.
        num_bands:            Total band count in the GeoTIFF.
        width:                Raster width in pixels.
        height:               Raster height in pixels.
        crs:                  Output CRS string.
        start_date:           Collection date-range start (YYYY-MM-DD).
        end_date:             Collection date-range end (YYYY-MM-DD).
        sensors:              Comma-separated sensor names, e.g. "L8,L9".
        composite_method:     Compositing method used.
        cloud_cover_limit:    Maximum cloud cover threshold applied.
        num_spectral_indices: Number of spectral indices in the image.
        file_size_bytes:      GeoTIFF file size in bytes.
        is_valid:             True if GeoTiffValidator found no issues.
    """

    scene_id:             str
    image_path:           str
    metadata_path:        str
    export_timestamp:     str
    num_bands:            int
    width:                int
    height:               int
    crs:                  str
    start_date:           str
    end_date:             str
    sensors:              str
    composite_method:     str
    cloud_cover_limit:    float
    num_spectral_indices: int
    file_size_bytes:      int
    is_valid:             bool

    def to_dict(self) -> dict:
        """Return a plain dict with all fields."""
        return asdict(self)


# ==============================================================================
# DatasetManifest (frozen result / data contract)
# ==============================================================================

@dataclass(frozen=True)
class DatasetManifest:
    """
    Immutable snapshot of the dataset manifest at a point in time.

    Produced by DatasetManifestManager.save() or from_csv() / from_json().
    Never constructed directly by external code.

    Attributes:
        entries:   Ordered tuple of all ManifestEntry records.
        csv_path:  Absolute path to the written CSV file (None if not saved).
        json_path: Absolute path to the written JSON file (None if not saved).
    """

    entries:   tuple[ManifestEntry, ...]
    csv_path:  Path | None
    json_path: Path | None

    @property
    def entry_count(self) -> int:
        """Number of scene entries in this manifest."""
        return len(self.entries)

    @classmethod
    def from_csv(cls, path: Path) -> DatasetManifest:
        """
        Load a DatasetManifest from a CSV file.

        Args:
            path: Path to a CSV file written by DatasetManifestManager.save().

        Returns:
            DatasetManifest with entries from the file.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        CSV structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {path}")
        try:
            entries: list[ManifestEntry] = []
            with open(path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    entries.append(_row_to_entry(row))
        except (csv.Error, KeyError, TypeError) as exc:
            raise ValueError(f"CSV manifest {path} has unexpected structure: {exc}") from exc
        _LOGGER.info("Manifest CSV loaded: %s (%d entries)", path.name, len(entries))
        return cls(entries=tuple(entries), csv_path=path, json_path=None)

    @classmethod
    def from_json(cls, path: Path) -> DatasetManifest:
        """
        Load a DatasetManifest from a JSON file.

        Args:
            path: Path to a JSON file written by DatasetManifestManager.save().

        Returns:
            DatasetManifest with entries from the file.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        JSON structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest JSON not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            entries = tuple(ManifestEntry(**e) for e in payload.get("entries", []))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(f"JSON manifest {path} has unexpected structure: {exc}") from exc
        _LOGGER.info("Manifest JSON loaded: %s (%d entries)", path.name, len(entries))
        return cls(entries=entries, csv_path=None, json_path=path)


# ==============================================================================
# DatasetManifestManager (mutable manager)
# ==============================================================================

class DatasetManifestManager:
    """
    Accumulates ManifestEntry records and persists them as DatasetManifest.

    Usage pattern:
        manager = DatasetManifestManager()
        manager.load_existing(dataset_root)           # optional, for append
        manager.add_entry(entry)
        manifest = manager.save(dataset_root, formats=["csv", "json"])

    Args:
        entries: Optional initial entries (e.g. loaded from an existing manifest).
    """

    _FIELD_NAMES: tuple[str, ...] = tuple(
        f.name for f in fields(ManifestEntry)
    )

    def __init__(
        self,
        entries: list[ManifestEntry] | None = None,
    ) -> None:
        self._entries: list[ManifestEntry] = list(entries) if entries else []
        self._logger: logging.Logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entry(self, entry: ManifestEntry) -> None:
        """
        Append a ManifestEntry to the manager.

        Args:
            entry: A ManifestEntry for a completed export.

        Raises:
            InvalidValueError: entry is not a ManifestEntry instance.
        """
        if not isinstance(entry, ManifestEntry):
            raise InvalidValueError(
                field="entry",
                value=type(entry).__name__,
                reason="must be a ManifestEntry instance",
            )
        self._entries.append(entry)
        self._logger.debug("Manifest entry added: scene_id=%s", entry.scene_id)

    def load_existing(self, dataset_root: Path) -> int:
        """
        Load entries from an existing manifest CSV at dataset_root.

        Replaces the current entries list. Returns the number loaded.
        If no manifest CSV exists, the manager retains its current state.

        Args:
            dataset_root: Root dataset directory (parent of manifest.csv).

        Returns:
            Number of entries loaded (0 if no file found).
        """
        csv_path = Path(dataset_root) / _MANIFEST_CSV_FILENAME
        if not csv_path.exists():
            return 0
        try:
            manifest = DatasetManifest.from_csv(csv_path)
            self._entries = list(manifest.entries)
            return len(self._entries)
        except (ValueError, OSError) as exc:
            self._logger.warning(
                "Could not load existing manifest at %s; starting fresh: %s",
                csv_path, exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        dataset_root: Path,
        formats:      list[str] | None = None,
    ) -> DatasetManifest:
        """
        Write entries to CSV and/or JSON and return a frozen DatasetManifest.

        Args:
            dataset_root: Root dataset directory. Files are written at:
                          dataset_root/manifest.csv and dataset_root/manifest.json.
            formats:      List of format strings: "csv", "json", or both.
                          Defaults to ["csv", "json"].

        Returns:
            Frozen DatasetManifest with entries and the written file paths.
        """
        dataset_root = Path(dataset_root).resolve()
        formats      = [f.lower() for f in (formats or ["csv", "json"])]

        csv_path:  Path | None = None
        json_path: Path | None = None

        if "csv" in formats:
            csv_path = dataset_root / _MANIFEST_CSV_FILENAME
            self._write_csv(csv_path)

        if "json" in formats:
            json_path = dataset_root / _MANIFEST_JSON_FILENAME
            self._write_json(json_path)

        return DatasetManifest(
            entries=tuple(self._entries),
            csv_path=csv_path,
            json_path=json_path,
        )

    # ------------------------------------------------------------------
    # Private write helpers
    # ------------------------------------------------------------------

    def _write_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self._FIELD_NAMES))
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(entry.to_dict())
        self._logger.info(
            "Manifest CSV written: %s (%d entries)", path.name, len(self._entries)
        )

    def _write_json(self, path: Path) -> None:
        payload = {
            "version":     MANIFEST_SCHEMA_VERSION,
            "entry_count": len(self._entries),
            "entries":     [e.to_dict() for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
        self._logger.info(
            "Manifest JSON written: %s (%d entries)", path.name, len(self._entries)
        )


# ==============================================================================
# Private helpers
# ==============================================================================

def _row_to_entry(row: dict) -> ManifestEntry:
    """Convert a CSV DictReader row (all-string values) to ManifestEntry."""
    return ManifestEntry(
        scene_id             = row["scene_id"],
        image_path           = row["image_path"],
        metadata_path        = row["metadata_path"],
        export_timestamp     = row["export_timestamp"],
        num_bands            = int(row["num_bands"]),
        width                = int(row["width"]),
        height               = int(row["height"]),
        crs                  = row["crs"],
        start_date           = row["start_date"],
        end_date             = row["end_date"],
        sensors              = row["sensors"],
        composite_method     = row["composite_method"],
        cloud_cover_limit    = float(row["cloud_cover_limit"]),
        num_spectral_indices = int(row["num_spectral_indices"]),
        file_size_bytes      = int(row["file_size_bytes"]),
        is_valid             = row["is_valid"].strip().lower() == "true",
    )