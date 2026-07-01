"""
Patch dataset manifest management for the Patch Generation pipeline (Module 8).

PatchManifestEntry is one record per generated patch, containing both
patch-level metadata (row/col grid position, pixel offsets, geo properties)
and provenance (source scene, validity ratio, file size).

PatchManifest is the frozen snapshot persisted as CSV/JSON.
PatchManifestManager accumulates entries during generation and persists
them, mirroring the design of src.export.manifest.DatasetManifestManager
from Module 7 (same pattern, distinct artifact: patch_manifest.csv/json
rather than manifest.csv/json).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.core.exceptions import InvalidValueError

__all__ = [
    "PatchManifestEntry",
    "PatchManifest",
    "PatchManifestManager",
    "PATCH_MANIFEST_SCHEMA_VERSION",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
PATCH_MANIFEST_SCHEMA_VERSION: str = "1.0"

_PATCH_MANIFEST_CSV_FILENAME:  str = "patch_manifest.csv"
_PATCH_MANIFEST_JSON_FILENAME: str = "patch_manifest.json"


# ==============================================================================
# PatchManifestEntry
# ==============================================================================

@dataclass(frozen=True)
class PatchManifestEntry:
    """
    Immutable record for one generated patch.

    All fields use str, int, float for lossless CSV round-trips. Paths are
    stored as strings (absolute, set by PatchGenerator).

    Attributes:
        patch_id:           Deterministic patch identifier,
                            e.g. "scene001_r000_c000".
        scene_id:            Source scene identifier.
        source_image_path:   Path to the source scene GeoTIFF (image.tif).
        patch_path:          Path to the written patch GeoTIFF.
        row_index:           Row position in the tiling grid.
        col_index:           Column position in the tiling grid.
        row_off:             Pixel row offset in the source raster.
        col_off:             Pixel column offset in the source raster.
        height:              Patch height in pixels.
        width:               Patch width in pixels.
        num_bands:           Number of bands in the patch.
        crs:                 CRS string of the patch.
        valid_pixel_ratio:   Fraction of valid (non-NoData) pixels [0, 1].
        file_size_bytes:     Patch GeoTIFF file size in bytes.
        created_at:          ISO 8601 UTC timestamp of patch creation.
    """

    patch_id:           str
    scene_id:           str
    source_image_path:  str
    patch_path:         str
    row_index:          int
    col_index:          int
    row_off:            int
    col_off:            int
    height:             int
    width:              int
    num_bands:          int
    crs:                str
    valid_pixel_ratio:  float
    file_size_bytes:    int
    created_at:         str

    def to_dict(self) -> dict:
        """Return a plain dict with all fields."""
        return asdict(self)


# ==============================================================================
# PatchManifest (frozen result / data contract)
# ==============================================================================

@dataclass(frozen=True)
class PatchManifest:
    """
    Immutable snapshot of the patch dataset manifest at a point in time.

    Produced by PatchManifestManager.save() or from_csv() / from_json().
    Never constructed directly by external code.

    Attributes:
        entries:   Ordered tuple of all PatchManifestEntry records.
        csv_path:  Absolute path to the written CSV file (None if not saved).
        json_path: Absolute path to the written JSON file (None if not saved).
    """

    entries:   tuple[PatchManifestEntry, ...]
    csv_path:  Path | None
    json_path: Path | None

    @property
    def entry_count(self) -> int:
        """Number of patch entries in this manifest."""
        return len(self.entries)

    @classmethod
    def from_csv(cls, path: Path) -> PatchManifest:
        """
        Load a PatchManifest from a CSV file.

        Args:
            path: Path to a CSV file written by PatchManifestManager.save().

        Returns:
            PatchManifest with entries from the file.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        CSV structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Patch manifest CSV not found: {path}")
        try:
            entries: list[PatchManifestEntry] = []
            with open(path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    entries.append(_row_to_entry(row))
        except (csv.Error, KeyError, TypeError) as exc:
            raise ValueError(
                f"CSV manifest {path} has unexpected structure: {exc}"
            ) from exc
        _LOGGER.info(
            "Patch manifest CSV loaded: %s (%d entries)", path.name, len(entries)
        )
        return cls(entries=tuple(entries), csv_path=path, json_path=None)

    @classmethod
    def from_json(cls, path: Path) -> PatchManifest:
        """
        Load a PatchManifest from a JSON file.

        Args:
            path: Path to a JSON file written by PatchManifestManager.save().

        Returns:
            PatchManifest with entries from the file.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        JSON structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Patch manifest JSON not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            entries = tuple(
                PatchManifestEntry(**e) for e in payload.get("entries", [])
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(
                f"JSON manifest {path} has unexpected structure: {exc}"
            ) from exc
        _LOGGER.info(
            "Patch manifest JSON loaded: %s (%d entries)", path.name, len(entries)
        )
        return cls(entries=entries, csv_path=None, json_path=path)


# ==============================================================================
# PatchManifestManager (mutable manager)
# ==============================================================================

class PatchManifestManager:
    """
    Accumulates PatchManifestEntry records and persists them as PatchManifest.

    Usage pattern:
        manager = PatchManifestManager()
        manager.load_existing(output_dir)        # optional, for append
        manager.add_entry(entry)
        manifest = manager.save(output_dir, formats=["csv", "json"])

    Args:
        entries: Optional initial entries (e.g. loaded from an existing
                 patch manifest).
    """

    _FIELD_NAMES: tuple[str, ...] = tuple(
        f.name for f in fields(PatchManifestEntry)
    )

    def __init__(
        self,
        entries: list[PatchManifestEntry] | None = None,
    ) -> None:
        self._entries: list[PatchManifestEntry] = list(entries) if entries else []
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def entry_count(self) -> int:
        """Number of entries currently held by this manager."""
        return len(self._entries)

    def add_entry(self, entry: PatchManifestEntry) -> None:
        """
        Append a PatchManifestEntry to the manager.

        Args:
            entry: A PatchManifestEntry for a successfully written patch.

        Raises:
            InvalidValueError: entry is not a PatchManifestEntry instance.
        """
        if not isinstance(entry, PatchManifestEntry):
            raise InvalidValueError(
                field="entry",
                value=type(entry).__name__,
                reason="must be a PatchManifestEntry instance",
            )
        self._entries.append(entry)
        self._logger.debug("Patch manifest entry added: patch_id=%s", entry.patch_id)

    def load_existing(self, output_dir: Path) -> int:
        """
        Load entries from an existing patch_manifest.csv at output_dir.

        Replaces the current entries list. Returns the number loaded.
        If no manifest CSV exists, the manager retains its current state.

        Args:
            output_dir: Root patch dataset directory.

        Returns:
            Number of entries loaded (0 if no file found).
        """
        csv_path = Path(output_dir) / _PATCH_MANIFEST_CSV_FILENAME
        if not csv_path.exists():
            return 0
        try:
            manifest = PatchManifest.from_csv(csv_path)
            self._entries = list(manifest.entries)
            return len(self._entries)
        except (ValueError, OSError) as exc:
            self._logger.warning(
                "Could not load existing patch manifest at %s; starting fresh: %s",
                csv_path, exc,
            )
            return 0

    def save(
        self,
        output_dir: Path,
        formats:    list[str] | None = None,
    ) -> PatchManifest:
        """
        Write entries to CSV and/or JSON and return a frozen PatchManifest.

        Args:
            output_dir: Root patch dataset directory. Files are written at:
                       output_dir/patch_manifest.csv and
                       output_dir/patch_manifest.json.
            formats:    List of format strings: "csv", "json", or both.
                       Defaults to ["csv", "json"].

        Returns:
            Frozen PatchManifest with entries and the written file paths.
        """
        output_dir = Path(output_dir).resolve()
        formats    = [f.lower() for f in (formats or ["csv", "json"])]

        csv_path:  Path | None = None
        json_path: Path | None = None

        if "csv" in formats:
            csv_path = output_dir / _PATCH_MANIFEST_CSV_FILENAME
            self._write_csv(csv_path)
        if "json" in formats:
            json_path = output_dir / _PATCH_MANIFEST_JSON_FILENAME
            self._write_json(json_path)

        return PatchManifest(
            entries=tuple(self._entries),
            csv_path=csv_path,
            json_path=json_path,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _write_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self._FIELD_NAMES))
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(entry.to_dict())
        self._logger.info(
            "Patch manifest CSV written: %s (%d entries)",
            path.name, len(self._entries),
        )

    def _write_json(self, path: Path) -> None:
        payload = {
            "version":     PATCH_MANIFEST_SCHEMA_VERSION,
            "entry_count": len(self._entries),
            "entries":     [e.to_dict() for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
        self._logger.info(
            "Patch manifest JSON written: %s (%d entries)",
            path.name, len(self._entries),
        )


# ==============================================================================
# Private helpers
# ==============================================================================

def _row_to_entry(row: dict) -> PatchManifestEntry:
    """Convert a CSV DictReader row (all-string values) to PatchManifestEntry."""
    return PatchManifestEntry(
        patch_id           = row["patch_id"],
        scene_id            = row["scene_id"],
        source_image_path   = row["source_image_path"],
        patch_path          = row["patch_path"],
        row_index           = int(row["row_index"]),
        col_index            = int(row["col_index"]),
        row_off              = int(row["row_off"]),
        col_off               = int(row["col_off"]),
        height                = int(row["height"]),
        width                 = int(row["width"]),
        num_bands             = int(row["num_bands"]),
        crs                   = row["crs"],
        valid_pixel_ratio     = float(row["valid_pixel_ratio"]),
        file_size_bytes       = int(row["file_size_bytes"]),
        created_at            = row["created_at"],
    )