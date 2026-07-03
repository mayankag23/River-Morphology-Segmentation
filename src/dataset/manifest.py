"""
Core data structures and manifest management for the Dataset Assembly
pipeline (Module 10).

DatasetSample is the central unit: a paired (patch, label) sample assembled
from PatchManifestEntry (Module 8) and LabelManifestEntry (Module 9) by
joining on patch_id. DatasetManifestEntry extends DatasetSample with a
split assignment. DatasetManifest is the frozen result; DatasetManifestManager
builds and persists all four CSV/JSON artifacts.

Output files:
    dataset_manifest.csv / .json  -- all samples with split assignments
    train.csv                     -- training split
    validation.csv                -- validation split
    test.csv                      -- test split
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.core.exceptions import InvalidValueError

__all__ = [
    "DatasetSample",
    "DatasetManifestEntry",
    "DatasetManifest",
    "DatasetManifestManager",
    "DATASET_MANIFEST_SCHEMA_VERSION",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
DATASET_MANIFEST_SCHEMA_VERSION: str = "1.0"

_DATASET_MANIFEST_CSV:  str = "dataset_manifest.csv"
_DATASET_MANIFEST_JSON: str = "dataset_manifest.json"
_TRAIN_CSV:             str = "train.csv"
_VALIDATION_CSV:        str = "validation.csv"
_TEST_CSV:              str = "test.csv"

_VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test", "excluded"})


# ==============================================================================
# DatasetSample
# ==============================================================================

@dataclass(frozen=True)
class DatasetSample:
    """
    Immutable paired (patch, label) sample assembled from Modules 8 and 9.

    Created by joining PatchManifestEntry and LabelManifestEntry on
    patch_id. Only samples where both a valid patch and a valid label exist
    (LabelManifestEntry.is_valid == True) are included.

    This record has NO split assignment. Split assignment is stored in
    DatasetManifestEntry after DatasetSplitter runs.

    Attributes:
        sample_id:                Unique sample identifier (equals patch_id).
        patch_id:                  Patch identifier from Module 8.
        scene_id:                  Scene identifier from Module 8.
        patch_path:                Absolute path to the patch GeoTIFF.
        mask_path:                 Absolute path to the label mask GeoTIFF.
        crs:                       CRS string of the patch.
        width:                     Patch width in pixels.
        height:                    Patch height in pixels.
        num_bands:                 Number of bands in the patch.
        row_index:                 Row position in the patch tiling grid.
        col_index:                 Column position in the patch tiling grid.
        patch_valid_pixel_ratio:    Valid pixel ratio from PatchManifestEntry.
        label_valid_pixel_ratio:    Valid pixel ratio from LabelManifestEntry.
        num_classes_present:        Distinct valid class IDs in the mask.
        acquisition_date:           Representative imagery date (YYYY-MM-DD).
        year:                       Calendar year of acquisition_date.
        month:                      Calendar month of acquisition_date.
        season:                     Resolved season name.
        hydrological_year:           Resolved hydrological year.
        sensor:                     Comma-separated sensor names.
        river_name:                  River name, or empty string if unset.
        reach_id:                    River reach identifier, or empty.
        basin_id:                    Drainage basin identifier, or empty.
        aoi_id:                      Area of interest identifier.
        label_version:                Label annotation version string.
        annotator:                    Annotation source identifier.
        confidence:                    Annotation confidence in [0.0, 1.0].
        confidence_source:             Origin of the confidence value.
    """

    sample_id:               str
    patch_id:                 str
    scene_id:                  str
    patch_path:                str
    mask_path:                 str
    crs:                       str
    width:                     int
    height:                    int
    num_bands:                 int
    row_index:                 int
    col_index:                 int
    patch_valid_pixel_ratio:    float
    label_valid_pixel_ratio:    float
    num_classes_present:        int
    acquisition_date:           str
    year:                       int
    month:                      int
    season:                     str
    hydrological_year:           int
    sensor:                     str
    river_name:                  str
    reach_id:                    str
    basin_id:                    str
    aoi_id:                      str
    label_version:                str
    annotator:                    str
    confidence:                    float
    confidence_source:             str


# ==============================================================================
# DatasetManifestEntry
# ==============================================================================

@dataclass(frozen=True)
class DatasetManifestEntry:
    """
    Immutable manifest record: a DatasetSample with its split assignment.

    All fields are str, int, float, or bool for lossless CSV round-trips.

    Attributes (inherits all DatasetSample fields, plus):
        split: "train", "validation", "test", or "excluded".
    """

    sample_id:               str
    patch_id:                 str
    scene_id:                  str
    patch_path:                str
    mask_path:                 str
    split:                     str
    crs:                       str
    width:                     int
    height:                    int
    num_bands:                 int
    row_index:                 int
    col_index:                 int
    patch_valid_pixel_ratio:    float
    label_valid_pixel_ratio:    float
    num_classes_present:        int
    acquisition_date:           str
    year:                       int
    month:                      int
    season:                     str
    hydrological_year:           int
    sensor:                     str
    river_name:                  str
    reach_id:                    str
    basin_id:                    str
    aoi_id:                      str
    label_version:                str
    annotator:                    str
    confidence:                    float
    confidence_source:             str

    def to_dict(self) -> dict:
        """Return a plain dict with all fields."""
        return asdict(self)

    @classmethod
    def from_sample(cls, sample: DatasetSample, split: str) -> DatasetManifestEntry:
        """
        Create a DatasetManifestEntry from a DatasetSample and split assignment.

        Args:
            sample: The DatasetSample to record.
            split:  Split assignment: "train", "validation", "test", or
                    "excluded".

        Raises:
            InvalidValueError: split is not a valid value.
        """
        if split not in _VALID_SPLITS:
            raise InvalidValueError(
                field="split",
                value=split,
                reason=f"must be one of {sorted(_VALID_SPLITS)}",
            )
        return cls(
            sample_id=sample.sample_id,
            patch_id=sample.patch_id,
            scene_id=sample.scene_id,
            patch_path=sample.patch_path,
            mask_path=sample.mask_path,
            split=split,
            crs=sample.crs,
            width=sample.width,
            height=sample.height,
            num_bands=sample.num_bands,
            row_index=sample.row_index,
            col_index=sample.col_index,
            patch_valid_pixel_ratio=sample.patch_valid_pixel_ratio,
            label_valid_pixel_ratio=sample.label_valid_pixel_ratio,
            num_classes_present=sample.num_classes_present,
            acquisition_date=sample.acquisition_date,
            year=sample.year,
            month=sample.month,
            season=sample.season,
            hydrological_year=sample.hydrological_year,
            sensor=sample.sensor,
            river_name=sample.river_name,
            reach_id=sample.reach_id,
            basin_id=sample.basin_id,
            aoi_id=sample.aoi_id,
            label_version=sample.label_version,
            annotator=sample.annotator,
            confidence=sample.confidence,
            confidence_source=sample.confidence_source,
        )


# ==============================================================================
# DatasetManifest (frozen result)
# ==============================================================================

@dataclass(frozen=True)
class DatasetManifest:
    """
    Immutable snapshot of all written dataset manifest files.

    Produced by DatasetManifestManager.save().

    Attributes:
        entries:                Ordered tuple of all DatasetManifestEntry records.
        dataset_manifest_csv:   Path to dataset_manifest.csv (None if not saved).
        dataset_manifest_json:  Path to dataset_manifest.json (None if not saved).
        train_csv:              Path to train.csv (None if not saved).
        validation_csv:         Path to validation.csv (None if not saved).
        test_csv:               Path to test.csv (None if not saved).
    """

    entries:               tuple[DatasetManifestEntry, ...]
    dataset_manifest_csv:   Path | None
    dataset_manifest_json:  Path | None
    train_csv:              Path | None
    validation_csv:         Path | None
    test_csv:               Path | None

    @property
    def entry_count(self) -> int:
        """Total number of entries (all splits)."""
        return len(self.entries)

    def entries_for_split(self, split: str) -> tuple[DatasetManifestEntry, ...]:
        """Return only entries assigned to the given split."""
        return tuple(e for e in self.entries if e.split == split)

    @classmethod
    def from_csv(cls, path: Path) -> DatasetManifest:
        """
        Load a DatasetManifest from dataset_manifest.csv.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        CSV structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset manifest CSV not found: {path}")
        try:
            entries: list[DatasetManifestEntry] = []
            with open(path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    entries.append(_row_to_entry(row))
        except (csv.Error, KeyError, TypeError) as exc:
            raise ValueError(f"Dataset manifest CSV has unexpected structure: {exc}") from exc
        return cls(
            entries=tuple(entries),
            dataset_manifest_csv=path, dataset_manifest_json=None,
            train_csv=None, validation_csv=None, test_csv=None,
        )


# ==============================================================================
# DatasetManifestManager (mutable manager)
# ==============================================================================

class DatasetManifestManager:
    """
    Accumulates DatasetManifestEntry records and writes manifest files.

    Writes:
        dataset_manifest.csv / .json -- all entries
        train.csv, validation.csv, test.csv -- per-split entries

    Args:
        entries: Optional initial entries.
    """

    _FIELD_NAMES: tuple[str, ...] = tuple(
        f.name for f in fields(DatasetManifestEntry)
    )

    def __init__(
        self,
        entries: list[DatasetManifestEntry] | None = None,
    ) -> None:
        self._entries: list[DatasetManifestEntry] = list(entries) if entries else []
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[DatasetManifestEntry, ...]:
        return tuple(self._entries)

    def add_entry(self, entry: DatasetManifestEntry) -> None:
        """
        Append a DatasetManifestEntry.

        Raises:
            InvalidValueError: entry is not a DatasetManifestEntry instance.
        """
        if not isinstance(entry, DatasetManifestEntry):
            raise InvalidValueError(
                field="entry",
                value=type(entry).__name__,
                reason="must be a DatasetManifestEntry instance",
            )
        self._entries.append(entry)

    def save(
        self,
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> DatasetManifest:
        """
        Write manifest files and return a frozen DatasetManifest.

        Args:
            output_dir: Directory to write manifest files into.
            formats:    ["csv"], ["json"], or both.

        Returns:
            Frozen DatasetManifest.
        """
        output_dir = Path(output_dir).resolve()
        formats    = [f.lower() for f in (formats or ["csv", "json"])]

        dataset_manifest_csv: Path | None  = None
        dataset_manifest_json: Path | None = None
        train_csv: Path | None             = None
        val_csv:   Path | None             = None
        test_csv:  Path | None             = None

        if "csv" in formats:
            dataset_manifest_csv = output_dir / _DATASET_MANIFEST_CSV
            self._write_csv(dataset_manifest_csv, self._entries)

            train_entries = [e for e in self._entries if e.split == "train"]
            val_entries   = [e for e in self._entries if e.split == "validation"]
            test_entries  = [e for e in self._entries if e.split == "test"]

            if train_entries:
                train_csv = output_dir / _TRAIN_CSV
                self._write_csv(train_csv, train_entries)
            if val_entries:
                val_csv = output_dir / _VALIDATION_CSV
                self._write_csv(val_csv, val_entries)
            if test_entries:
                test_csv = output_dir / _TEST_CSV
                self._write_csv(test_csv, test_entries)

        if "json" in formats:
            dataset_manifest_json = output_dir / _DATASET_MANIFEST_JSON
            self._write_json(dataset_manifest_json, self._entries)

        self._logger.info(
            "Dataset manifest written: %d entries (train=%d, val=%d, test=%d).",
            len(self._entries),
            sum(1 for e in self._entries if e.split == "train"),
            sum(1 for e in self._entries if e.split == "validation"),
            sum(1 for e in self._entries if e.split == "test"),
        )
        return DatasetManifest(
            entries=tuple(self._entries),
            dataset_manifest_csv=dataset_manifest_csv,
            dataset_manifest_json=dataset_manifest_json,
            train_csv=train_csv,
            validation_csv=val_csv,
            test_csv=test_csv,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _write_csv(self, path: Path, entries: list[DatasetManifestEntry]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self._FIELD_NAMES))
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry.to_dict())
        self._logger.debug("CSV written: %s (%d rows)", path.name, len(entries))

    def _write_json(
        self, path: Path, entries: list[DatasetManifestEntry]
    ) -> None:
        payload = {
            "version":     DATASET_MANIFEST_SCHEMA_VERSION,
            "entry_count": len(entries),
            "entries":     [e.to_dict() for e in entries],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
        self._logger.debug("JSON written: %s (%d rows)", path.name, len(entries))


# ==============================================================================
# Private helpers
# ==============================================================================

def _row_to_entry(row: dict) -> DatasetManifestEntry:
    """Convert a CSV DictReader row (all-string values) to DatasetManifestEntry."""
    return DatasetManifestEntry(
        sample_id=row["sample_id"],
        patch_id=row["patch_id"],
        scene_id=row["scene_id"],
        patch_path=row["patch_path"],
        mask_path=row["mask_path"],
        split=row["split"],
        crs=row["crs"],
        width=int(row["width"]),
        height=int(row["height"]),
        num_bands=int(row["num_bands"]),
        row_index=int(row["row_index"]),
        col_index=int(row["col_index"]),
        patch_valid_pixel_ratio=float(row["patch_valid_pixel_ratio"]),
        label_valid_pixel_ratio=float(row["label_valid_pixel_ratio"]),
        num_classes_present=int(row["num_classes_present"]),
        acquisition_date=row["acquisition_date"],
        year=int(row["year"]),
        month=int(row["month"]),
        season=row["season"],
        hydrological_year=int(row["hydrological_year"]),
        sensor=row["sensor"],
        river_name=row["river_name"],
        reach_id=row["reach_id"],
        basin_id=row["basin_id"],
        aoi_id=row["aoi_id"],
        label_version=row["label_version"],
        annotator=row["annotator"],
        confidence=float(row["confidence"]),
        confidence_source=row["confidence_source"],
    )