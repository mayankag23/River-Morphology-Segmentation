"""
Label dataset manifest management for the Label Management pipeline
(Module 9 -- Pseudo-Label Generation).

Mirrors the design of src.export.manifest.DatasetManifestManager (Module 7)
and src.patches.manifest.PatchManifestManager (Module 8): one
LabelManifestEntry per processed patch, persisted as both CSV and JSON,
append-safe across multiple scene-processing runs.

Schema version 1.1 adds the `generation_strategy` field to LabelManifestEntry.
Existing CSV files written by version 1.0 are loaded with a default value of
"spectral_rules" for the missing column (backward-compatible via dict.get()).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.core.exceptions import InvalidValueError

__all__ = [
    "LabelManifestEntry",
    "LabelManifest",
    "LabelManifestManager",
    "LABEL_MANIFEST_SCHEMA_VERSION",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
LABEL_MANIFEST_SCHEMA_VERSION: str = "1.1"

_LABEL_MANIFEST_CSV_FILENAME:  str = "label_manifest.csv"
_LABEL_MANIFEST_JSON_FILENAME: str = "label_manifest.json"

# Default generation_strategy for CSV rows written before schema v1.1.
_DEFAULT_GENERATION_STRATEGY: str = "spectral_rules"


# ==============================================================================
# LabelManifestEntry
# ==============================================================================

@dataclass(frozen=True)
class LabelManifestEntry:
    """
    Immutable record for one processed label (schema version 1.1).

    All fields use str, int, float, or bool for lossless CSV round-trips.
    Optional temporal fields (river_name, reach_id, basin_id) are stored
    as empty strings when unset.

    Attributes:
        patch_id:              Deterministic patch identifier (Module 8).
        scene_id:                Source scene identifier.
        patch_path:               Path to the source patch GeoTIFF.
        mask_path:                Path to the generated label mask GeoTIFF.
        crs:                      CRS string of the patch.
        width:                    Patch width in pixels.
        height:                   Patch height in pixels.
        is_valid:                 True if label passed all validation checks.
        validation_issues:         Semicolon-joined human-readable issue list.
        num_classes_present:       Distinct valid class IDs found in the mask.
        valid_pixel_ratio:          Fraction of valid (non-nodata) mask pixels.
        source_type:                Label origin category. Always "pseudo_label"
                                  for automatically generated labels.
        acquisition_date:            Representative imagery date (YYYY-MM-DD).
        year:                        Calendar year of acquisition_date.
        month:                       Calendar month of acquisition_date.
        season:                      Resolved season name.
        hydrological_year:            Resolved hydrological year.
        sensor:                      Comma-separated sensor names.
        river_name:                   River name, or empty string if unset.
        reach_id:                     River reach identifier, or empty if unset.
        basin_id:                     Drainage basin identifier, or empty if unset.
        aoi_id:                       AOI identifier.
        label_version:                 Label annotation version string.
        annotator:                     Annotation source identifier.
        confidence:                    Annotation confidence in [0.0, 1.0].
        confidence_source:              Origin of the confidence value.
        processing_history:             Comma-joined processing steps applied.
        created_at:                     ISO 8601 UTC timestamp of manifest
                                     entry creation.
        generation_strategy:             Strategy used to generate this label.
                                     e.g. "spectral_rules", "sam", "ml".
                                     Default "spectral_rules" when loading v1.0
                                     CSV files that lack this column.
    """

    patch_id:              str
    scene_id:               str
    patch_path:              str
    mask_path:               str
    crs:                     str
    width:                   int
    height:                  int
    is_valid:                bool
    validation_issues:        str
    num_classes_present:      int
    valid_pixel_ratio:         float
    source_type:                str
    acquisition_date:           str
    year:                       int
    month:                      int
    season:                     str
    hydrological_year:            int
    sensor:                     str
    river_name:                  str
    reach_id:                    str
    basin_id:                    str
    aoi_id:                      str
    label_version:                str
    annotator:                    str
    confidence:                   float
    confidence_source:             str
    processing_history:            str
    created_at:                    str
    generation_strategy:           str = _DEFAULT_GENERATION_STRATEGY

    def to_dict(self) -> dict:
        """Return a plain dict with all fields."""
        return asdict(self)


# ==============================================================================
# LabelManifest (frozen result / data contract)
# ==============================================================================

@dataclass(frozen=True)
class LabelManifest:
    """
    Immutable snapshot of the label manifest at a point in time.

    Produced by LabelManifestManager.save() or from_csv() / from_json().
    Never constructed directly by external code.

    Attributes:
        entries:   Ordered tuple of all LabelManifestEntry records.
        csv_path:  Absolute path to the written CSV file (None if not saved).
        json_path: Absolute path to the written JSON file (None if not saved).
    """

    entries:   tuple[LabelManifestEntry, ...]
    csv_path:  Path | None
    json_path: Path | None

    @property
    def entry_count(self) -> int:
        """Number of label entries in this manifest."""
        return len(self.entries)

    @classmethod
    def from_csv(cls, path: Path) -> LabelManifest:
        """
        Load a LabelManifest from a CSV file.

        Backward-compatible with schema v1.0 files that lack the
        generation_strategy column (defaults to "spectral_rules").

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        CSV structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Label manifest CSV not found: {path}")
        try:
            entries: list[LabelManifestEntry] = []
            with open(path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    entries.append(_row_to_entry(row))
        except (csv.Error, KeyError, TypeError) as exc:
            raise ValueError(
                f"CSV manifest {path} has unexpected structure: {exc}"
            ) from exc
        _LOGGER.info(
            "Label manifest CSV loaded: %s (%d entries)", path.name, len(entries)
        )
        return cls(entries=tuple(entries), csv_path=path, json_path=None)

    @classmethod
    def from_json(cls, path: Path) -> LabelManifest:
        """
        Load a LabelManifest from a JSON file.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        JSON structure is invalid.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Label manifest JSON not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            entries = tuple(
                _dict_to_entry(e) for e in payload.get("entries", [])
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(
                f"JSON manifest {path} has unexpected structure: {exc}"
            ) from exc
        _LOGGER.info(
            "Label manifest JSON loaded: %s (%d entries)", path.name, len(entries)
        )
        return cls(entries=entries, csv_path=None, json_path=path)


# ==============================================================================
# LabelManifestManager (mutable manager)
# ==============================================================================

class LabelManifestManager:
    """
    Accumulates LabelManifestEntry records and persists them as LabelManifest.

    Args:
        entries: Optional initial entries (e.g. loaded from an existing
                 label manifest).
    """

    _FIELD_NAMES: tuple[str, ...] = tuple(
        f.name for f in fields(LabelManifestEntry)
    )

    def __init__(
        self,
        entries: list[LabelManifestEntry] | None = None,
    ) -> None:
        self._entries: list[LabelManifestEntry] = list(entries) if entries else []
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def entry_count(self) -> int:
        """Number of entries currently held by this manager."""
        return len(self._entries)

    @property
    def entries(self) -> tuple[LabelManifestEntry, ...]:
        """Currently held entries, in insertion order."""
        return tuple(self._entries)

    def add_entry(self, entry: LabelManifestEntry) -> None:
        """
        Append a LabelManifestEntry to the manager.

        Raises:
            InvalidValueError: entry is not a LabelManifestEntry instance.
        """
        if not isinstance(entry, LabelManifestEntry):
            raise InvalidValueError(
                field="entry",
                value=type(entry).__name__,
                reason="must be a LabelManifestEntry instance",
            )
        self._entries.append(entry)
        self._logger.debug("Label manifest entry added: patch_id=%s", entry.patch_id)

    def load_existing(self, output_dir: Path) -> int:
        """
        Load entries from an existing label_manifest.csv at output_dir.

        Returns:
            Number of entries loaded (0 if no file found).
        """
        csv_path = Path(output_dir) / _LABEL_MANIFEST_CSV_FILENAME
        if not csv_path.exists():
            return 0
        try:
            manifest = LabelManifest.from_csv(csv_path)
            self._entries = list(manifest.entries)
            return len(self._entries)
        except (ValueError, OSError) as exc:
            self._logger.warning(
                "Could not load existing label manifest at %s; "
                "starting fresh: %s",
                csv_path, exc,
            )
            return 0

    def save(
        self,
        output_dir: Path,
        formats:    list[str] | None = None,
    ) -> LabelManifest:
        """
        Write entries to CSV and/or JSON and return a frozen LabelManifest.

        Args:
            output_dir: Root label dataset directory.
            formats:    List of format strings: "csv", "json", or both.
                       Defaults to ["csv", "json"].

        Returns:
            Frozen LabelManifest with entries and the written file paths.
        """
        output_dir = Path(output_dir).resolve()
        formats = [f.lower() for f in (formats or ["csv", "json"])]

        csv_path: Path | None = None
        json_path: Path | None = None

        if "csv" in formats:
            csv_path = output_dir / _LABEL_MANIFEST_CSV_FILENAME
            self._write_csv(csv_path)
        if "json" in formats:
            json_path = output_dir / _LABEL_MANIFEST_JSON_FILENAME
            self._write_json(json_path)

        return LabelManifest(
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
            "Label manifest CSV written: %s (%d entries)",
            path.name, len(self._entries),
        )

    def _write_json(self, path: Path) -> None:
        payload = {
            "version":     LABEL_MANIFEST_SCHEMA_VERSION,
            "entry_count": len(self._entries),
            "entries":     [e.to_dict() for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
        self._logger.info(
            "Label manifest JSON written: %s (%d entries)",
            path.name, len(self._entries),
        )


# ==============================================================================
# Private helpers
# ==============================================================================

def _row_to_entry(row: dict) -> LabelManifestEntry:
    """
    Convert a CSV DictReader row (all-string values) to LabelManifestEntry.

    Uses dict.get() for generation_strategy to remain backward-compatible
    with schema v1.0 CSV files that lack that column.
    """
    return LabelManifestEntry(
        patch_id             = row["patch_id"],
        scene_id              = row["scene_id"],
        patch_path             = row["patch_path"],
        mask_path               = row["mask_path"],
        crs                      = row["crs"],
        width                    = int(row["width"]),
        height                   = int(row["height"]),
        is_valid                 = row["is_valid"].strip().lower() == "true",
        validation_issues         = row["validation_issues"],
        num_classes_present        = int(row["num_classes_present"]),
        valid_pixel_ratio           = float(row["valid_pixel_ratio"]),
        source_type                  = row["source_type"],
        acquisition_date              = row["acquisition_date"],
        year                          = int(row["year"]),
        month                         = int(row["month"]),
        season                        = row["season"],
        hydrological_year              = int(row["hydrological_year"]),
        sensor                        = row["sensor"],
        river_name                     = row["river_name"],
        reach_id                       = row["reach_id"],
        basin_id                       = row["basin_id"],
        aoi_id                         = row["aoi_id"],
        label_version                   = row["label_version"],
        annotator                       = row["annotator"],
        confidence                      = float(row["confidence"]),
        confidence_source                = row["confidence_source"],
        processing_history               = row["processing_history"],
        created_at                       = row["created_at"],
        generation_strategy              = row.get(
            "generation_strategy", _DEFAULT_GENERATION_STRATEGY
        ),
    )


def _dict_to_entry(data: dict) -> LabelManifestEntry:
    """
    Convert a JSON dict to LabelManifestEntry.

    Uses dict.get() for generation_strategy for backward compatibility.
    """
    return LabelManifestEntry(
        patch_id             = data["patch_id"],
        scene_id              = data["scene_id"],
        patch_path             = data["patch_path"],
        mask_path               = data["mask_path"],
        crs                      = data["crs"],
        width                    = int(data["width"]),
        height                   = int(data["height"]),
        is_valid                 = bool(data["is_valid"]),
        validation_issues         = data["validation_issues"],
        num_classes_present        = int(data["num_classes_present"]),
        valid_pixel_ratio           = float(data["valid_pixel_ratio"]),
        source_type                  = data["source_type"],
        acquisition_date              = data["acquisition_date"],
        year                          = int(data["year"]),
        month                         = int(data["month"]),
        season                        = data["season"],
        hydrological_year              = int(data["hydrological_year"]),
        sensor                        = data["sensor"],
        river_name                     = data["river_name"],
        reach_id                       = data["reach_id"],
        basin_id                       = data["basin_id"],
        aoi_id                         = data["aoi_id"],
        label_version                   = data["label_version"],
        annotator                       = data["annotator"],
        confidence                      = float(data["confidence"]),
        confidence_source                = data["confidence_source"],
        processing_history               = data["processing_history"],
        created_at                       = data["created_at"],
        generation_strategy              = data.get(
            "generation_strategy", _DEFAULT_GENERATION_STRATEGY
        ),
    )

# # no change
# """
# Label dataset manifest management for the Label Management pipeline
# (Module 9).

# Mirrors the design of src.export.manifest.DatasetManifestManager (Module 7)
# and src.patches.manifest.PatchManifestManager (Module 8): one
# LabelManifestEntry per processed patch, persisted as both CSV and JSON,
# append-safe across multiple scene-processing runs.
# """

# from __future__ import annotations

# import csv
# import json
# import logging
# from dataclasses import asdict, dataclass, fields
# from pathlib import Path

# from src.core.exceptions import InvalidValueError

# __all__ = [
#     "LabelManifestEntry",
#     "LabelManifest",
#     "LabelManifestManager",
#     "LABEL_MANIFEST_SCHEMA_VERSION",
# ]

# _LOGGER: logging.Logger = logging.getLogger(__name__)
# LABEL_MANIFEST_SCHEMA_VERSION: str = "1.0"

# _LABEL_MANIFEST_CSV_FILENAME:  str = "label_manifest.csv"
# _LABEL_MANIFEST_JSON_FILENAME: str = "label_manifest.json"


# # ==============================================================================
# # LabelManifestEntry
# # ==============================================================================

# @dataclass(frozen=True)
# class LabelManifestEntry:
#     """
#     Immutable record for one processed label.

#     All fields use str, int, float, or bool for lossless CSV round-trips.
#     Optional temporal fields (river_name, reach_id, basin_id) are stored
#     as empty strings when unset.

#     Attributes:
#         patch_id:              Deterministic patch identifier (Module 8).
#         scene_id:                Source scene identifier.
#         patch_path:               Path to the source patch GeoTIFF.
#         mask_path:                Path to the organized label mask GeoTIFF.
#         crs:                      CRS string of the patch.
#         width:                    Patch width in pixels.
#         height:                   Patch height in pixels.
#         is_valid:                 True if the label passed all validation checks.
#         validation_issues:         Semicolon-joined human-readable issue list.
#         num_classes_present:       Distinct valid class IDs found in the mask.
#         valid_pixel_ratio:          Fraction of valid (non-nodata) mask pixels.
#         source_type:                LabelSource implementation identifier
#                                   that discovered this label (e.g. "filesystem").
#         acquisition_date:            Representative imagery date (YYYY-MM-DD).
#         year:                        Calendar year of acquisition_date.
#         month:                       Calendar month of acquisition_date.
#         season:                      Resolved season name.
#         hydrological_year:            Resolved hydrological year.
#         sensor:                      Comma-separated sensor names.
#         river_name:                   River name, or empty string if unset.
#         reach_id:                     River reach identifier, or empty if unset.
#         basin_id:                     Drainage basin identifier, or empty if unset.
#         aoi_id:                       AOI identifier.
#         label_version:                 Label annotation version string.
#         annotator:                     Annotation source identifier.
#         confidence:                    Annotation confidence in [0.0, 1.0].
#         confidence_source:              Origin of the confidence value.
#         processing_history:             Comma-joined processing steps applied.
#         created_at:                     ISO 8601 UTC timestamp of manifest
#                                      entry creation.
#     """

#     patch_id:              str
#     scene_id:               str
#     patch_path:              str
#     mask_path:               str
#     crs:                     str
#     width:                   int
#     height:                  int
#     is_valid:                bool
#     validation_issues:        str
#     num_classes_present:      int
#     valid_pixel_ratio:         float
#     source_type:                str
#     acquisition_date:           str
#     year:                       int
#     month:                      int
#     season:                     str
#     hydrological_year:            int
#     sensor:                     str
#     river_name:                  str
#     reach_id:                    str
#     basin_id:                    str
#     aoi_id:                      str
#     label_version:                str
#     annotator:                    str
#     confidence:                   float
#     confidence_source:             str
#     processing_history:            str
#     created_at:                    str

#     def to_dict(self) -> dict:
#         """Return a plain dict with all fields."""
#         return asdict(self)


# # ==============================================================================
# # LabelManifest (frozen result / data contract)
# # ==============================================================================

# @dataclass(frozen=True)
# class LabelManifest:
#     """
#     Immutable snapshot of the label manifest at a point in time.

#     Produced by LabelManifestManager.save() or from_csv() / from_json().
#     Never constructed directly by external code.

#     Attributes:
#         entries:   Ordered tuple of all LabelManifestEntry records.
#         csv_path:  Absolute path to the written CSV file (None if not saved).
#         json_path: Absolute path to the written JSON file (None if not saved).
#     """

#     entries:   tuple[LabelManifestEntry, ...]
#     csv_path:  Path | None
#     json_path: Path | None

#     @property
#     def entry_count(self) -> int:
#         """Number of label entries in this manifest."""
#         return len(self.entries)

#     @classmethod
#     def from_csv(cls, path: Path) -> LabelManifest:
#         """
#         Load a LabelManifest from a CSV file.

#         Raises:
#             FileNotFoundError: path does not exist.
#             ValueError:        CSV structure is invalid.
#         """
#         path = Path(path).resolve()
#         if not path.exists():
#             raise FileNotFoundError(f"Label manifest CSV not found: {path}")
#         try:
#             entries: list[LabelManifestEntry] = []
#             with open(path, "r", newline="", encoding="utf-8") as fh:
#                 for row in csv.DictReader(fh):
#                     entries.append(_row_to_entry(row))
#         except (csv.Error, KeyError, TypeError) as exc:
#             raise ValueError(
#                 f"CSV manifest {path} has unexpected structure: {exc}"
#             ) from exc
#         _LOGGER.info(
#             "Label manifest CSV loaded: %s (%d entries)", path.name, len(entries)
#         )
#         return cls(entries=tuple(entries), csv_path=path, json_path=None)

#     @classmethod
#     def from_json(cls, path: Path) -> LabelManifest:
#         """
#         Load a LabelManifest from a JSON file.

#         Raises:
#             FileNotFoundError: path does not exist.
#             ValueError:        JSON structure is invalid.
#         """
#         path = Path(path).resolve()
#         if not path.exists():
#             raise FileNotFoundError(f"Label manifest JSON not found: {path}")
#         try:
#             with open(path, "r", encoding="utf-8") as fh:
#                 payload = json.load(fh)
#             entries = tuple(
#                 LabelManifestEntry(**e) for e in payload.get("entries", [])
#             )
#         except (json.JSONDecodeError, TypeError, KeyError) as exc:
#             raise ValueError(
#                 f"JSON manifest {path} has unexpected structure: {exc}"
#             ) from exc
#         _LOGGER.info(
#             "Label manifest JSON loaded: %s (%d entries)", path.name, len(entries)
#         )
#         return cls(entries=entries, csv_path=None, json_path=path)


# # ==============================================================================
# # LabelManifestManager (mutable manager)
# # ==============================================================================

# class LabelManifestManager:
#     """
#     Accumulates LabelManifestEntry records and persists them as LabelManifest.

#     Args:
#         entries: Optional initial entries (e.g. loaded from an existing
#                  label manifest).
#     """

#     _FIELD_NAMES: tuple[str, ...] = tuple(
#         f.name for f in fields(LabelManifestEntry)
#     )

#     def __init__(
#         self,
#         entries: list[LabelManifestEntry] | None = None,
#     ) -> None:
#         self._entries: list[LabelManifestEntry] = list(entries) if entries else []
#         self._logger: logging.Logger = logging.getLogger(__name__)

#     @property
#     def entry_count(self) -> int:
#         """Number of entries currently held by this manager."""
#         return len(self._entries)

#     @property
#     def entries(self) -> tuple[LabelManifestEntry, ...]:
#         """Currently held entries, in insertion order."""
#         return tuple(self._entries)

#     def add_entry(self, entry: LabelManifestEntry) -> None:
#         """
#         Append a LabelManifestEntry to the manager.

#         Raises:
#             InvalidValueError: entry is not a LabelManifestEntry instance.
#         """
#         if not isinstance(entry, LabelManifestEntry):
#             raise InvalidValueError(
#                 field="entry",
#                 value=type(entry).__name__,
#                 reason="must be a LabelManifestEntry instance",
#             )
#         self._entries.append(entry)
#         self._logger.debug("Label manifest entry added: patch_id=%s", entry.patch_id)

#     def load_existing(self, output_dir: Path) -> int:
#         """
#         Load entries from an existing label_manifest.csv at output_dir.

#         Returns:
#             Number of entries loaded (0 if no file found).
#         """
#         csv_path = Path(output_dir) / _LABEL_MANIFEST_CSV_FILENAME
#         if not csv_path.exists():
#             return 0
#         try:
#             manifest = LabelManifest.from_csv(csv_path)
#             self._entries = list(manifest.entries)
#             return len(self._entries)
#         except (ValueError, OSError) as exc:
#             self._logger.warning(
#                 "Could not load existing label manifest at %s; starting fresh: %s",
#                 csv_path, exc,
#             )
#             return 0

#     def save(
#         self,
#         output_dir: Path,
#         formats:    list[str] | None = None,
#     ) -> LabelManifest:
#         """
#         Write entries to CSV and/or JSON and return a frozen LabelManifest.

#         Args:
#             output_dir: Root label dataset directory.
#             formats:    List of format strings: "csv", "json", or both.
#                        Defaults to ["csv", "json"].

#         Returns:
#             Frozen LabelManifest with entries and the written file paths.
#         """
#         output_dir = Path(output_dir).resolve()
#         formats = [f.lower() for f in (formats or ["csv", "json"])]

#         csv_path: Path | None = None
#         json_path: Path | None = None

#         if "csv" in formats:
#             csv_path = output_dir / _LABEL_MANIFEST_CSV_FILENAME
#             self._write_csv(csv_path)
#         if "json" in formats:
#             json_path = output_dir / _LABEL_MANIFEST_JSON_FILENAME
#             self._write_json(json_path)

#         return LabelManifest(
#             entries=tuple(self._entries),
#             csv_path=csv_path,
#             json_path=json_path,
#         )

#     # ------------------------------------------------------------------
#     # Private
#     # ------------------------------------------------------------------

#     def _write_csv(self, path: Path) -> None:
#         with open(path, "w", newline="", encoding="utf-8") as fh:
#             writer = csv.DictWriter(fh, fieldnames=list(self._FIELD_NAMES))
#             writer.writeheader()
#             for entry in self._entries:
#                 writer.writerow(entry.to_dict())
#         self._logger.info(
#             "Label manifest CSV written: %s (%d entries)",
#             path.name, len(self._entries),
#         )

#     def _write_json(self, path: Path) -> None:
#         payload = {
#             "version":     LABEL_MANIFEST_SCHEMA_VERSION,
#             "entry_count": len(self._entries),
#             "entries":     [e.to_dict() for e in self._entries],
#         }
#         with open(path, "w", encoding="utf-8") as fh:
#             json.dump(payload, fh, indent=2, ensure_ascii=True)
#         self._logger.info(
#             "Label manifest JSON written: %s (%d entries)",
#             path.name, len(self._entries),
#         )


# # ==============================================================================
# # Private helpers
# # ==============================================================================

# def _row_to_entry(row: dict) -> LabelManifestEntry:
#     """Convert a CSV DictReader row (all-string values) to LabelManifestEntry."""
#     return LabelManifestEntry(
#         patch_id              = row["patch_id"],
#         scene_id               = row["scene_id"],
#         patch_path              = row["patch_path"],
#         mask_path                = row["mask_path"],
#         crs                       = row["crs"],
#         width                     = int(row["width"]),
#         height                    = int(row["height"]),
#         is_valid                  = row["is_valid"].strip().lower() == "true",
#         validation_issues          = row["validation_issues"],
#         num_classes_present         = int(row["num_classes_present"]),
#         valid_pixel_ratio             = float(row["valid_pixel_ratio"]),
#         source_type                   = row["source_type"],
#         acquisition_date               = row["acquisition_date"],
#         year                            = int(row["year"]),
#         month                           = int(row["month"]),
#         season                          = row["season"],
#         hydrological_year                = int(row["hydrological_year"]),
#         sensor                          = row["sensor"],
#         river_name                       = row["river_name"],
#         reach_id                         = row["reach_id"],
#         basin_id                         = row["basin_id"],
#         aoi_id                           = row["aoi_id"],
#         label_version                     = row["label_version"],
#         annotator                         = row["annotator"],
#         confidence                        = float(row["confidence"]),
#         confidence_source                  = row["confidence_source"],
#         processing_history                 = row["processing_history"],
#         created_at                         = row["created_at"],
#     )