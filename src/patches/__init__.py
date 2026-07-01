"""
Patch generation package for the River Morphology Segmentation System
(Module 8).

Transforms a DatasetExportResult (Module 7) into a directory of fixed-size,
georeferenced patches plus a patch_manifest. Every public interface uses
immutable typed dataclasses, following the project's data-contract
architecture established in Module 7.

Input:   DatasetExportResult  (from src.export.exporter)
Output:  PatchDatasetResult   (immutable)

Components:
    PatchTiler             -- Computes the deterministic patch window grid.
    PatchReader             -- Reads pixel windows from a source GeoTIFF via rasterio.
    PatchValidator           -- Computes valid-pixel ratio and accept/reject decision.
    PatchManifestEntry       -- Frozen single-patch record.
    PatchManifest            -- Frozen manifest snapshot.
    PatchManifestManager     -- Accumulates and persists patch records.
    PatchDatasetResult       -- Frozen orchestration result.
    PatchGenerator           -- Orchestrates all of the above. Reuses
                                GeoTiffWriter from src.export.geotiff
                                (Module 7) for all patch GeoTIFF output.
"""

from src.patches.generator import PatchDatasetResult, PatchGenerator
from src.patches.manifest import (
    PATCH_MANIFEST_SCHEMA_VERSION,
    PatchManifest,
    PatchManifestEntry,
    PatchManifestManager,
)
from src.patches.reader import PatchReader
from src.patches.tiler import PatchTiler, PatchWindow
from src.patches.validator import PatchValidationResult, PatchValidator

__all__ = [
    "PatchWindow",
    "PatchTiler",
    "PatchReader",
    "PatchValidationResult",
    "PatchValidator",
    "PatchManifestEntry",
    "PatchManifest",
    "PatchManifestManager",
    "PATCH_MANIFEST_SCHEMA_VERSION",
    "PatchDatasetResult",
    "PatchGenerator",
]