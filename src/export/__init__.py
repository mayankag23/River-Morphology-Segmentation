"""
Dataset export package for the River Morphology Segmentation System.

Transforms a FeatureStackResult (Module 6) into a reproducible, versioned
dataset package on local disk. Every public interface uses immutable typed
dataclasses following the project's data-contract architecture.

Input:   FeatureStackResult  (from src.gee.feature_stack)
Output:  DatasetExportResult (immutable, contains all result paths and metadata)

Data contracts:
    DownloadResult          -- Downloaded image array with georef metadata
    GeoTiffWriteResult      -- Written GeoTIFF path and properties
    GeoTiffValidationResult -- Validation status and any issues found
    SceneMetadata           -- Per-scene provenance and processing record
    ManifestEntry           -- Single scene record in the dataset manifest
    DatasetManifest         -- Complete manifest (entries + file paths)
    VersionInfo             -- Dataset versioning and lineage record
    DatasetExportResult     -- Complete export result (all of the above)

Components:
    EarthEngineDownloader   -- Downloads ee.Image to DownloadResult
    GeoTiffWriter           -- Writes DownloadResult to GeoTiffWriteResult
    GeoTiffValidator        -- Validates a GeoTiffWriteResult
    MetadataWriter          -- Generates and persists SceneMetadata
    DatasetManifestManager  -- Accumulates and persists DatasetManifest
    DatasetVersionManager   -- Generates and persists VersionInfo
    DatasetExporter         -- Orchestrates all components
"""

from src.export.downloader import (
    AffineTransform,
    AoiBounds,
    DownloadResult,
    EarthEngineDownloader,
    TileSpec,
)
from src.export.exporter import DatasetExporter, DatasetExportResult
from src.export.geotiff import (
    GeoTiffProfile,
    GeoTiffValidationResult,
    GeoTiffValidator,
    GeoTiffWriteResult,
    GeoTiffWriter,
)
from src.export.manifest import DatasetManifest, DatasetManifestManager, ManifestEntry
from src.export.metadata import MetadataWriter, SceneMetadata
from src.export.version import DatasetVersionManager, VersionInfo

__all__ = [
    # Downloader
    "AoiBounds",
    "AffineTransform",
    "TileSpec",
    "DownloadResult",
    "EarthEngineDownloader",
    # GeoTIFF
    "GeoTiffProfile",
    "GeoTiffWriteResult",
    "GeoTiffValidationResult",
    "GeoTiffWriter",
    "GeoTiffValidator",
    # Metadata
    "SceneMetadata",
    "MetadataWriter",
    # Manifest
    "ManifestEntry",
    "DatasetManifest",
    "DatasetManifestManager",
    # Version
    "VersionInfo",
    "DatasetVersionManager",
    # Exporter
    "DatasetExportResult",
    "DatasetExporter",
]