"""
Dataset export orchestrator for the River Morphology Segmentation System.

DatasetExporter is the single entry point for exporting a FeatureStackResult
to a versioned, self-describing dataset package on local disk.

Responsibility:
    Orchestrate five independent components. DatasetExporter contains no I/O,
    EE, or rasterio logic of its own. Every specific operation is delegated.

Components:
    EarthEngineDownloader  -> DownloadResult
    GeoTiffWriter          -> GeoTiffWriteResult
    GeoTiffValidator       -> GeoTiffValidationResult
    MetadataWriter         -> SceneMetadata
    DatasetManifestManager -> DatasetManifest
    DatasetVersionManager  -> VersionInfo

Output layout:
    {output_dir}/
        version.json
        manifest.csv
        manifest.json
        scenes/
            {scene_id}/
                image.tif
                metadata.json

Input contract:  FeatureStackResult (from src.gee.feature_stack)
Output contract: DatasetExportResult (immutable, contains all paths and results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError, MissingFieldError
from src.export.downloader import AoiBounds, EarthEngineDownloader
from src.export.geotiff import (
    GeoTiffProfile,
    GeoTiffValidationResult,
    GeoTiffValidator,
    GeoTiffWriteResult,
    GeoTiffWriter,
)
from src.export.manifest import (
    DatasetManifest,
    DatasetManifestManager,
    ManifestEntry,
)
from src.export.metadata import MetadataWriter, SceneMetadata
from src.export.version import DatasetVersionManager, VersionInfo

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient
    from src.gee.feature_stack import FeatureStackResult

__all__ = [
    "DatasetExportResult",
    "DatasetExporter",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_SCENES_SUBDIR:       str = "scenes"
_IMAGE_FILENAME:      str = "image.tif"
_METADATA_FILENAME:   str = "metadata.json"


# ==============================================================================
# DatasetExportResult
# ==============================================================================

@dataclass(frozen=True)
class DatasetExportResult:
    """
    Immutable output of DatasetExporter.export().

    Contains every result produced during the export: paths to output files,
    structured metadata, validation status, and back-references to the typed
    result objects from each component.

    Attributes:
        scene_id:         Unique scene identifier used for directory naming.
        dataset_root:     Absolute path to the root dataset directory.
        scenes_dir:       Absolute path to the scenes/ subdirectory.
        scene_dir:        Absolute path to this scene's subdirectory.
        image_path:       Absolute path to the GeoTIFF (image.tif).
        metadata_path:    Absolute path to the scene metadata JSON.
        version_path:     Absolute path to version.json.
        export_timestamp: ISO 8601 UTC timestamp of this export.
        scene_metadata:   Immutable SceneMetadata record for this scene.
        version_info:     Immutable VersionInfo for the dataset.
        manifest:         Frozen DatasetManifest snapshot after this export.
        write_result:     Immutable GeoTiffWriteResult from GeoTiffWriter.
        validation:       Immutable GeoTiffValidationResult from GeoTiffValidator.
        operations_log:   Ordered tuple of operation descriptions.
        is_valid:         True if GeoTiffValidator found no issues.
    """

    scene_id:         str
    dataset_root:     Path
    scenes_dir:       Path
    scene_dir:        Path
    image_path:       Path
    metadata_path:    Path
    version_path:     Path
    export_timestamp: str
    scene_metadata:   SceneMetadata
    version_info:     VersionInfo
    manifest:         DatasetManifest
    write_result:     GeoTiffWriteResult
    validation:       GeoTiffValidationResult
    operations_log:   tuple[str, ...]
    is_valid:         bool

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines for logging and display."""
        status = "[OK]  " if self.is_valid else "[FAIL]"
        return [
            f"  {status}  scene_id:   {self.scene_id}",
            f"         root:       {self.dataset_root}",
            f"         image:      {self.image_path.name}",
            f"         bands:      {self.scene_metadata.num_bands}",
            f"         size:       {self.write_result.width}x{self.write_result.height} px",
            f"         crs:        {self.scene_metadata.crs}",
            f"         file:       {self.write_result.file_size_bytes // 1024 // 1024} MB",
        ]


# ==============================================================================
# DatasetExporter
# ==============================================================================

class DatasetExporter:
    """
    Exports a FeatureStackResult to a versioned dataset package on local disk.

    Reads all parameters from Config. Constructs and coordinates the five
    independent components. Contains no I/O or EE logic itself.

    Args:
        client: Initialized EarthEngineClient.
        config: Fully initialized Config with AOI and satellite settings.
    """

    def __init__(
        self,
        client: EarthEngineClient,
        config: Config,
    ) -> None:
        self._client = client
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        # Build components once at construction time.
        geotiff_profile      = self._build_geotiff_profile()
        max_tile_pixels      = self._read_max_tile_pixels()

        self._downloader = EarthEngineDownloader(client, max_tile_pixels)
        self._writer     = GeoTiffWriter(geotiff_profile)
        self._validator  = GeoTiffValidator()
        self._meta_writer = MetadataWriter(config)
        self._version_mgr = DatasetVersionManager(config)

        self._logger.debug(
            "DatasetExporter initialized. max_tile_pixels=%d", max_tile_pixels
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        feature_stack_result: FeatureStackResult,
        output_dir:           Path,
        scene_id:             str | None  = None,
        append_to_manifest:   bool        = True,
    ) -> DatasetExportResult:
        """
        Export the feature stack to a local dataset package.

        Execution sequence:
            1. Validate inputs (AOI set, band names present).
            2. Create directory tree: output_dir/scenes/{scene_id}/.
            3. Download image from GEE -> DownloadResult.
            4. Write GeoTIFF -> GeoTiffWriteResult.
            5. Generate and save SceneMetadata.
            6. Validate GeoTIFF -> GeoTiffValidationResult.
            7. Generate and save VersionInfo -> version.json.
            8. Add ManifestEntry -> save DatasetManifest.
            9. Return DatasetExportResult.

        Args:
            feature_stack_result: From SpectralFeatureGenerator.generate().
            output_dir:           Root dataset directory. Created if absent.
            scene_id:             Optional. Auto-generated from UTC timestamp
                                  if not provided.
            append_to_manifest:   Load and extend existing manifest when True.
                                  Start a new manifest when False.

        Returns:
            Frozen DatasetExportResult with all result objects and paths.

        Raises:
            MissingFieldError:  AOI not configured or band names absent.
            InvalidValueError:  Input validation failed.
            OSError:            File I/O failure.
        """
        self._validate_inputs(feature_stack_result)

        scene_id    = scene_id or self._generate_scene_id()
        output_dir  = Path(output_dir).resolve()
        scenes_dir  = output_dir / _SCENES_SUBDIR
        scene_dir   = scenes_dir / scene_id
        image_path  = scene_dir / _IMAGE_FILENAME
        meta_path   = scene_dir / _METADATA_FILENAME

        scene_dir.mkdir(parents=True, exist_ok=True)

        timestamp  = datetime.now(timezone.utc).isoformat()
        operations: list[str] = []

        # Step 3: Download from GEE.
        aoi_bounds   = self._get_aoi_bounds()
        band_names   = list(feature_stack_result.all_band_names)
        scale        = self._get_scale()
        crs          = self._get_crs()

        self._logger.info(
            "Starting export. scene_id=%s, bands=%d, scale=%gm",
            scene_id, len(band_names), scale,
        )


        print("=" * 80)
        print("EE IMAGE BANDS")
        print(feature_stack_result.image.bandNames().getInfo())
        print()

        print("FEATURE STACK BAND NAMES")
        print(feature_stack_result.all_band_names)
        print()

        print("NUMBER OF EE BANDS")
        print(feature_stack_result.image.bandNames().size().getInfo())
        print("=" * 80)

        download_result = self._downloader.download(
            image=feature_stack_result.image,
            aoi_bounds=aoi_bounds,
            band_names=band_names,
            scale_meters=scale,
            crs=crs,
        )
        operations.append(
            f"download: {download_result.num_tiles} tile(s), "
            f"{download_result.width}x{download_result.height}px"
        )
        self._logger.info(
            "Download complete: %dx%d px, %d tile(s).",
            download_result.width, download_result.height, download_result.num_tiles,
        )

        # Step 4: Write GeoTIFF.
        write_result = self._writer.write(download_result, image_path)
        operations.append(f"write_geotiff: {image_path.name}")
        self._logger.info(
            "GeoTIFF written: %s (%.1f MB)",
            image_path.name, write_result.file_size_bytes / 1024 / 1024,
        )

        # Step 5: Generate and save metadata.
        scene_metadata = self._meta_writer.generate(
            scene_id=scene_id,
            feature_stack_result=feature_stack_result,
            download_result=download_result,
        )
        self._meta_writer.save(scene_metadata, meta_path)
        operations.append(f"write_metadata: {meta_path.name}")

        # Step 6: Validate GeoTIFF.
        validation = self._validator.validate(write_result)
        operations.append(
            f"validate: {'ok' if validation.is_valid else 'FAILED'}"
        )
        if not validation.is_valid:
            for issue in validation.issues:
                self._logger.error("Validation issue: %s", issue)
        else:
            self._logger.info("GeoTIFF validation passed.")

        # Step 7: Version info.
        version_info  = self._version_mgr.generate()
        version_path  = self._version_mgr.save(version_info, output_dir)
        operations.append(f"write_version: {version_path.name}")

        # Step 8: Update manifest.
        manifest_formats = self._read_manifest_formats()
        manifest_manager = DatasetManifestManager()
        if append_to_manifest:
            manifest_manager.load_existing(output_dir)

        entry = ManifestEntry(
            scene_id             = scene_id,
            image_path           = str(image_path),
            metadata_path        = str(meta_path),
            export_timestamp     = scene_metadata.export_timestamp,
            num_bands            = scene_metadata.num_bands,
            width                = download_result.width,
            height               = download_result.height,
            crs                  = scene_metadata.crs,
            start_date           = scene_metadata.start_date,
            end_date             = scene_metadata.end_date,
            sensors              = ",".join(scene_metadata.sensors),
            composite_method     = scene_metadata.composite_method,
            cloud_cover_limit    = scene_metadata.cloud_cover_limit,
            num_spectral_indices = len(scene_metadata.spectral_indices),
            file_size_bytes      = write_result.file_size_bytes,
            is_valid             = validation.is_valid,
        )
        manifest_manager.add_entry(entry)
        manifest = manifest_manager.save(output_dir, formats=manifest_formats)
        operations.append(
            f"write_manifest: {manifest.entry_count} total entries"
        )

        result = DatasetExportResult(
            scene_id=scene_id,
            dataset_root=output_dir,
            scenes_dir=scenes_dir,
            scene_dir=scene_dir,
            image_path=image_path,
            metadata_path=meta_path,
            version_path=version_path,
            export_timestamp=timestamp,
            scene_metadata=scene_metadata,
            version_info=version_info,
            manifest=manifest,
            write_result=write_result,
            validation=validation,
            operations_log=tuple(operations),
            is_valid=validation.is_valid,
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, feature_stack_result: FeatureStackResult) -> None:
        """Raise if config or feature stack state prevents a successful export."""
        if not self._config.has_aoi:
            raise MissingFieldError(
                field="aoi.[min_lon, min_lat, max_lon, max_lat]",
                context=(
                    "Set all four AOI coordinates in config.yaml before "
                    "calling DatasetExporter.export()."
                ),
            )
        band_names = getattr(feature_stack_result, "all_band_names", ())
        if not band_names:
            raise InvalidValueError(
                field="feature_stack_result.all_band_names",
                value=band_names,
                reason=(
                    "all_band_names is empty. Ensure SpectralFeatureGenerator "
                    "ran with at least one enabled index and that "
                    "LandsatPreprocessor ran with apply_harmonization=True."
                ),
            )

    def _generate_scene_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"RM_export_{ts}"

    def _get_aoi_bounds(self) -> AoiBounds:
        aoi = self._config.aoi
        return AoiBounds(
            min_lon=float(aoi.min_lon),
            min_lat=float(aoi.min_lat),
            max_lon=float(aoi.max_lon),
            max_lat=float(aoi.max_lat),
        )

    def _get_scale(self) -> float:
        return float(
            getattr(getattr(self._config, "satellite", None), "resolution_meters", 30.0)
        )

    def _get_crs(self) -> str:
        return str(
            getattr(getattr(self._config, "satellite", None), "output_crs", "EPSG:4326")
        )

    def _build_geotiff_profile(self) -> GeoTiffProfile:
        geotiff_cfg = getattr(getattr(self._config, "export", None), "geotiff", None)
        if geotiff_cfg is None:
            return GeoTiffProfile()
        return GeoTiffProfile(
            compress  = str(getattr(geotiff_cfg, "compress",  "LZW")),
            tiled     = bool(getattr(geotiff_cfg, "tiled",    True)),
            tile_size = int(getattr(geotiff_cfg,  "tile_size", 256)),
            dtype     = str(getattr(geotiff_cfg,  "dtype",    "float32")),
            overviews = bool(getattr(geotiff_cfg, "overviews", True)),
        )

    def _read_max_tile_pixels(self) -> int:
        return int(getattr(getattr(self._config, "export", None), "max_tile_pixels", 1_000_000))

    def _read_manifest_formats(self) -> list[str]:
        manifest_cfg = getattr(getattr(self._config, "export", None), "manifest", None)
        raw          = getattr(manifest_cfg, "formats", ["csv", "json"])
        return [str(f).lower() for f in raw]