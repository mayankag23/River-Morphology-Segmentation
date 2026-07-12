"""
Patch generation orchestrator for the River Morphology Segmentation System.

PatchGenerator is the single entry point for transforming an exported scene
(DatasetExportResult from Module 7) into a directory of fixed-size,
georeferenced patches plus a patch_manifest.

Responsibility: orchestrate PatchTiler, PatchReader, and PatchValidator.
GeoTiffWriter from Module 7 (src.export.geotiff) is REUSED for all patch
GeoTIFF output -- this module contains no GeoTIFF writing logic of its own.

Output layout:
    {output_dir}/
        patch_manifest.csv
        patch_manifest.json
        scenes/
            {scene_id}/
                patches/
                    {scene_id}_r000_c000.tif
                    {scene_id}_r000_c001.tif
                    ...

Input contract:  DatasetExportResult (Module 7)
Output contract: PatchDatasetResult (immutable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import InvalidValueError
from src.export.downloader import AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.patches.manifest import (
    PatchManifest,
    PatchManifestEntry,
    PatchManifestManager,
)
from src.patches.reader import PatchReader
from src.patches.tiler import PatchTiler, PatchWindow
from src.patches.validator import PatchValidator

if TYPE_CHECKING:
    from src.export.exporter import DatasetExportResult

__all__ = ["PatchDatasetResult", "PatchGenerator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_SCENES_SUBDIR:  str = "scenes"
_PATCHES_SUBDIR: str = "patches"


# ==============================================================================
# PatchDatasetResult
# ==============================================================================

@dataclass(frozen=True)
class PatchDatasetResult:
    """
    Immutable output of PatchGenerator.generate().

    Attributes:
        scene_id:          Source scene identifier.
        output_dir:         Root directory for the patch dataset.
        scene_patches_dir:  Directory containing this scene's patch files.
        manifest:           Frozen PatchManifest snapshot after this run.
        patches_generated:  Number of patches successfully written.
        patches_skipped:    Number of candidate windows rejected by validation.
        total_windows:      Total candidate windows computed by the tiler.
        patch_size:         Patch side length in pixels used for this run.
        stride:             Stride in pixels used for this run.
        operations_log:     Ordered tuple of operation descriptions.
    """

    scene_id:          str
    output_dir:        Path
    scene_patches_dir: Path
    manifest:           PatchManifest
    patches_generated:  int
    patches_skipped:    int
    total_windows:      int
    patch_size:         int
    stride:             int
    operations_log:     tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines for logging and display."""
        return [
            f"  scene_id:          {self.scene_id}",
            f"  patches generated: {self.patches_generated}",
            f"  patches skipped:   {self.patches_skipped}",
            f"  total windows:     {self.total_windows}",
            f"  patch size:        {self.patch_size}",
            f"  stride:            {self.stride}",
        ]


# ==============================================================================
# PatchGenerator
# ==============================================================================

class PatchGenerator:
    """
    Generates fixed-size georeferenced patches from an exported scene GeoTIFF.

    Reuses GeoTiffWriter from Module 7 for all patch GeoTIFF output -- never
    writes GeoTIFFs manually. Reuses SceneMetadata (CRS, band names) from the
    input DatasetExportResult rather than re-reading that information from
    the file.

    All patch parameters (patch_size, stride, nodata_value,
    min_valid_pixel_ratio) come from config.patch_generation. GeoTIFF write
    parameters come from config.export.geotiff. Manifest formats come from
    config.export.manifest.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        pg = config.patch_generation
        self._patch_size            = int(pg.patch_size)
        self._stride                = int(getattr(pg, "train_stride", pg.patch_size))
        self._nodata_value          = float(pg.nodata_value)
        self._min_valid_pixel_ratio = float(pg.min_valid_pixel_ratio)

        self._tiler     = PatchTiler(self._patch_size, self._stride)
        self._validator = PatchValidator(
            nodata_value=self._nodata_value,
            min_valid_pixel_ratio=self._min_valid_pixel_ratio,
        )
        self._writer = GeoTiffWriter(self._build_geotiff_profile())

        self._logger.debug(
            "PatchGenerator initialized. patch_size=%d, stride=%d, "
            "min_valid_pixel_ratio=%.2f",
            self._patch_size, self._stride, self._min_valid_pixel_ratio,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        dataset_export_result: DatasetExportResult,
        output_dir:            Path,
        stride:                int | None = None,
        append_to_manifest:    bool       = True,
    ) -> PatchDatasetResult:
        """
        Generate patches for one exported scene.

        Sequence:
            1. Validate inputs (source band names present).
            2. Create directory tree: output_dir/scenes/{scene_id}/patches/.
            3. Open the source GeoTIFF and compute the patch window grid.
            4. For each window: read, validate, write (if valid), record entry.
            5. Persist patch_manifest.csv / patch_manifest.json.
            6. Return PatchDatasetResult.

        Args:
            dataset_export_result: DatasetExportResult from Module 7's
                                   DatasetExporter.export().
            output_dir:             Root directory for the patch dataset.
                                    Created if absent.
            stride:                 Optional override for stride (pixels).
                                    Uses config.patch_generation.train_stride
                                    by default.
            append_to_manifest:     Load and extend an existing patch
                                    manifest when True; start fresh when False.

        Returns:
            Frozen PatchDatasetResult describing all generated patches.

        Raises:
            InvalidValueError: Source band names are empty, or stride
                               override is not a positive integer.
            OSError:            Source GeoTIFF cannot be opened.
        """
        scene_id   = dataset_export_result.scene_id
        scene_meta = dataset_export_result.scene_metadata
        image_path = dataset_export_result.image_path

        self._validate_inputs(scene_meta)

        tiler             = self._tiler if stride is None else PatchTiler(self._patch_size, stride)
        effective_stride  = tiler.stride

        output_dir         = Path(output_dir).resolve()
        scene_patches_dir  = output_dir / _SCENES_SUBDIR / scene_id / _PATCHES_SUBDIR
        scene_patches_dir.mkdir(parents=True, exist_ok=True)

        operations: list[str] = []
        self._logger.info(
            "Generating patches. scene_id=%s, patch_size=%d, stride=%d",
            scene_id, self._patch_size, effective_stride,
        )

        generated = 0
        skipped   = 0

        # Keep the persistent manifest cumulative, but return only entries
        # generated for this scene to downstream pipeline stages.
        current_scene_entries: list[PatchManifestEntry] = []

        with PatchReader(image_path) as reader:
            windows = tiler.compute_windows(reader.width, reader.height)
            operations.append(f"tiling: {len(windows)} candidate window(s)")

            manifest_manager = PatchManifestManager()
            if append_to_manifest:
                manifest_manager.load_existing(output_dir)

            for window in windows:
                patch_data, patch_transform = reader.read_window(window)
                validation = self._validator.validate(patch_data)

                if not validation.is_valid:
                    skipped += 1
                    continue

                patch_id   = self._build_patch_id(scene_id, window)
                patch_path = scene_patches_dir / f"{patch_id}.tif"

                download_result = DownloadResult(
                    data=patch_data,
                    crs=scene_meta.crs,
                    transform=patch_transform,
                    band_names=tuple(scene_meta.band_names),
                    width=window.width,
                    height=window.height,
                    aoi_bounds=self._estimate_patch_bounds(patch_transform, window),
                    num_tiles=1,
                )
                write_result = self._writer.write(download_result, patch_path)

                entry = PatchManifestEntry(
                    patch_id=patch_id,
                    scene_id=scene_id,
                    source_image_path=str(image_path),
                    patch_path=str(write_result.path),
                    row_index=window.row_index,
                    col_index=window.col_index,
                    row_off=window.row_off,
                    col_off=window.col_off,
                    height=window.height,
                    width=window.width,
                    num_bands=write_result.num_bands,
                    crs=write_result.crs,
                    valid_pixel_ratio=validation.valid_pixel_ratio,
                    file_size_bytes=write_result.file_size_bytes,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                manifest_manager.add_entry(entry)
                current_scene_entries.append(entry)
                generated += 1

        operations.append(f"generated: {generated}, skipped: {skipped}")

        manifest_formats = self._read_manifest_formats()

        # Persist the complete historical + current manifest.
        persisted_manifest = manifest_manager.save(
            output_dir,
            formats=manifest_formats,
        )
        operations.append(
            f"write_manifest: {persisted_manifest.entry_count} total entries"
        )

        # Downstream stages must process only this generation call's patches.
        manifest = PatchManifest(
            entries=tuple(current_scene_entries),
            csv_path=persisted_manifest.csv_path,
            json_path=persisted_manifest.json_path,
        )
        operations.append(
            f"return_manifest: {manifest.entry_count} current-scene entries"
        )

        result = PatchDatasetResult(
            scene_id=scene_id,
            output_dir=output_dir,
            scene_patches_dir=scene_patches_dir,
            manifest=manifest,
            patches_generated=generated,
            patches_skipped=skipped,
            total_windows=len(windows),
            patch_size=self._patch_size,
            stride=effective_stride,
            operations_log=tuple(operations),
        )

        for line in result.summary_lines():
            self._logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, scene_meta: Any) -> None:
        """Raise if the source scene metadata is unsuitable for patching."""
        band_names = getattr(scene_meta, "band_names", ())
        if not band_names:
            raise InvalidValueError(
                field="dataset_export_result.scene_metadata.band_names",
                value=band_names,
                reason="must be non-empty to generate patches",
            )

    def _build_patch_id(self, scene_id: str, window: PatchWindow) -> str:
        """
        Build a deterministic patch identifier.

        Format: "{scene_id}_r{row_index:03d}_c{col_index:03d}".
        Uses grid row/col indices, not pixel offsets or sequential counters,
        so the same window always produces the same ID across runs.
        """
        return f"{scene_id}_r{window.row_index:03d}_c{window.col_index:03d}"

    def _estimate_patch_bounds(self, transform: Any, window: PatchWindow) -> AoiBounds:
        """
        Compute the patch's bounding box from its affine transform.

        Applies the affine transform to the window's top-left and
        bottom-right pixel corners to derive an extent box. Field names
        follow AoiBounds (min_lon, min_lat, etc.) for type-contract
        consistency with Module 7; values are expressed in the raster's
        native CRS units, which are decimal degrees for the project's
        default geographic CRS (EPSG:4326).
        """
        x0 = transform.c
        y0 = transform.f
        x1 = transform.c + transform.a * window.width
        y1 = transform.f + transform.e * window.height
        return AoiBounds(
            min_lon=min(x0, x1),
            min_lat=min(y0, y1),
            max_lon=max(x0, x1),
            max_lat=max(y0, y1),
        )

    def _build_geotiff_profile(self) -> GeoTiffProfile:
        """Read GeoTIFF write settings from config.export.geotiff."""
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

    def _read_manifest_formats(self) -> list[str]:
        """Read manifest output format(s) from config.export.manifest."""
        manifest_cfg = getattr(getattr(self._config, "export", None), "manifest", None)
        raw          = getattr(manifest_cfg, "formats", ["csv", "json"])
        return [str(f).lower() for f in raw]
    