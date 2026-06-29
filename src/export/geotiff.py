"""
GeoTIFF writer and validator for the River Morphology Dataset Export pipeline.

GeoTiffWriter converts a DownloadResult into a GeoTIFF file using rasterio.
Band names are stored as rasterio band descriptions so that all GDAL-based
tools can read them. Band order matches DownloadResult.band_names exactly.

GeoTiffValidator reads the written file back and verifies band count, CRS,
and band descriptions. Called after every write to catch silent failures
before reporting success to DatasetExporter.

Design decisions:
    - LZW compression (lossless, fast random access, universal GDAL support).
    - Band descriptions as the canonical band name store (not custom GDAL tags).
    - BigTIFF mode "IF_SAFER" lets GDAL auto-promote for large files.
    - Writer accepts DownloadResult directly (no loose parameters).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import InvalidValueError
from src.export.downloader import AffineTransform, AoiBounds, DownloadResult

__all__ = [
    "GeoTiffProfile",
    "GeoTiffWriteResult",
    "GeoTiffValidationResult",
    "GeoTiffWriter",
    "GeoTiffValidator",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# Immutable configuration and result types
# ==============================================================================

@dataclass(frozen=True)
class GeoTiffProfile:
    """
    Immutable write parameters for GeoTiffWriter.

    Attributes:
        compress:  GDAL compression name ("LZW", "DEFLATE", "NONE").
        tiled:     Write in tiled layout for efficient random access.
        tile_size: Tile width and height in pixels.
        dtype:     numpy/rasterio dtype string for the output bands.
        bigtiff:   "IF_SAFER" lets GDAL choose based on projected file size.
        overviews: Build pyramid overviews for fast display at low zoom.
    """

    compress:  str  = "LZW"
    tiled:     bool = True
    tile_size: int  = 256
    dtype:     str  = "float32"
    bigtiff:   str  = "IF_SAFER"
    overviews: bool = True


@dataclass(frozen=True)
class GeoTiffWriteResult:
    """
    Immutable result of GeoTiffWriter.write().

    Attributes:
        path:           Absolute path to the written GeoTIFF.
        num_bands:      Number of bands in the file.
        width:          Raster width in pixels.
        height:         Raster height in pixels.
        crs:            CRS string of the written raster.
        transform:      Affine coefficients as AffineTransform.
        band_names:     Ordered band names stored in file descriptions.
        file_size_bytes: File size on disk in bytes.
        compress:       Compression method applied.
    """

    path:            Path
    num_bands:       int
    width:           int
    height:          int
    crs:             str
    transform:       AffineTransform
    band_names:      tuple[str, ...]
    file_size_bytes: int
    compress:        str


@dataclass(frozen=True)
class GeoTiffValidationResult:
    """
    Immutable result of GeoTiffValidator.validate().

    Attributes:
        is_valid:    True if all checks passed.
        path:        Absolute path to the validated file.
        band_count:  Actual band count (-1 if unreadable).
        crs:         CRS string from the file.
        width:       Width in pixels (-1 if unreadable).
        height:      Height in pixels (-1 if unreadable).
        band_names:  Band descriptions read from the file.
        issues:      Ordered tuple of human-readable problem descriptions.
    """

    is_valid:   bool
    path:       Path
    band_count: int
    crs:        str
    width:      int
    height:     int
    band_names: tuple[str, ...]
    issues:     tuple[str, ...]

    def summary(self) -> str:
        """Return a single ASCII status line."""
        tag = "[OK]  " if self.is_valid else "[FAIL]"
        return (
            f"{tag} {self.path.name}: "
            f"bands={self.band_count}, "
            f"size={self.width}x{self.height}, "
            f"crs={self.crs}"
        )


# ==============================================================================
# GeoTiffWriter
# ==============================================================================

class GeoTiffWriter:
    """
    Writes a DownloadResult to a GeoTIFF file using rasterio.

    Band names are stored in rasterio band descriptions (portable across
    GDAL tools). Band order matches DownloadResult.band_names exactly and
    is never reordered.

    Args:
        profile: Write parameters. Defaults to LZW-compressed tiled GeoTIFF.
    """

    def __init__(self, profile: GeoTiffProfile | None = None) -> None:
        self._profile = profile or GeoTiffProfile()
        self._logger: logging.Logger = logging.getLogger(__name__)

    def write(
        self,
        download_result: DownloadResult,
        output_path:     Path,
    ) -> GeoTiffWriteResult:
        """
        Write a DownloadResult to a GeoTIFF file.

        Args:
            download_result: DownloadResult from EarthEngineDownloader.download().
            output_path:     Destination path. Parent directory must exist.

        Returns:
            GeoTiffWriteResult with the path and file properties.

        Raises:
            InvalidValueError: download_result.data has unexpected shape.
            OSError:           rasterio cannot write to output_path.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError as exc:
            raise OSError(
                "rasterio is not installed. "
                "Install with: pip install rasterio==1.3.9"
            ) from exc

        data       = download_result.data
        band_names = list(download_result.band_names)
        transform  = download_result.transform.to_affine()

        self._validate_array(data, band_names)

        n_bands, height, width = data.shape
        output_path = Path(output_path).resolve()

        rasterio_profile: dict[str, Any] = {
            "driver":    "GTiff",
            "dtype":     self._profile.dtype,
            "width":     width,
            "height":    height,
            "count":     n_bands,
            "crs":       download_result.crs,
            "transform": transform,
            "compress":  self._profile.compress,
            "bigtiff":   self._profile.bigtiff,
        }
        if self._profile.tiled:
            rasterio_profile.update({
                "tiled":     True,
                "blockxsize": self._profile.tile_size,
                "blockysize": self._profile.tile_size,
            })

        self._logger.debug(
            "Writing GeoTIFF: %s (%d bands, %dx%d, crs=%s)",
            output_path.name, n_bands, width, height, download_result.crs,
        )

        try:
            with rasterio.open(output_path, "w", **rasterio_profile) as ds:
                for idx, name in enumerate(band_names, start=1):
                    ds.write(data[idx - 1].astype(np.float32), idx)
                    ds.update_tags(idx, name=name)
                    ds.set_band_description(idx, name)

            if self._profile.overviews:
                self._build_overviews(output_path)

        except rasterio.errors.RasterioIOError as exc:
            raise OSError(
                f"rasterio failed to write {output_path}: {exc}"
            ) from exc

        file_size = output_path.stat().st_size
        self._logger.info(
            "GeoTIFF written: %s (%.1f MB)",
            output_path.name,
            file_size / (1024 * 1024),
        )

        return GeoTiffWriteResult(
            path=output_path,
            num_bands=n_bands,
            width=width,
            height=height,
            crs=download_result.crs,
            transform=download_result.transform,
            band_names=tuple(band_names),
            file_size_bytes=file_size,
            compress=self._profile.compress,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_array(data: Any, band_names: list[str]) -> None:
        """Raise InvalidValueError if array shape or band names are inconsistent."""
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise InvalidValueError(
                field="download_result.data",
                value=type(data).__name__,
                reason="data must be a 3-D numpy array (bands, height, width)",
            )
        if len(band_names) != data.shape[0]:
            raise InvalidValueError(
                field="download_result.band_names",
                value=len(band_names),
                reason=(
                    f"len(band_names) = {len(band_names)} must equal "
                    f"data.shape[0] = {data.shape[0]}"
                ),
            )

    def _build_overviews(self, path: Path) -> None:
        """Build pyramid overviews for fast low-zoom display."""
        try:
            import rasterio
            from rasterio.enums import Resampling
            with rasterio.open(path, "r+") as ds:
                ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
                ds.update_tags(ns="rio_overview", resampling="average")
        except Exception as exc:
            self._logger.warning("Overview generation failed (non-fatal): %s", exc)


# ==============================================================================
# GeoTiffValidator
# ==============================================================================

class GeoTiffValidator:
    """
    Validates an exported GeoTIFF by reading it back with rasterio.

    Checks performed:
        - File exists and has non-zero size.
        - File can be opened by rasterio.
        - Band count matches GeoTiffWriteResult.num_bands.
        - CRS matches GeoTiffWriteResult.crs (when expected_crs provided).
        - Band descriptions match GeoTiffWriteResult.band_names.
    """

    def validate(
        self,
        write_result:         GeoTiffWriteResult,
        expected_band_names:  tuple[str, ...] | None = None,
        expected_crs:         str | None             = None,
    ) -> GeoTiffValidationResult:
        """
        Validate the file recorded in write_result.

        Args:
            write_result:        Result from GeoTiffWriter.write().
            expected_band_names: Override expected band names. Uses
                                 write_result.band_names when None.
            expected_crs:        Override expected CRS. Uses write_result.crs
                                 when None.

        Returns:
            GeoTiffValidationResult with is_valid flag and issue descriptions.
        """
        expected_names = expected_band_names or write_result.band_names
        expected_crs_s = expected_crs        or write_result.crs
        path           = write_result.path

        issues: list[str] = []
        band_count = -1
        crs        = ""
        width      = -1
        height     = -1
        read_names: tuple[str, ...] = ()

        if not path.exists():
            issues.append(f"File does not exist: {path}")
            return self._make_result(
                False, path, band_count, crs, width, height, read_names, issues
            )

        if path.stat().st_size == 0:
            issues.append(f"File is empty (0 bytes): {path}")
            return self._make_result(
                False, path, band_count, crs, width, height, read_names, issues
            )

        try:
            import rasterio
        except ImportError:
            issues.append("rasterio is not installed; cannot validate GeoTIFF.")
            return self._make_result(
                False, path, band_count, crs, width, height, read_names, issues
            )

        try:
            with rasterio.open(path) as ds:
                band_count = ds.count
                crs        = ds.crs.to_string() if ds.crs else ""
                width      = ds.width
                height     = ds.height
                read_names = tuple(
                    ds.descriptions[i] or f"band_{i + 1}"
                    for i in range(ds.count)
                )
        except rasterio.errors.RasterioIOError as exc:
            issues.append(f"Cannot open file: {exc}")
            return self._make_result(
                False, path, band_count, crs, width, height, read_names, issues
            )
        except Exception as exc:
            issues.append(f"Unexpected error reading file: {exc}")
            return self._make_result(
                False, path, band_count, crs, width, height, read_names, issues
            )

        if band_count != len(expected_names):
            issues.append(
                f"Band count: expected {len(expected_names)}, got {band_count}."
            )

        if crs != expected_crs_s:
            issues.append(
                f"CRS: expected '{expected_crs_s}', got '{crs}'."
            )

        for i, exp in enumerate(expected_names):
            actual = read_names[i] if i < len(read_names) else ""
            if actual != exp:
                issues.append(
                    f"Band {i + 1} name: expected '{exp}', got '{actual}'."
                )

        return self._make_result(
            len(issues) == 0,
            path, band_count, crs, width, height, read_names, issues,
        )

    @staticmethod
    def _make_result(
        is_valid:   bool,
        path:       Path,
        band_count: int,
        crs:        str,
        width:      int,
        height:     int,
        band_names: tuple[str, ...],
        issues:     list[str],
    ) -> GeoTiffValidationResult:
        return GeoTiffValidationResult(
            is_valid=is_valid,
            path=path,
            band_count=band_count,
            crs=crs,
            width=width,
            height=height,
            band_names=band_names,
            issues=tuple(issues),
        )