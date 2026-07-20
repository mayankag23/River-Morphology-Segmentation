"""
Earth Engine image downloader for the River Morphology Dataset Export pipeline.

Architectural note:
    EarthEngineDownloader lives in src/export/ (not src/gee/) because the
    Earth Engine processing pipeline ends at Module 6. Module 7 is in the
    export layer and treats EE only as a data source.

    All EE interaction occurs through EarthEngineClient.execute_with_retry().
    No 'import ee' statement appears in this module. The image parameter is
    typed as Any (it is an ee.Image created by src/gee modules) and its
    .getDownloadURL() method is called without needing the ee module import.
    This cleanly separates the export concern from the EE API dependency.

Download workflow:
    1. Compute tile grid from AOI bounds and pixel scale.
    2. For each tile:
       a. Call image.getDownloadURL(params) via execute_with_retry().
       b. Fetch the response URL to bytes via stdlib urllib.
       c. Parse bytes (GeoTIFF or ZIP) with rasterio MemoryFile.
       d. Write pixels into the correct window of the output array.
    3. Return DownloadResult with the assembled array and georef metadata.
"""

from __future__ import annotations

import io
import logging
import math
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import rasterio

import numpy as np

from src.gee import GEEAPIError

if TYPE_CHECKING:
    from src.gee.client import EarthEngineClient

__all__ = [
    "AoiBounds",
    "AffineTransform",
    "TileSpec",
    "DownloadResult",
    "EarthEngineDownloader",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_METRES_PER_DEGREE_LAT: float = 111_000.0
_DOWNLOAD_TIMEOUT_SECONDS:    int = 600


# ==============================================================================
# Value objects
# ==============================================================================

@dataclass(frozen=True)
class AoiBounds:
    """
    Immutable WGS84 bounding box in decimal degrees.

    Attributes:
        min_lon: Western boundary (longitude).
        min_lat: Southern boundary (latitude).
        max_lon: Eastern boundary (longitude).
        max_lat: Northern boundary (latitude).
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_geojson_polygon(self) -> dict:
        """Return a GeoJSON Polygon dict for this bounding box."""
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.min_lon, self.min_lat],
                [self.max_lon, self.min_lat],
                [self.max_lon, self.max_lat],
                [self.min_lon, self.max_lat],
                [self.min_lon, self.min_lat],
            ]],
        }

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (min_lon, min_lat, max_lon, max_lat) as a plain tuple."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


@dataclass(frozen=True)
class AffineTransform:
    """
    Immutable affine transform coefficients for a geospatial raster.

    Follows the GDAL / rasterio convention:
        | x' |   | a  b  c | | x |
        | y' | = | d  e  f | | y |
        | 1  |   | 0  0  1 | | 1 |

    For a north-up raster: b == 0, d == 0, a > 0, e < 0.

    Attributes:
        a: X pixel size (positive for west-to-east).
        b: X rotation (zero for north-up rasters).
        c: X coordinate of the upper-left corner.
        d: Y rotation (zero for north-up rasters).
        e: Y pixel size (negative for north-up / top-to-bottom).
        f: Y coordinate of the upper-left corner.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @classmethod
    def from_affine(cls, transform: Any) -> AffineTransform:
        """
        Construct from an affine.Affine (or rasterio transform) object.

        Args:
            transform: An object with .a .b .c .d .e .f attributes.

        Returns:
            AffineTransform with values copied from transform.
        """
        return cls(
            a=float(getattr(transform, "a", 0.0)),
            b=float(getattr(transform, "b", 0.0)),
            c=float(getattr(transform, "c", 0.0)),
            d=float(getattr(transform, "d", 0.0)),
            e=float(getattr(transform, "e", 0.0)),
            f=float(getattr(transform, "f", 0.0)),
        )

    def to_affine(self) -> Any:
        """
        Convert to an affine.Affine object for use with rasterio.

        Returns:
            affine.Affine instance.

        Raises:
            ImportError: affine package is not installed (bundled with rasterio).
        """
        from affine import Affine
        return Affine(self.a, self.b, self.c, self.d, self.e, self.f)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dict of the transform coefficients."""
        return {"a": self.a, "b": self.b, "c": self.c,
                "d": self.d, "e": self.e, "f": self.f}


@dataclass(frozen=True)
class TileSpec:
    """
    Immutable descriptor for one download tile within an AOI.

    Attributes:
        bounds:  Geographic bounding box of this tile.
        tile_id: Human-readable identifier, e.g. "tile_0_0".
    """

    bounds:  AoiBounds
    tile_id: str

    def to_geojson_polygon(self) -> dict:
        """Delegate to bounds.to_geojson_polygon()."""
        return self.bounds.to_geojson_polygon()


@dataclass(frozen=True)
class DownloadResult:
    """
    Immutable result of EarthEngineDownloader.download().

    The data field is a float32 numpy array (bands, height, width). The
    dataclass is frozen, meaning the field reference cannot be reassigned.
    The array contents are mutable by numpy convention; callers must not
    modify them after the DownloadResult is returned.

    Attributes:
        data:       Float32 numpy array, shape (bands, height, width).
        crs:        Output CRS string, e.g. "EPSG:4326".
        transform:  Affine coefficients as an immutable AffineTransform.
        band_names: Ordered tuple of band names matching the band axis.
        width:      Raster width in pixels.
        height:     Raster height in pixels.
        aoi_bounds: Geographic extent of the download as AoiBounds.
        num_tiles:  Number of GEE download tiles assembled into data.
    """

    data:       Any               # np.ndarray float32 (bands, height, width)
    crs:        str
    transform:  AffineTransform
    band_names: tuple[str, ...]
    width:      int
    height:     int
    aoi_bounds: AoiBounds
    num_tiles:  int


# ==============================================================================
# EarthEngineDownloader
# ==============================================================================

class EarthEngineDownloader:
    """
    Downloads an ee.Image to a float32 numpy array without importing ee.

    Receives the ee.Image as an opaque Any object (created by src/gee modules)
    and calls only its .getDownloadURL() method via execute_with_retry(). No
    'import ee' statement is required because no EE type construction occurs here.

    Tiles large AOIs automatically to stay within GEE's per-request size limit.

    Args:
        client:           Initialized EarthEngineClient (from src.gee).
        max_tile_pixels:  Maximum pixels per download tile (width * height).
                          Default 1,000,000 is safe for 11-band float32 images.
    """

    def __init__(
        self,
        client: EarthEngineClient,
        max_tile_pixels: int = 1_000_000,
    ) -> None:
        self._client          = client
        self._max_tile_pixels = max_tile_pixels
        self._logger: logging.Logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def download(
        self,
        image: Any,
        aoi_bounds: AoiBounds,
        band_names: list[str],
        scale_meters: float,
        crs: str,
    ) -> DownloadResult:
        """
        Download an ee.Image to a float32 numpy array.
        """

        tiles = self._compute_tiles(aoi_bounds, scale_meters)

        n_bands = len(band_names)

        self._logger.info(
            "Downloading image. bands=%d tiles=%d scale=%gm crs=%s",
            n_bands,
            len(tiles),
            scale_meters,
            crs,
        )

        # ---------------------------------------------------------
        # PASS 1
        # Download every tile.
        # Nothing is stitched yet.
        # ---------------------------------------------------------

        tile_records = []

        out_transform = None
        out_crs = crs

        for tile in tiles:

            self._logger.debug("Downloading %s", tile.tile_id)

            tile_data, tile_crs, tile_transform = self._download_tile(
                image,
                tile,
                band_names,
                scale_meters,
                crs,
            )

            if out_transform is None:
                out_transform = tile_transform
                out_crs = tile_crs

            row = int(tile.tile_id.split("_")[1])
            col = int(tile.tile_id.split("_")[2])

            tile_records.append(
                {
                    "row": row,
                    "col": col,
                    "data": tile_data,
                    "transform": tile_transform,
                }
            )

        if out_transform is None:
            raise GEEAPIError(
                operation="download",
                reason="No tiles were successfully downloaded.",
            )   

        # ---------------------------------------------------------
        # PASS 2
        # Determine the true size of every row and column from the
        # downloaded tiles.
        # ---------------------------------------------------------

        row_heights = {}
        col_widths = {}

        for rec in tile_records:
            _, h, w = rec["data"].shape

            row = rec["row"]
            col = rec["col"]

            row_heights[row] = max(row_heights.get(row, 0), h)
            col_widths[col] = max(col_widths.get(col, 0), w)

        n_rows = max(row_heights.keys()) + 1
        n_cols = max(col_widths.keys()) + 1

        total_h = sum(row_heights[r] for r in range(n_rows))
        total_w = sum(col_widths[c] for c in range(n_cols))

        output = np.full(
            (n_bands, total_h, total_w),
            np.nan,
            dtype=np.float32,
        )

        #
        # cumulative offsets
        #

        row_offsets = {}
        running = 0

        for r in reversed(range(n_rows)):
            row_offsets[r] = running
            running += row_heights[r]

        col_offsets = {}
        running = 0

        for c in range(n_cols):
            col_offsets[c] = running
            running += col_widths[c]

        # ---------------------------------------------------------
        # PASS 3
        # Stitch tiles into the output mosaic.
        # ---------------------------------------------------------

        for rec in tile_records:

            tile_data = rec["data"]
            row = rec["row"]
            col = rec["col"]

            _, th, tw = tile_data.shape

            r0 = row_offsets[row]
            c0 = col_offsets[col]

            r1 = r0 + th
            c1 = c0 + tw

            output[:, r0:r1, c0:c1] = tile_data

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        rgb = output[[2, 1, 0]].transpose(1, 2, 0)
        rgb = np.nan_to_num(rgb)

        mn = rgb.min()
        mx = rgb.max()

        if mx > mn:
            rgb = (rgb - mn) / (mx - mn)

        from PIL import Image

        Image.fromarray((rgb * 255).astype(np.uint8)).save("stitched.png")

        actual_h = output.shape[1]
        actual_w = output.shape[2]

        return DownloadResult(
            data=output,
            crs=out_crs,
            transform=out_transform,
            band_names=tuple(band_names),
            width=actual_w,
            height=actual_h,
            aoi_bounds=aoi_bounds,
            num_tiles=len(tile_records),
        )








    # ------------------------------------------------------------------
    # Private — tiling
    # ------------------------------------------------------------------

    def _estimate_dimensions(
        self,
        aoi:         AoiBounds,
        scale_meters: float,
    ) -> tuple[int, int]:
        """Estimate (width_px, height_px) for the AOI at the given scale."""
        centre_lat = (aoi.min_lat + aoi.max_lat) / 2.0
        lon_scale  = math.cos(math.radians(centre_lat))


        width_m    = (aoi.max_lon - aoi.min_lon) * _METRES_PER_DEGREE_LAT * lon_scale
        height_m   = (aoi.max_lat - aoi.min_lat) * _METRES_PER_DEGREE_LAT

        return (
            max(1, int(round(width_m  / scale_meters))),
            max(1, int(round(height_m / scale_meters))),
        )

    def _compute_tiles(
        self,
        aoi:         AoiBounds,
        scale_meters: float,
    ) -> list[TileSpec]:
        """Return a list of TileSpec objects covering the AOI."""
        total_w, total_h = self._estimate_dimensions(aoi, scale_meters)

        if total_w * total_h <= self._max_tile_pixels:
            return [TileSpec(bounds=aoi, tile_id="tile_0_0")]

        side   = int(math.sqrt(self._max_tile_pixels))
        n_cols = math.ceil(total_w / side)
        n_rows = math.ceil(total_h / side)
        lon_step = (aoi.max_lon - aoi.min_lon) / n_cols
        lat_step = (aoi.max_lat - aoi.min_lat) / n_rows

        tiles: list[TileSpec] = []
        for row in range(n_rows):
            for col in range(n_cols):
                bounds = AoiBounds(
                    min_lon=aoi.min_lon + col * lon_step,
                    min_lat=aoi.min_lat + row * lat_step,
                    max_lon=min(aoi.max_lon, aoi.min_lon + (col + 1) * lon_step),
                    max_lat=min(aoi.max_lat, aoi.min_lat + (row + 1) * lat_step),
                )
                tiles.append(TileSpec(bounds=bounds, tile_id=f"tile_{row}_{col}"))
                # print("=" * 60)
                # print(f"Tile {row}, {col}")
                # print("min_lon =", bounds.min_lon)
                # print("max_lon =", bounds.max_lon)
                # print("min_lat =", bounds.min_lat)
                # print("max_lat =", bounds.max_lat)
                # print("=" * 60)
                

        self._logger.info(
            "AOI tiled: %d rows x %d cols = %d tiles.", n_rows, n_cols, len(tiles)
        )
        return tiles

    def _pixel_offset(
        self,
        tile_t: AffineTransform,
        out_t:  AffineTransform,
    ) -> tuple[int, int]:
        """
        Compute (col_offset, row_offset) of tile_t in the out_t grid.

        Uses the affine transforms to map geographic origin differences
        to integer pixel offsets in the output array.
        """
        col = int(round((tile_t.c - out_t.c) / out_t.a))
        row = int(round((tile_t.f - out_t.f) / abs(out_t.e)))
        return col, row

    # ------------------------------------------------------------------
    # Private — per-tile download
    # ------------------------------------------------------------------

    def _download_tile(
        self,
        image:       Any,
        tile:        TileSpec,
        band_names:  list[str],
        scale_meters: float,
        crs:         str,
    ) -> tuple[Any, str, AffineTransform]:
        """Download one tile. Returns (array, crs_str, AffineTransform)."""
        params = {
            "bands":       [{"id": name} for name in band_names],
            "region":      tile.to_geojson_polygon(),
            "scale":       scale_meters,
            "crs":         crs,
            "format":      "GEO_TIFF",
            "filePerBand": False,
        }
        url        = self._get_download_url(image, params)
        data_bytes = self._fetch_url(url)
        return self._parse_bytes(data_bytes, band_names)

    def _get_download_url(self, image: Any, params: dict) -> str:
        """
        Call image.getDownloadURL(params) via execute_with_retry().

        No 'import ee' is needed because image is an ee.Image passed from
        src/gee; we only call a method on it.
        """
        try:
            return self._client.execute_with_retry(
                lambda: image.getDownloadURL(params)
            )
        except Exception as exc:
            raise GEEAPIError(
                operation="_get_download_url",
                reason=f"image.getDownloadURL() failed: {exc}",
            ) from exc

    def _fetch_url(self, url: str) -> bytes:
        """Download raw bytes from url using stdlib urllib."""
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise GEEAPIError(
                operation="_fetch_url",
                reason=f"HTTP {exc.code}: {exc.reason}",
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise GEEAPIError(
                operation="_fetch_url",
                reason=f"Network error: {exc}",
            ) from exc

    def _parse_bytes(
        self,
        data: bytes,
        band_names: list[str],
    ) -> tuple[Any, str, AffineTransform]:
        """Detect ZIP vs raw GeoTIFF by magic bytes and dispatch."""
        if data[:2] == b"PK":
            return self._parse_zip(data, band_names)
        return self._parse_geotiff(data)

    def _parse_geotiff(self, data: bytes) -> tuple[Any, str, AffineTransform]:
        """Parse a raw GeoTIFF byte sequence with rasterio MemoryFile."""
        try:
            import rasterio
            from rasterio.io import MemoryFile
        except ImportError as exc:
            raise GEEAPIError(
                operation="_parse_geotiff",
                reason="rasterio is not installed.",
            ) from exc

        try:
            with MemoryFile(data) as mf:
                with mf.open() as ds:
                    # print("=" * 60)
                    # print("WIDTH :", ds.width)
                    # print("HEIGHT:", ds.height)
                    # print("BOUNDS:", ds.bounds)
                    # print("TRANSFORM:", ds.transform)
                    # print("=" * 60) 
                    arr       = ds.read().astype(np.float32)
                    crs_str   = ds.crs.to_string()
                    transform = AffineTransform.from_affine(ds.transform)
            return arr, crs_str, transform
        except Exception as exc:
            raise GEEAPIError(
                operation="_parse_geotiff",
                reason=f"rasterio failed to parse GeoTIFF bytes: {exc}",
            ) from exc

    def _parse_zip(
        self,
        data: bytes,
        band_names: list[str],
    ) -> tuple[Any, str, AffineTransform]:
        """Parse a ZIP of per-band GeoTIFFs (older GEE response format)."""
        try:
            import rasterio
            from rasterio.io import MemoryFile
        except ImportError as exc:
            raise GEEAPIError(
                operation="_parse_zip",
                reason="rasterio is not installed.",
            ) from exc

        bands:    dict[str, Any] = {}
        crs_str:  str | None      = None
        transform: AffineTransform | None = None

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(".tif"):
                        continue
                    stem  = name.rsplit(".", 1)[0].split(".")[-1]
                    with zf.open(name) as fh:
                        raw = fh.read()
                    with MemoryFile(raw) as mf:
                        with mf.open() as ds:
                            if crs_str is None:
                                crs_str   = ds.crs.to_string()
                                transform = AffineTransform.from_affine(ds.transform)
                            bands[stem] = ds.read(1).astype(np.float32)
        except zipfile.BadZipFile as exc:
            raise GEEAPIError(
                operation="_parse_zip",
                reason=f"Response is not a valid ZIP: {exc}",
            ) from exc
        except Exception as exc:
            raise GEEAPIError(
                operation="_parse_zip",
                reason=f"Failed to extract bands from ZIP: {exc}",
            ) from exc

        if not bands or crs_str is None or transform is None:
            raise GEEAPIError(
                operation="_parse_zip",
                reason="ZIP contained no valid GeoTIFF entries.",
            )

        sample   = next(iter(bands.values()))
        h, w     = sample.shape
        stacked  = np.stack([
            bands.get(n, np.zeros((h, w), dtype=np.float32))
            for n in band_names
        ])
        return stacked, crs_str, transform