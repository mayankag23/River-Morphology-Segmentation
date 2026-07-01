"""
GeoTIFF window reading for the River Morphology Patch Generation pipeline.

PatchReader wraps a rasterio dataset to read fixed-size pixel windows on
demand, avoiding loading the full source raster into memory at once. This
supports large AOIs without exhausting available RAM.

Source raster CRS and band names are taken from SceneMetadata (Module 7)
by the caller (PatchGenerator), not re-read here -- this class is
responsible only for pixel-level window reads and per-window transform
computation.

No EE imports. No GeoTIFF writing. Pure rasterio I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.export.downloader import AffineTransform
from src.patches.tiler import PatchWindow

__all__ = ["PatchReader"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class PatchReader:
    """
    Reads fixed-size pixel windows from a GeoTIFF using rasterio.

    Opens the source dataset once at construction time and keeps it open
    for the lifetime of this reader. Use as a context manager (recommended)
    or call close() explicitly when done.

    Args:
        image_path: Path to the source scene GeoTIFF (image.tif, written
                    by Module 7's GeoTiffWriter).

    Raises:
        OSError: rasterio is not installed, the file does not exist, or
                 the file cannot be opened as a valid GeoTIFF.
    """

    def __init__(self, image_path: Path) -> None:
        self._image_path = Path(image_path).resolve()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._dataset: Any = self._open_dataset()

    def __enter__(self) -> PatchReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying rasterio dataset. Safe to call multiple times."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    @property
    def width(self) -> int:
        """Source raster width in pixels."""
        return int(self._dataset.width)

    @property
    def height(self) -> int:
        """Source raster height in pixels."""
        return int(self._dataset.height)

    @property
    def num_bands(self) -> int:
        """Number of bands in the source raster."""
        return int(self._dataset.count)

    def read_window(
        self,
        window: PatchWindow,
    ) -> tuple[Any, AffineTransform]:
        """
        Read pixel data for one PatchWindow.

        Uses rasterio's windowed read so only the requested pixel region
        is loaded into memory, regardless of the source raster's total size.

        Args:
            window: PatchWindow describing the pixel region to read.

        Returns:
            Tuple of (data, transform):
                data:      float32 numpy array, shape (bands, height, width).
                transform: AffineTransform for this specific patch window,
                          with origin shifted to the window's top-left
                          corner in geographic coordinates. Computed via
                          rasterio's window_transform(), which correctly
                          accounts for both row and column offsets.
        """
        from rasterio.windows import Window

        rio_window = Window(
            col_off=window.col_off,
            row_off=window.row_off,
            width=window.width,
            height=window.height,
        )
        data = self._dataset.read(window=rio_window).astype(np.float32)
        window_transform = self._dataset.window_transform(rio_window)
        patch_transform = AffineTransform.from_affine(window_transform)

        return data, patch_transform

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _open_dataset(self) -> Any:
        """Open the source GeoTIFF with rasterio."""
        try:
            import rasterio
        except ImportError as exc:
            raise OSError(
                "rasterio is not installed. "
                "Install with: pip install rasterio==1.3.9"
            ) from exc

        if not self._image_path.exists():
            raise OSError(f"Source image not found: {self._image_path}")

        try:
            return rasterio.open(self._image_path)
        except Exception as exc:
            raise OSError(
                f"Failed to open GeoTIFF: {self._image_path}: {exc}"
            ) from exc