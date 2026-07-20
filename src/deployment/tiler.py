"""
Sliding-window AOI tiler.

Reads a GeoTIFF lazily using rasterio and yields overlapping
tiles suitable for semantic segmentation inference.

This module never loads the full raster into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass(frozen=True)
class AOITile:
    """
    One inference tile.

    image:
        (C,H,W) float32 array.

    row:
        Pixel row in the original raster.

    col:
        Pixel column in the original raster.
    """

    image: np.ndarray

    row: int
    col: int

    height: int
    width: int

    window: Window


class AOITiler:

    def __init__(
        self,
        patch_size: int = 128,
        stride: int = 64,
    ):

        if patch_size <= 0:
            raise ValueError("patch_size must be >0")

        if stride <= 0:
            raise ValueError("stride must be >0")

        self.patch_size = patch_size
        self.stride = stride

    def generate(
        self,
        raster_path: str | Path,
    ):
        """
        Yield AOITile objects.
        """

        raster_path = Path(raster_path)

        with rasterio.open(raster_path) as src:

            H = src.height
            W = src.width

            patch = self.patch_size
            stride = self.stride

            rows = list(range(0, max(H - patch + 1, 1), stride))
            cols = list(range(0, max(W - patch + 1, 1), stride))

            #
            # Ensure last tile reaches image border.
            #
            if rows[-1] != H - patch:
                rows.append(max(H - patch, 0))

            if cols[-1] != W - patch:
                cols.append(max(W - patch, 0))

            for row in rows:

                for col in cols:

                    window = Window(
                        col_off=col,
                        row_off=row,
                        width=min(patch, W - col),
                        height=min(patch, H - row),
                    )

                    image = src.read(
                        window=window,
                    ).astype(np.float32)

                    #
                    # Pad edge tiles.
                    #
                    if (
                        image.shape[1] != patch
                        or image.shape[2] != patch
                    ):

                        padded = np.zeros(
                            (
                                image.shape[0],
                                patch,
                                patch,
                            ),
                            dtype=np.float32,
                        )

                        padded[
                            :,
                            : image.shape[1],
                            : image.shape[2],
                        ] = image

                        image = padded

                    yield AOITile(
                        image=image,
                        row=row,
                        col=col,
                        height=window.height,
                        width=window.width,
                        window=window,
                    )