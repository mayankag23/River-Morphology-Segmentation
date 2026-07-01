"""
Patch tiling for the River Morphology Patch Generation pipeline.

PatchTiler computes a deterministic grid of fixed-size patch windows over a
source raster, given patch_size and stride. Only full-size windows are
produced; partial windows at the right/bottom edges (smaller than
patch_size) are dropped, since downstream ML modules require uniform
patch dimensions for batching.

All operations are pure Python / index arithmetic. No I/O, no EE, no rasterio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.core.exceptions import InvalidValueError

__all__ = ["PatchWindow", "PatchTiler"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchWindow:
    """
    Immutable pixel window describing one patch's location in the source raster.

    Attributes:
        row_index: Zero-indexed row position in the tiling grid (not pixels).
                   Used to build the deterministic patch ID, e.g. "r001".
        col_index: Zero-indexed column position in the tiling grid (not pixels).
                   Used to build the deterministic patch ID, e.g. "c001".
        row_off:   Pixel row offset of the window's top-left corner in the
                   source raster.
        col_off:   Pixel column offset of the window's top-left corner in
                   the source raster.
        height:    Window height in pixels. Always equals patch_size.
        width:     Window width in pixels. Always equals patch_size.
    """

    row_index: int
    col_index: int
    row_off:   int
    col_off:   int
    height:    int
    width:     int


class PatchTiler:
    """
    Computes a deterministic grid of fixed-size patch windows over a raster.

    Only full patch_size x patch_size windows are produced. Incomplete
    windows at the right and bottom edges of the raster (smaller than
    patch_size) are dropped, ensuring every output patch has identical
    dimensions as required by downstream ML modules.

    Args:
        patch_size: Side length of each square patch in pixels. Must be
                    a positive integer. Sourced from
                    config.patch_generation.patch_size.
        stride:     Pixel distance between consecutive patch origins.
                    stride < patch_size  -> overlapping patches.
                    stride == patch_size -> non-overlapping patches.
                    stride > patch_size  -> gaps between patches.
                    Sourced from config.patch_generation.train_stride.

    Raises:
        InvalidValueError: patch_size or stride is not a positive integer.
    """

    def __init__(self, patch_size: int, stride: int) -> None:
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise InvalidValueError(
                field="patch_size",
                value=patch_size,
                reason="must be a positive integer",
            )
        if not isinstance(stride, int) or stride <= 0:
            raise InvalidValueError(
                field="stride",
                value=stride,
                reason="must be a positive integer",
            )
        self._patch_size = patch_size
        self._stride     = stride
        self._logger: logging.Logger = logging.getLogger(__name__)

    @property
    def patch_size(self) -> int:
        """The configured patch side length in pixels."""
        return self._patch_size

    @property
    def stride(self) -> int:
        """The configured stride in pixels."""
        return self._stride

    def compute_windows(
        self,
        raster_width:  int,
        raster_height: int,
    ) -> tuple[PatchWindow, ...]:
        """
        Compute all full-size patch windows fitting within the raster.

        Args:
            raster_width:  Source raster width in pixels.
            raster_height: Source raster height in pixels.

        Returns:
            Ordered tuple of PatchWindow, row-major (top to bottom,
            left to right). Empty tuple if the raster is smaller than
            patch_size along either axis.

        Raises:
            InvalidValueError: raster dimensions are not positive integers.
        """
        if raster_width <= 0 or raster_height <= 0:
            raise InvalidValueError(
                field="raster dimensions",
                value=f"{raster_width}x{raster_height}",
                reason="must both be positive integers",
            )

        row_offsets = self._compute_offsets(raster_height)
        col_offsets = self._compute_offsets(raster_width)

        windows: list[PatchWindow] = []
        for row_idx, row_off in enumerate(row_offsets):
            for col_idx, col_off in enumerate(col_offsets):
                windows.append(PatchWindow(
                    row_index=row_idx,
                    col_index=col_idx,
                    row_off=row_off,
                    col_off=col_off,
                    height=self._patch_size,
                    width=self._patch_size,
                ))

        self._logger.debug(
            "Computed %d patch window(s) for %dx%d raster "
            "(patch_size=%d, stride=%d).",
            len(windows), raster_width, raster_height,
            self._patch_size, self._stride,
        )
        return tuple(windows)

    def _compute_offsets(self, dimension: int) -> list[int]:
        """Return valid full-patch starting pixel offsets along one axis."""
        if dimension < self._patch_size:
            return []
        last_offset = dimension - self._patch_size
        return list(range(0, last_offset + 1, self._stride))