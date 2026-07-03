"""
PyTorch Dataset for the River Morphology Segmentation System (Module 11).

RiverMorphologyDataset reads a patch GeoTIFF (image) and a mask GeoTIFF
(label) for each sample in a DatasetManifest split.  Both files are read
via rasterio in __getitem__; no pixels are held in memory at construction.

Transform decoupling:
    RiverMorphologyDataset calls

        image_np, mask_np = self._transform(image_np, mask_np)

    without any knowledge of whether the transform is an identity operation,
    an albumentations pipeline, or any future implementation.  The default
    is IdentityTransform so that eval splits and tests constructed without
    an explicit transform work correctly without an if-branch.

Output per sample:
    image : float32 torch.Tensor  shape (C, H, W)
    mask  : int64   torch.Tensor  shape (H, W)
    meta  : SampleMetadata

No GeoTIFF writing. No EE imports. torch and rasterio are lazy-imported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.training.transforms import IdentityTransform, Transform

__all__ = [
    "SampleMetadata",
    "RiverMorphologySample",
    "RiverMorphologyDataset",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# SampleMetadata
# ==============================================================================

@dataclass(frozen=True)
class SampleMetadata:
    """
    Immutable per-sample temporal and spatial provenance.

    Carried alongside image/mask tensors through the DataLoader so that
    training loops can implement season-stratified logging, per-year
    evaluation, or reach-specific analysis.

    Attributes:
        sample_id:               Unique sample identifier (== patch_id).
        scene_id:                 Source scene identifier.
        split:                     "train", "validation", or "test".
        acquisition_date:           Representative imagery date (YYYY-MM-DD).
        year:                       Calendar year.
        month:                      Calendar month [1, 12].
        season:                     Resolved season name.
        hydrological_year:           Resolved hydrological year.
        sensor:                     Comma-separated sensor names.
        aoi_id:                     Area of interest identifier.
        river_name:                  River name, or empty string.
        reach_id:                    River reach identifier, or empty string.
        basin_id:                    Drainage basin identifier, or empty string.
        patch_path:                  Absolute path to the patch GeoTIFF.
        mask_path:                   Absolute path to the mask GeoTIFF.
        label_valid_pixel_ratio:     Fraction of valid pixels in the mask.
    """

    sample_id:               str
    scene_id:                 str
    split:                     str
    acquisition_date:           str
    year:                       int
    month:                      int
    season:                     str
    hydrological_year:           int
    sensor:                     str
    aoi_id:                     str
    river_name:                  str
    reach_id:                    str
    basin_id:                    str
    patch_path:                  str
    mask_path:                   str
    label_valid_pixel_ratio:     float


@dataclass(frozen=True)
class RiverMorphologySample:
    """
    Immutable sample returned by RiverMorphologyDataset.__getitem__.

    Attributes:
        image:    Float32 torch.Tensor, shape (C, H, W).  Normalized.
        mask:     Int64   torch.Tensor, shape (H, W).     Class IDs.
        metadata: SampleMetadata for provenance tracking.
    """

    image:    Any   # torch.Tensor float32 (C, H, W)
    mask:     Any   # torch.Tensor int64   (H, W)
    metadata: SampleMetadata


# ==============================================================================
# RiverMorphologyDataset
# ==============================================================================

class RiverMorphologyDataset:
    """
    PyTorch-compatible Dataset for river morphology semantic segmentation.

    Reads one (patch, mask) pair per __getitem__ call via rasterio. No pixel
    data is pre-loaded at construction time (fully lazy).

    The transform parameter accepts any Transform subclass.  When None,
    an IdentityTransform is installed so that __getitem__ can call
    transform(image, mask) unconditionally.

    Compatible with torch.utils.data.DataLoader via __len__ and __getitem__.

    Args:
        entries:      Ordered list of DatasetManifestEntry for this split.
        norm_stats:   NormalizationStats to apply, or None for raw values.
        transform:    Transform instance. Defaults to IdentityTransform().
        nodata_value: Nodata sentinel in mask files.
        ignore_index: Replaces nodata pixels in the output mask tensor.
        split:        Split name for metadata ("train", "validation", "test").
        band_names:   Ordered band names. Used by band_names property.
        class_names:  Ordered class names. Used by class_names property.
        num_classes:  Number of segmentation classes. Used by num_classes property.
    """

    def __init__(
        self,
        entries:      list[Any],
        norm_stats:   Any | None    = None,
        transform:    Transform | None = None,
        nodata_value: int           = 255,
        ignore_index: int           = 255,
        split:        str           = "train",
        band_names:   tuple[str, ...] = (),
        class_names:  tuple[str, ...] = (),
        num_classes:  int           = 0,
    ) -> None:
        self._entries       = list(entries)
        self._norm_stats    = norm_stats
        self._transform     = transform if transform is not None else IdentityTransform()
        self._nodata_value  = int(nodata_value)
        self._ignore_index  = int(ignore_index)
        self._split         = split
        self._band_names    = tuple(band_names)
        self._class_names   = tuple(class_names)
        self._num_classes   = int(num_classes)
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._logger.debug(
            "RiverMorphologyDataset: split=%s, n=%d, "
            "normalize=%s, transform=%s",
            split, len(self._entries),
            norm_stats is not None,
            type(self._transform).__name__,
        )

    # ------------------------------------------------------------------
    # Required Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self._entries)

    def __getitem__(self, index: int) -> tuple[Any, Any, SampleMetadata]:
        """
        Load and return one (image, mask, metadata) sample.

        Sequence:
            1. Read patch GeoTIFF    -> float32 (C, H, W)
            2. Read mask GeoTIFF     -> uint8   (H, W)
            3. Apply normalization   (if norm_stats provided)
            4. Apply transform       (IdentityTransform or any Transform subclass)
            5. Replace nodata pixels -> ignore_index
            6. Convert to tensors    -> float32 (C,H,W), int64 (H,W)
            7. Build SampleMetadata
            8. Return (image_tensor, mask_tensor, metadata)

        Returns:
            Tuple of:
                image:    float32 torch.Tensor (C, H, W)
                mask:     int64   torch.Tensor (H, W)
                metadata: SampleMetadata
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is not installed. Install with: pip install torch>=2.0"
            ) from exc

        entry = self._entries[index]

        image_np = self._read_image(Path(entry.patch_path))
        mask_np  = self._read_mask(Path(entry.mask_path))

        if self._norm_stats is not None:
            image_np = self._norm_stats.normalize(image_np)

        image_np, mask_np = self._transform(image_np, mask_np)

        mask_np = np.where(
            mask_np == self._nodata_value,
            self._ignore_index,
            mask_np,
        ).astype(np.int64)

        image_tensor = torch.from_numpy(image_np.copy()).float()
        mask_tensor  = torch.from_numpy(mask_np.copy()).long()

        metadata = SampleMetadata(
            sample_id               = entry.sample_id,
            scene_id                = entry.scene_id,
            split                   = self._split,
            acquisition_date        = entry.acquisition_date,
            year                    = entry.year,
            month                   = entry.month,
            season                  = entry.season,
            hydrological_year       = entry.hydrological_year,
            sensor                  = entry.sensor,
            aoi_id                  = entry.aoi_id,
            river_name              = entry.river_name,
            reach_id                = entry.reach_id,
            basin_id                = entry.basin_id,
            patch_path              = entry.patch_path,
            mask_path               = entry.mask_path,
            label_valid_pixel_ratio = entry.label_valid_pixel_ratio,
        )

        return image_tensor, mask_tensor, metadata

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def num_bands(self) -> int:
        """
        Number of spectral bands in each patch.

        Derived from num_bands on the first manifest entry to avoid
        opening any file at construction time. Returns 0 when the
        dataset is empty.
        """
        if not self._entries:
            return 0
        return int(getattr(self._entries[0], "num_bands", 0))

    @property
    def band_names(self) -> tuple[str, ...]:
        """
        Ordered tuple of spectral band names (e.g. "Blue", "Green", "MNDWI").

        Populated by DataLoaderFactory when the full band list is known.
        Returns an empty tuple when not provided at construction.
        """
        return self._band_names

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Ordered tuple of segmentation class names (e.g. "background", "water").

        Populated by DataLoaderFactory from ClassSchema.
        Returns an empty tuple when not provided at construction.
        """
        return self._class_names

    @property
    def num_classes(self) -> int:
        """
        Number of segmentation classes for this dataset.

        Populated by DataLoaderFactory from ClassSchema.num_classes.
        Returns 0 when not provided at construction.
        """
        return self._num_classes

    @property
    def patch_size(self) -> int:
        """
        Patch side length in pixels (width of the first entry, assuming
        Module 8 always produces square patches).

        Returns 0 when the dataset is empty.
        """
        if not self._entries:
            return 0
        return int(getattr(self._entries[0], "width", 0))

    @property
    def ignore_index(self) -> int:
        """
        The value used to replace nodata pixels in the output mask tensor.

        Passed directly to torch loss functions as ignore_index so that
        nodata pixels are excluded from gradient computation.
        """
        return self._ignore_index

    @property
    def split(self) -> str:
        """The split this dataset represents ("train", "validation", "test")."""
        return self._split

    @property
    def num_samples(self) -> int:
        """Alias for __len__() for clarity in non-DataLoader contexts."""
        return len(self._entries)

    def summary(self) -> str:
        """
        Return a multi-line ASCII description of this dataset.

        Includes split name, sample count, band/class counts, patch size,
        active transform type, and whether normalization is applied.

        Returns:
            Multi-line ASCII string. All characters are in [0, 127].
        """
        lines = [
            "RiverMorphologyDataset",
            f"  split:        {self._split}",
            f"  samples:      {len(self._entries)}",
            f"  num_bands:    {self.num_bands}",
            f"  band_names:   {list(self._band_names) if self._band_names else 'not set'}",
            f"  num_classes:  {self._num_classes}",
            f"  class_names:  {list(self._class_names) if self._class_names else 'not set'}",
            f"  patch_size:   {self.patch_size} px",
            f"  ignore_index: {self._ignore_index}",
            f"  normalize:    {self._norm_stats is not None}",
            f"  transform:    {type(self._transform).__name__}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_image(self, path: Path) -> np.ndarray:
        """
        Read all bands from a patch GeoTIFF as float32 (C, H, W).

        Raises:
            OSError: rasterio cannot open the file.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise OSError("rasterio is not installed.") from exc

        try:
            with rasterio.open(path) as ds:
                return ds.read().astype(np.float32)
        except Exception as exc:
            raise OSError(
                f"Failed to read patch GeoTIFF: {path}: {exc}"
            ) from exc

    def _read_mask(self, path: Path) -> np.ndarray:
        """
        Read band 1 from a mask GeoTIFF as uint8 (H, W).

        Raises:
            OSError: rasterio cannot open the file.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise OSError("rasterio is not installed.") from exc

        try:
            with rasterio.open(path) as ds:
                return ds.read(1).astype(np.uint8)
        except Exception as exc:
            raise OSError(
                f"Failed to read mask GeoTIFF: {path}: {exc}"
            ) from exc