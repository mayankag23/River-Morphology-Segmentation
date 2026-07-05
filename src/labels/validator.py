# no change
"""
Label mask validation for the River Morphology Label Management pipeline
(Module 9).

LabelValidator checks each label mask against its source patch for
geospatial consistency (CRS, transform, dimensions) and content validity
(class IDs, NoData ratio, class diversity). Reads both rasters via rasterio
directly; never writes GeoTIFFs (writing is owned exclusively by Module
7's GeoTiffWriter and is not duplicated here).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import InvalidValueError
from src.labels.schema import ClassSchema

__all__ = ["LabelValidationResult", "LabelValidator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_TRANSFORM_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class LabelValidationResult:
    """
    Immutable result of LabelValidator.validate().

    Attributes:
        patch_id:             Patch identifier this result applies to.
        is_valid:               True if no blocking issues were found.
        issues:                  Ordered tuple of human-readable problem
                               descriptions.
        num_classes_present:     Count of distinct valid class IDs found
                               in the mask.
        is_single_class:         True if exactly one valid class is present.
        valid_pixel_ratio:        Fraction of pixels that are valid
                               (non-nodata, in-schema) class values.
        crs_match:                True if mask CRS matches the patch CRS.
        transform_match:          True if mask transform matches the patch
                               transform within tolerance.
        dimension_match:          True if mask dimensions match the patch
                               dimensions.
        mask_exists:              True if the mask file was found on disk.
    """

    patch_id:             str
    is_valid:               bool
    issues:                  tuple[str, ...]
    num_classes_present:     int
    is_single_class:         bool
    valid_pixel_ratio:        float
    crs_match:                bool
    transform_match:          bool
    dimension_match:          bool
    mask_exists:              bool


class LabelValidator:
    """
    Validates a label mask GeoTIFF against its source patch GeoTIFF.

    Args:
        class_schema:                ClassSchema defining valid class IDs.
        nodata_value:                 Integer sentinel for invalid/unlabeled
                                     mask pixels. Must not collide with a
                                     valid class ID.
        max_nodata_ratio:              Maximum allowable fraction of nodata
                                     pixels for a mask to be considered valid.
        min_distinct_classes:           Informational minimum distinct
                                     valid-class count.
        reject_single_class_masks:     If True, masks with exactly one
                                     distinct valid class are rejected.

    Raises:
        InvalidValueError: nodata_value collides with a valid class ID.
    """

    def __init__(
        self,
        class_schema:                ClassSchema,
        nodata_value:                 int,
        max_nodata_ratio:              float,
        min_distinct_classes:           int = 1,
        reject_single_class_masks:     bool = False,
    ) -> None:
        if class_schema.is_valid_class_id(nodata_value):
            raise InvalidValueError(
                field="labels.nodata_value",
                value=nodata_value,
                reason="must not collide with a valid class ID",
            )
        self._schema = class_schema
        self._nodata_value = int(nodata_value)
        self._max_nodata_ratio = float(max_nodata_ratio)
        self._min_distinct_classes = int(min_distinct_classes)
        self._reject_single_class_masks = bool(reject_single_class_masks)
        self._logger: logging.Logger = logging.getLogger(__name__)

    def validate(self, patch_path: Path, mask_path: Path | None) -> LabelValidationResult:
        """
        Validate one label mask against its source patch.

        Args:
            patch_path: Path to the source patch GeoTIFF (from Module 8).
            mask_path:  Path to the corresponding label mask GeoTIFF, or
                       None if no mask was discovered for this patch.

        Returns:
            LabelValidationResult describing all checks performed.
        """
        patch_id = Path(patch_path).stem

        if mask_path is None or not Path(mask_path).exists():
            return self._missing_result(patch_id)

        mask_path = Path(mask_path)

        try:
            import rasterio
        except ImportError:
            return self._error_result(
                patch_id, "rasterio is not installed; cannot validate mask",
                mask_exists=True,
            )

        issues: list[str] = []

        try:
            with rasterio.open(patch_path) as patch_ds, rasterio.open(mask_path) as mask_ds:
                crs_match = self._crs_match(patch_ds.crs, mask_ds.crs)
                if not crs_match:
                    issues.append(
                        f"CRS mismatch: patch={patch_ds.crs}, mask={mask_ds.crs}"
                    )

                transform_match = self._transforms_match(
                    patch_ds.transform, mask_ds.transform
                )
                if not transform_match:
                    issues.append(
                        "affine transform mismatch between patch and mask"
                    )

                dimension_match = (
                    patch_ds.width == mask_ds.width
                    and patch_ds.height == mask_ds.height
                )
                if not dimension_match:
                    issues.append(
                        f"dimension mismatch: patch={patch_ds.width}x{patch_ds.height}, "
                        f"mask={mask_ds.width}x{mask_ds.height}"
                    )

                mask_data = mask_ds.read(1)
        except Exception as exc:
            return self._error_result(
                patch_id, f"failed to read patch/mask rasters: {exc}",
                mask_exists=True,
            )

        content = self._validate_class_content(mask_data)
        issues.extend(content["issues"])

        is_single_class = content["num_classes_present"] == 1
        if is_single_class and self._reject_single_class_masks:
            issues.append(
                "mask contains only one distinct class "
                "(reject_single_class_masks is enabled)"
            )

        is_valid = (
            crs_match
            and transform_match
            and dimension_match
            and content["valid_pixel_ratio"] >= (1.0 - self._max_nodata_ratio)
            and not content["has_invalid_class_ids"]
            and not (is_single_class and self._reject_single_class_masks)
        )

        return LabelValidationResult(
            patch_id=patch_id,
            is_valid=is_valid,
            issues=tuple(issues),
            num_classes_present=content["num_classes_present"],
            is_single_class=is_single_class,
            valid_pixel_ratio=content["valid_pixel_ratio"],
            crs_match=crs_match,
            transform_match=transform_match,
            dimension_match=dimension_match,
            mask_exists=True,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _missing_result(patch_id: str) -> LabelValidationResult:
        return LabelValidationResult(
            patch_id=patch_id, is_valid=False,
            issues=("missing label file: mask not found on disk",),
            num_classes_present=0, is_single_class=False,
            valid_pixel_ratio=0.0, crs_match=False,
            transform_match=False, dimension_match=False,
            mask_exists=False,
        )

    @staticmethod
    def _error_result(
        patch_id: str, message: str, mask_exists: bool
    ) -> LabelValidationResult:
        return LabelValidationResult(
            patch_id=patch_id, is_valid=False,
            issues=(message,),
            num_classes_present=0, is_single_class=False,
            valid_pixel_ratio=0.0, crs_match=False,
            transform_match=False, dimension_match=False,
            mask_exists=mask_exists,
        )

    @staticmethod
    def _crs_match(crs_a: Any, crs_b: Any) -> bool:
        if crs_a is None or crs_b is None:
            return False
        return crs_a.to_string() == crs_b.to_string()

    @staticmethod
    def _transforms_match(transform_a: Any, transform_b: Any) -> bool:
        """Compare two affine transforms within a small numeric tolerance."""
        coeffs_a = (
            transform_a.a, transform_a.b, transform_a.c,
            transform_a.d, transform_a.e, transform_a.f,
        )
        coeffs_b = (
            transform_b.a, transform_b.b, transform_b.c,
            transform_b.d, transform_b.e, transform_b.f,
        )
        return all(
            math.isclose(x, y, abs_tol=_TRANSFORM_TOLERANCE)
            for x, y in zip(coeffs_a, coeffs_b)
        )

    def _validate_class_content(self, mask_data: Any) -> dict[str, Any]:
        """Check class ID validity, nodata ratio, and distinct class count."""
        issues: list[str] = []
        total_pixels = int(mask_data.size)

        unique_values = np.unique(mask_data)
        valid_class_ids = set(self._schema.class_ids)

        invalid_values = [
            int(v) for v in unique_values
            if int(v) not in valid_class_ids and int(v) != self._nodata_value
        ]
        has_invalid = len(invalid_values) > 0
        if has_invalid:
            issues.append(
                f"mask contains invalid class IDs not in schema: {invalid_values}"
            )

        nodata_count = int(np.sum(mask_data == self._nodata_value))
        valid_pixels = total_pixels - nodata_count
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

        present_classes = sorted(
            int(v) for v in unique_values if int(v) in valid_class_ids
        )

        if valid_ratio < (1.0 - self._max_nodata_ratio):
            issues.append(
                f"mask exceeds max NoData ratio: "
                f"{1.0 - valid_ratio:.2%} nodata "
                f"(limit {self._max_nodata_ratio:.2%})"
            )

        return {
            "issues": issues,
            "num_classes_present": len(present_classes),
            "valid_pixel_ratio": valid_ratio,
            "has_invalid_class_ids": has_invalid,
        }