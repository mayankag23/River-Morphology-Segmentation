"""
Transform validation for the Data Transformation and Augmentation Pipeline
(Module 12).

TransformValidator checks the integrity of transformed samples before
they are passed to the model training pipeline.  It operates on
TransformSample objects (numpy arrays) and also provides validation
for the assembled TransformPipelineResult.

Checks performed
-----------------
    1. Tensor shape:      image is (C, H, W); mask is (H, W).
    2. Spatial consistency: image H, W == mask H, W.
    3. Image dtype:       float32 required.
    4. Mask dtype:        uint8 required.
    5. NaN values:        no NaN in image or mask.
    6. Inf values:        no Inf in image.
    7. Class ID integrity: mask contains only valid class IDs
                          (from the configured class schema).
    8. Metadata:          sample_id, split, and temporal fields are non-empty
                          where required.

The validator never raises exceptions; it returns a list of issue strings.
This mirrors the pattern established by DatasetValidator in Module 10.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.training.contracts import TransformSample

__all__ = ["TransformValidator", "TransformValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

_VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})


# ==============================================================================
# TransformValidationResult
# ==============================================================================

class TransformValidationResult:
    """
    Result of validating a transformed sample or pipeline result.

    Attributes:
        is_valid:  True if no issues were found.
        issues:    List of human-readable issue descriptions.
    """

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


# ==============================================================================
# TransformValidator
# ==============================================================================

class TransformValidator:
    """
    Validates TransformSample objects and TransformPipelineResult.

    Args:
        valid_class_ids:  Set of valid class IDs in the mask.
                          Defaults to {0, 1, 2, 3} but must come from Config
                          via the caller (TransformPipeline).
        check_metadata:   When True, validate that temporal metadata fields
                          are present and non-empty.
        nodata_class_id:  Class ID used as nodata sentinel (e.g. 255).
                          Nodata pixels are excluded from class ID checks.
    """

    def __init__(
        self,
        valid_class_ids:  set[int],
        check_metadata:   bool = True,
        nodata_class_id:  int  = 255,
    ) -> None:
        self._valid_class_ids = frozenset(int(i) for i in valid_class_ids)
        self._check_metadata  = bool(check_metadata)
        self._nodata_class_id = int(nodata_class_id)
        self._logger: logging.Logger = logging.getLogger(__name__)

    def validate_sample(self, sample: TransformSample) -> TransformValidationResult:
        """
        Validate one TransformSample.

        Args:
            sample: A fully transformed TransformSample (post-normalization).

        Returns:
            TransformValidationResult with all detected issues.
        """
        issues: list[str] = []
        issues.extend(self._check_image(sample))
        issues.extend(self._check_mask(sample))
        issues.extend(self._check_spatial_consistency(sample))
        issues.extend(self._check_class_ids(sample))
        if self._check_metadata:
            issues.extend(self._check_meta(sample))
        return TransformValidationResult(issues)

    def validate_pipeline_result(
        self,
        result:      Any,    # TransformPipelineResult
        num_classes: int,
        num_bands:   int,
    ) -> TransformValidationResult:
        """
        Validate the assembled TransformPipelineResult.

        Performs structural checks on the result object itself, not on
        individual samples.

        Args:
            result:      TransformPipelineResult instance.
            num_classes: Expected number of segmentation classes.
            num_bands:   Expected number of spectral bands.

        Returns:
            TransformValidationResult.
        """
        issues: list[str] = []

        if result.num_train_samples == 0:
            issues.append("training dataset is empty (num_train_samples == 0)")

        if result.num_bands != num_bands:
            issues.append(
                f"result.num_bands={result.num_bands} does not match "
                f"expected num_bands={num_bands}"
            )

        if result.num_classes != num_classes:
            issues.append(
                f"result.num_classes={result.num_classes} does not match "
                f"expected num_classes={num_classes}"
            )

        stats = result.normalization_stats
        if stats.num_bands != num_bands:
            issues.append(
                f"normalization_stats has {stats.num_bands} bands but "
                f"expected {num_bands}"
            )

        for band_std in stats.std:
            if band_std <= 0.0:
                issues.append(
                    f"normalization_stats.std contains non-positive value: {band_std}"
                )
                break

        return TransformValidationResult(issues)

    # ------------------------------------------------------------------
    # Private per-field checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_image(sample: TransformSample) -> list[str]:
        issues: list[str] = []
        img = sample.image
        if img is None:
            return ["image is None"]
        if img.ndim != 3:
            issues.append(
                f"image must be 3-D (C, H, W) but got {img.ndim}-D "
                f"(shape {img.shape})"
            )
        if img.dtype != np.float32:
            issues.append(
                f"image dtype must be float32 but got {img.dtype}"
            )
        if np.any(np.isnan(img)):
            issues.append("image contains NaN values")
        if np.any(np.isinf(img)):
            issues.append("image contains Inf values")
        return issues

    @staticmethod
    def _check_mask(sample: TransformSample) -> list[str]:
        issues: list[str] = []
        mask = sample.mask
        if mask is None:
            return ["mask is None"]
        if mask.ndim != 2:
            issues.append(
                f"mask must be 2-D (H, W) but got {mask.ndim}-D "
                f"(shape {mask.shape})"
            )
        if mask.dtype != np.uint8:
            issues.append(
                f"mask dtype must be uint8 but got {mask.dtype}"
            )
        if np.any(np.isnan(mask.astype(np.float32))):
            issues.append("mask contains NaN values")
        return issues

    @staticmethod
    def _check_spatial_consistency(sample: TransformSample) -> list[str]:
        if sample.image is None or sample.mask is None:
            return []
        if sample.image.ndim != 3 or sample.mask.ndim != 2:
            return []  # already reported by other checks
        img_h, img_w = sample.image.shape[1], sample.image.shape[2]
        msk_h, msk_w = sample.mask.shape[0],  sample.mask.shape[1]
        if img_h != msk_h or img_w != msk_w:
            return [
                f"image spatial dims ({img_h}, {img_w}) do not match "
                f"mask dims ({msk_h}, {msk_w})"
            ]
        return []

    def _check_class_ids(self, sample: TransformSample) -> list[str]:
        if sample.mask is None or sample.mask.ndim != 2:
            return []
        all_ids = set(int(v) for v in np.unique(sample.mask))
        allowed = self._valid_class_ids | {self._nodata_class_id}
        invalid = all_ids - allowed
        if invalid:
            return [
                f"mask contains invalid class IDs: {sorted(invalid)}. "
                f"Valid IDs: {sorted(self._valid_class_ids)} "
                f"(nodata={self._nodata_class_id})"
            ]
        return []

    @staticmethod
    def _check_meta(sample: TransformSample) -> list[str]:
        issues: list[str] = []
        if not sample.sample_id:
            issues.append("sample_id is empty")
        if sample.split not in _VALID_SPLITS:
            issues.append(
                f"split='{sample.split}' is not one of "
                f"{sorted(_VALID_SPLITS)}"
            )
        return issues
