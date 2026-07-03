"""
Dataset quality validation for the Dataset Assembly pipeline (Module 10).

DatasetValidator checks the assembled sample collection for structural
integrity: missing files, duplicate sample IDs, CRS consistency, and
minimum quality thresholds. Produces a DatasetValidationResult without
reading any GeoTIFF pixel data (file-existence checks only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import Config

from src.dataset.manifest import DatasetSample

__all__ = ["ValidationIssue", "DatasetValidationResult", "DatasetValidator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    """
    Immutable descriptor for one dataset-level validation issue.

    Attributes:
        sample_id:    Affected sample ID, or "" for global issues.
        issue_type:    Short issue identifier, e.g. "missing_patch_file".
        description:   Human-readable description.
    """

    sample_id:   str
    issue_type:   str
    description:  str


@dataclass(frozen=True)
class DatasetValidationResult:
    """
    Immutable result of DatasetValidator.validate().

    Attributes:
        is_valid:                 True if no blocking issues were found.
        total_samples:            Number of samples examined.
        valid_samples:             Number of samples that passed all checks.
        invalid_samples:           Number of samples rejected.
        issues:                    Ordered tuple of all issues found.
        duplicate_sample_ids:       Tuple of duplicate sample_id values.
        missing_patch_files:        Tuple of sample_ids with missing patches.
        missing_mask_files:          Tuple of sample_ids with missing masks.
        crs_values_found:            All distinct CRS strings in the dataset.
        crs_is_consistent:            True if all samples share one CRS.
        below_min_pixel_ratio_count:  Samples below min_valid_pixel_ratio.
        min_total_samples_met:         True if enough samples pass QC.
    """

    is_valid:                 bool
    total_samples:            int
    valid_samples:             int
    invalid_samples:           int
    issues:                    tuple[ValidationIssue, ...]
    duplicate_sample_ids:       tuple[str, ...]
    missing_patch_files:        tuple[str, ...]
    missing_mask_files:          tuple[str, ...]
    crs_values_found:            tuple[str, ...]
    crs_is_consistent:            bool
    below_min_pixel_ratio_count:  int
    min_total_samples_met:         bool


class DatasetValidator:
    """
    Validates the assembled dataset sample collection.

    Checks:
        - Duplicate sample IDs
        - Missing patch files (configurable, may be slow for large datasets)
        - Missing mask files
        - CRS consistency
        - Minimum valid pixel ratio
        - Minimum total sample count

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

        dataset_cfg = getattr(config, "dataset", None)
        quality_cfg = getattr(dataset_cfg, "quality", None)

        self._min_valid_pixel_ratio = float(
            getattr(quality_cfg, "min_valid_pixel_ratio", 0.5)
        )
        self._min_total_samples = int(
            getattr(dataset_cfg, "min_total_samples", 10)
        )

    def validate(
        self,
        samples: list[DatasetSample],
        check_files: bool = True,
    ) -> DatasetValidationResult:
        """
        Run all validation checks on the sample list.

        Args:
            samples:      List of DatasetSample objects to validate.
            check_files:  When True, verify that patch and mask files
                          exist on disk (may be slow for large datasets).

        Returns:
            DatasetValidationResult with all findings.
        """
        issues: list[ValidationIssue] = []
        invalid_ids: set[str]         = set()

        # Duplicate sample IDs
        seen_ids: dict[str, int] = {}
        for s in samples:
            seen_ids[s.sample_id] = seen_ids.get(s.sample_id, 0) + 1
        duplicates = tuple(k for k, v in seen_ids.items() if v > 1)
        for dup_id in duplicates:
            issues.append(ValidationIssue(
                sample_id=dup_id,
                issue_type="duplicate_sample_id",
                description=f"sample_id '{dup_id}' appears more than once",
            ))
            invalid_ids.add(dup_id)

        # File existence checks
        missing_patches: list[str] = []
        missing_masks:   list[str] = []
        if check_files:
            for s in samples:
                if not Path(s.patch_path).exists():
                    missing_patches.append(s.sample_id)
                    issues.append(ValidationIssue(
                        sample_id=s.sample_id,
                        issue_type="missing_patch_file",
                        description=f"patch not found: {s.patch_path}",
                    ))
                    invalid_ids.add(s.sample_id)
                if not Path(s.mask_path).exists():
                    missing_masks.append(s.sample_id)
                    issues.append(ValidationIssue(
                        sample_id=s.sample_id,
                        issue_type="missing_mask_file",
                        description=f"mask not found: {s.mask_path}",
                    ))
                    invalid_ids.add(s.sample_id)

        # CRS consistency
        crs_values = tuple(sorted({s.crs for s in samples}))
        crs_consistent = len(crs_values) <= 1
        if not crs_consistent:
            issues.append(ValidationIssue(
                sample_id="",
                issue_type="inconsistent_crs",
                description=f"multiple CRS values found: {list(crs_values)}",
            ))

        # Minimum valid pixel ratio
        below_ratio_ids = [
            s.sample_id for s in samples
            if s.label_valid_pixel_ratio < self._min_valid_pixel_ratio
        ]
        for sid in below_ratio_ids:
            issues.append(ValidationIssue(
                sample_id=sid,
                issue_type="below_min_valid_pixel_ratio",
                description=(
                    f"label_valid_pixel_ratio below threshold "
                    f"{self._min_valid_pixel_ratio}"
                ),
            ))
            invalid_ids.add(sid)

        valid_samples = len(samples) - len(invalid_ids)
        min_total_met = valid_samples >= self._min_total_samples

        if not min_total_met:
            issues.append(ValidationIssue(
                sample_id="",
                issue_type="insufficient_samples",
                description=(
                    f"valid sample count {valid_samples} is below "
                    f"minimum required {self._min_total_samples}"
                ),
            ))

        is_valid = (
            len(duplicates) == 0
            and len(missing_patches) == 0
            and len(missing_masks) == 0
            and crs_consistent
            and min_total_met
        )

        self._logger.info(
            "Dataset validation complete. is_valid=%s, valid=%d/%d, issues=%d",
            is_valid, valid_samples, len(samples), len(issues),
        )

        return DatasetValidationResult(
            is_valid=is_valid,
            total_samples=len(samples),
            valid_samples=valid_samples,
            invalid_samples=len(invalid_ids),
            issues=tuple(issues),
            duplicate_sample_ids=duplicates,
            missing_patch_files=tuple(missing_patches),
            missing_mask_files=tuple(missing_masks),
            crs_values_found=crs_values,
            crs_is_consistent=crs_consistent,
            below_min_pixel_ratio_count=len(below_ratio_ids),
            min_total_samples_met=min_total_met,
        )