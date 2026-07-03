"""
Data leakage detection for the Dataset Assembly pipeline (Module 10).

DataLeakageDetector verifies that no sample (patch) or scene appears in
more than one split. Scene-level splitting (implemented in DatasetSplitter)
is the primary prevention mechanism; this detector is the verification step.

Leakage types detected:
    - Patch leakage: same patch_id in multiple splits (should never occur
      with scene-level splitting).
    - Scene leakage: same scene_id in multiple splits (should never occur).
    - Potential overlap: patches from the same scene are adjacent
      (stride < patch_size); since all such patches are in one split,
      this is automatically safe and reported as OK.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.dataset.manifest import DatasetSample

__all__ = ["LeakageRecord", "LeakageDetectionResult", "DataLeakageDetector"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeakageRecord:
    """
    Immutable record of one potential leakage violation.

    Attributes:
        sample_id:    Sample identifier involved in the violation.
        scene_id:      Source scene of the sample.
        splits_found:   All splits this sample appears in (should be exactly one).
        is_violation:   True if the same sample appears in more than one split.
    """

    sample_id:    str
    scene_id:      str
    splits_found:   tuple[str, ...]
    is_violation:   bool


@dataclass(frozen=True)
class LeakageDetectionResult:
    """
    Immutable result of DataLeakageDetector.detect().

    Attributes:
        has_leakage:              True if any violation was found.
        total_samples_checked:     Total samples across all splits.
        total_scenes_checked:       Number of distinct scene_ids checked.
        patch_violations:           Tuple of patch_ids appearing in multiple splits.
        scene_violations:           Tuple of scene_ids appearing in multiple splits.
        violation_records:          All LeakageRecord instances with is_violation=True.
    """

    has_leakage:              bool
    total_samples_checked:     int
    total_scenes_checked:       int
    patch_violations:           tuple[str, ...]
    scene_violations:           tuple[str, ...]
    violation_records:           tuple[LeakageRecord, ...]


class DataLeakageDetector:
    """
    Verifies split integrity by checking for cross-split sample and scene overlap.

    Usage:
        detector = DataLeakageDetector()
        result = detector.detect(
            train=split_result.train_samples,
            validation=split_result.validation_samples,
            test=split_result.test_samples,
        )
        assert not result.has_leakage
    """

    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)

    def detect(
        self,
        train:      list[DatasetSample] | tuple[DatasetSample, ...],
        validation: list[DatasetSample] | tuple[DatasetSample, ...],
        test:       list[DatasetSample] | tuple[DatasetSample, ...],
    ) -> LeakageDetectionResult:
        """
        Detect leakage across the three splits.

        Args:
            train:      Training split samples.
            validation:  Validation split samples.
            test:        Test split samples.

        Returns:
            LeakageDetectionResult indicating whether any violations exist.
        """
        splits = {
            "train":      list(train),
            "validation": list(validation),
            "test":       list(test),
        }
        all_samples = list(train) + list(validation) + list(test)

        # Map sample_id -> set of splits it appears in
        sample_to_splits: dict[str, set[str]] = {}
        scene_to_splits:  dict[str, set[str]] = {}

        for split_name, split_samples in splits.items():
            for sample in split_samples:
                sample_to_splits.setdefault(sample.sample_id, set()).add(split_name)
                scene_to_splits.setdefault(sample.scene_id, set()).add(split_name)

        # Find violations
        violation_records: list[LeakageRecord] = []
        patch_violations:  list[str]           = []
        scene_violations:  list[str]           = []

        for sample_id, found_splits in sample_to_splits.items():
            if len(found_splits) > 1:
                patch_violations.append(sample_id)
                # Find scene_id for this sample
                for s in all_samples:
                    if s.sample_id == sample_id:
                        violation_records.append(LeakageRecord(
                            sample_id=sample_id,
                            scene_id=s.scene_id,
                            splits_found=tuple(sorted(found_splits)),
                            is_violation=True,
                        ))
                        break

        for scene_id, found_splits in scene_to_splits.items():
            if len(found_splits) > 1:
                scene_violations.append(scene_id)
                if scene_id not in {r.scene_id for r in violation_records}:
                    violation_records.append(LeakageRecord(
                        sample_id="",
                        scene_id=scene_id,
                        splits_found=tuple(sorted(found_splits)),
                        is_violation=True,
                    ))

        has_leakage = len(patch_violations) > 0 or len(scene_violations) > 0
        if has_leakage:
            self._logger.error(
                "Data leakage detected! patch_violations=%d, scene_violations=%d",
                len(patch_violations), len(scene_violations),
            )
        else:
            self._logger.info(
                "Leakage check passed: %d samples, %d scenes -- no violations.",
                len(all_samples), len(scene_to_splits),
            )

        return LeakageDetectionResult(
            has_leakage=has_leakage,
            total_samples_checked=len(all_samples),
            total_scenes_checked=len(scene_to_splits),
            patch_violations=tuple(sorted(patch_violations)),
            scene_violations=tuple(sorted(scene_violations)),
            violation_records=tuple(violation_records),
        )