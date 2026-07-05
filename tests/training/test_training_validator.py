"""
Tests for src/training/validator.py

Run:
    pytest tests/training/test_training_validator.py -v \
        --cov=src/training/validator --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.contracts import TransformSample
from src.training.validator import TransformValidationResult, TransformValidator


def _validator(**kw) -> TransformValidator:
    return TransformValidator(
        valid_class_ids={0, 1, 2, 3},
        check_metadata=True,
        nodata_class_id=255,
        **kw,
    )


def _good_sample() -> TransformSample:
    return TransformSample(
        image=np.zeros((4, 8, 8), dtype=np.float32),
        mask=np.zeros((8, 8), dtype=np.uint8),
        sample_id="p1",
        split="train",
    )


class TestTransformValidationResult:
    def test_is_valid_when_no_issues(self) -> None:
        assert TransformValidationResult([]).is_valid

    def test_is_invalid_when_issues_exist(self) -> None:
        assert not TransformValidationResult(["issue1"]).is_valid

    def test_issues_are_copy(self) -> None:
        r = TransformValidationResult(["a"])
        r.issues.append("b")
        assert len(r.issues) == 1   # original not mutated


class TestTransformValidator:
    def test_good_sample_is_valid(self) -> None:
        result = _validator().validate_sample(_good_sample())
        assert result.is_valid

    def test_wrong_image_ndim_detected(self) -> None:
        s = _good_sample()
        s.image = np.zeros((4, 8), dtype=np.float32)   # 2-D, wrong
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("3-D" in issue for issue in r.issues)

    def test_wrong_image_dtype_detected(self) -> None:
        s = _good_sample()
        s.image = np.zeros((4, 8, 8), dtype=np.float64)
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("float32" in issue for issue in r.issues)

    def test_nan_in_image_detected(self) -> None:
        s = _good_sample()
        s.image[0, 0, 0] = np.nan
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("NaN" in issue for issue in r.issues)

    def test_inf_in_image_detected(self) -> None:
        s = _good_sample()
        s.image[0, 0, 0] = np.inf
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("Inf" in issue for issue in r.issues)

    def test_wrong_mask_ndim_detected(self) -> None:
        s = _good_sample()
        s.mask = np.zeros((1, 8, 8), dtype=np.uint8)
        r = _validator().validate_sample(s)
        assert not r.is_valid

    def test_wrong_mask_dtype_detected(self) -> None:
        s = _good_sample()
        s.mask = np.zeros((8, 8), dtype=np.float32)
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("uint8" in issue for issue in r.issues)

    def test_spatial_mismatch_detected(self) -> None:
        s = _good_sample()
        s.mask = np.zeros((4, 4), dtype=np.uint8)   # image is (4,8,8)
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("mismatch" in issue or "dims" in issue for issue in r.issues)

    def test_invalid_class_id_detected(self) -> None:
        s = _good_sample()
        s.mask[0, 0] = 99   # not in {0,1,2,3} and not 255
        r = _validator().validate_sample(s)
        assert not r.is_valid
        assert any("invalid class" in issue for issue in r.issues)

    def test_nodata_class_id_is_allowed(self) -> None:
        s = _good_sample()
        s.mask[0, 0] = 255   # nodata -- always allowed
        r = _validator().validate_sample(s)
        assert r.is_valid

    def test_empty_sample_id_detected(self) -> None:
        s = _good_sample()
        s.sample_id = ""
        r = _validator().validate_sample(s)
        assert not r.is_valid

    def test_invalid_split_detected(self) -> None:
        s = _good_sample()
        s.split = "holdout"   # not in {train, validation, test}
        r = _validator().validate_sample(s)
        assert not r.is_valid

    def test_metadata_check_disabled(self) -> None:
        s = _good_sample()
        s.sample_id = ""
        s.split = "invalid_split"
        v = TransformValidator(valid_class_ids={0, 1, 2, 3}, check_metadata=False)
        r = v.validate_sample(s)
        # Metadata issues should be absent when check_metadata=False.
        assert not any("sample_id" in issue or "split" in issue for issue in r.issues)

    def test_valid_class_ids_respected(self) -> None:
        s = _good_sample()
        s.mask[0, 0] = 5   # class 5 is valid in extended schema
        v = TransformValidator(valid_class_ids={0, 1, 2, 3, 4, 5})
        r = v.validate_sample(s)
        assert r.is_valid
