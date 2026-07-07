"""Tests for pipeline/contracts.py and pipeline/validator.py"""
from __future__ import annotations
import json
import types
import pytest
from src.pipeline.contracts import (
    AOIConfig, PipelineConfig, PipelineResult, StageResult, VALID_MODES,
)
from src.pipeline.validator import PipelineValidator, PipelineValidationResult


# ==============================================================================
# PipelineConfig
# ==============================================================================

class TestPipelineConfig:
    def test_frozen(self):
        cfg = PipelineConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.mode = "inference"  # type: ignore[misc]

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.mode    == "full"
        assert cfg.dry_run is False
        assert cfg.seed    == 42

    def test_from_config_reads_pipeline_section(self):
        class _P:
            mode="inference"; run_id="r1"; dry_run=True
            output_dir="out"; aoi_ids=["A1","A2"]; resume_from=""
        class _R:
            class reproducibility: seed = 7
            class device: device = "cpu"
            pipeline = _P()
            aoi      = types.SimpleNamespace(id="default")
        cfg = PipelineConfig.from_config(_R(), cli_overrides={"device": "cuda"})
        assert cfg.mode       == "inference"
        assert cfg.run_id     == "r1"
        assert cfg.dry_run    is True
        assert cfg.aoi_ids    == ("A1","A2")
        assert cfg.device     == "cuda"   # CLI override wins

    def test_from_config_cli_overrides_mode(self):
        class _Cfg:
            class reproducibility: seed = 42
            class device: device = "cpu"
            aoi = types.SimpleNamespace(id="default")
        cfg = PipelineConfig.from_config(_Cfg(), {"mode": "reporting"})
        assert cfg.mode == "reporting"

    def test_from_config_no_pipeline_section_uses_defaults(self):
        class _Cfg:
            class reproducibility: seed = 42
            class device: device = "cpu"
            aoi = types.SimpleNamespace(id="default")
        cfg = PipelineConfig.from_config(_Cfg())
        assert cfg.mode == "full"

    def test_aoi_ids_single_fallback(self):
        class _Cfg:
            class reproducibility: seed = 42
            class device: device = "cpu"
            aoi = types.SimpleNamespace(id="kosi")
        cfg = PipelineConfig.from_config(_Cfg())
        assert "kosi" in cfg.aoi_ids


class TestAOIConfig:
    def test_frozen(self):
        aoi = AOIConfig("a1")
        with pytest.raises((AttributeError, TypeError)):
            aoi.aoi_id = "other"  # type: ignore[misc]

    def test_is_complete_true(self):
        aoi = AOIConfig("a1", 87.0, 26.0, 87.5, 26.5)
        assert aoi.is_complete is True

    def test_is_complete_false_when_null(self):
        assert AOIConfig("a1").is_complete is False
        assert AOIConfig("a1", 87.0, None, 87.5, 26.5).is_complete is False

    def test_bbox_returns_tuple(self):
        aoi = AOIConfig("a1", 87.0, 26.0, 87.5, 26.5)
        assert aoi.bbox == (87.0, 26.0, 87.5, 26.5)

    def test_bbox_none_when_incomplete(self):
        assert AOIConfig("a1").bbox is None

    def test_as_dict_json_serialisable(self):
        aoi = AOIConfig("a1", 87.0, 26.0, 87.5, 26.5, "2023-01-01", "2023-06-30")
        assert json.dumps(aoi.as_dict())


class TestStageResult:
    def test_frozen(self):
        r = StageResult("training", "a1", True)
        with pytest.raises((AttributeError, TypeError)):
            r.success = False  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        r = StageResult("inference", "kosi", True, duration_s=5.2)
        assert json.dumps(r.as_dict())

    def test_failed_stage(self):
        r = StageResult("training", "a1", False, error="RuntimeError: OOM")
        assert r.success is False
        assert "OOM" in r.error


class TestPipelineResult:
    def _make(self):
        stages = (
            StageResult("training", "a1", True, duration_s=10.0),
            StageResult("evaluation", "a1", True, duration_s=2.0),
        )
        return PipelineResult(
            run_id="r1", mode="training", aoi_ids=("a1",),
            stage_results=stages, success=True, total_duration_s=12.0,
            output_dirs={"a1": "/out/r1/a1"}, warnings=(), operations_log=("s1",),
            dry_run=False, num_stages=2, num_failed=0,
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.success = False  # type: ignore[misc]

    def test_summary_lines_ascii(self):
        lines = self._make().summary_lines()
        assert all(ord(c) < 128 for l in lines for c in l)

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_failed_stages(self):
        r = self._make()
        assert r.failed_stages() == []

    def test_failed_stages_returns_failures(self):
        stages = (
            StageResult("training",   "a1", True),
            StageResult("evaluation", "a1", False, error="fail"),
        )
        r = PipelineResult(
            run_id="r1", mode="evaluation", aoi_ids=("a1",),
            stage_results=stages, success=False, total_duration_s=5.0,
            output_dirs={}, warnings=(), operations_log=(), dry_run=False,
            num_stages=2, num_failed=1,
        )
        failed = r.failed_stages()
        assert len(failed) == 1
        assert failed[0].stage == "evaluation"

    def test_stages_for_aoi(self):
        stages = (
            StageResult("training", "a1", True),
            StageResult("training", "a2", True),
        )
        r = PipelineResult(
            run_id="r1", mode="training", aoi_ids=("a1","a2"),
            stage_results=stages, success=True, total_duration_s=5.0,
            output_dirs={}, warnings=(), operations_log=(), dry_run=False,
            num_stages=2, num_failed=0,
        )
        assert len(r.stages_for_aoi("a1")) == 1
        assert len(r.stages_for_aoi("a2")) == 1
        assert len(r.stages_for_aoi("a3")) == 0


# ==============================================================================
# PipelineValidator
# ==============================================================================

def _config(
    num_channels=11, in_channels=11, num_classes_model=4,
    num_classes_cfg=4, n_names=4, n_colors=4,
    pg_patch=256, inf_patch=256, max_tile_pixels=250000, spectral_band_count=11,
):
    """Build a minimal config stub that passes all checks."""
    bands = [types.SimpleNamespace(name=f"B{i}") for i in range(spectral_band_count)]
    return types.SimpleNamespace(
        num_channels   = num_channels,
        spectral_bands = bands,
        model          = types.SimpleNamespace(in_channels=in_channels,
                                               num_classes=num_classes_model),
        classes        = types.SimpleNamespace(
            num_classes = num_classes_cfg,
            names       = [f"class_{i}" for i in range(n_names)],
            colors      = {f"c{i}": [0,0,0] for i in range(n_colors)},
        ),
        patch_generation = types.SimpleNamespace(patch_size=pg_patch),
        inference        = types.SimpleNamespace(patch_size=inf_patch),
        export           = types.SimpleNamespace(max_tile_pixels=max_tile_pixels),
    )


def _pcfg(**kw) -> PipelineConfig:
    defaults = dict(mode="full", run_id="r1", dry_run=False,
                    output_dir="out", aoi_ids=("a1",), device="cpu",
                    seed=42, resume_from="")
    defaults.update(kw)
    return PipelineConfig(**defaults)


def _aoi(complete=True, start="2023-01-01", end="2023-06-30") -> AOIConfig:
    if complete:
        return AOIConfig("a1", 87.0, 26.0, 87.5, 26.5, start, end)
    return AOIConfig("a1", start_date=start, end_date=end)


class TestPipelineValidator:
    def test_valid_config_passes(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(), _config(), [_aoi()])
        assert r.is_valid

    def test_invalid_mode_detected(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(mode="unknown"), _config(), [_aoi()])
        assert not r.is_valid
        assert any("mode" in i for i in r.issues)

    def test_channel_mismatch_detected(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(), _config(num_channels=11, in_channels=12), [_aoi()])
        assert not r.is_valid
        assert any("in_channels" in i for i in r.issues)

    def test_spectral_bands_vs_num_channels_mismatch(self):
        v = PipelineValidator()
        # 11 bands but num_channels says 12
        r = v.validate(_pcfg(), _config(num_channels=12, in_channels=12), [_aoi()])
        assert not r.is_valid
        assert any("spectral_bands" in i or "num_channels" in i for i in r.issues)

    def test_num_classes_mismatch_detected(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(),
                       _config(num_classes_model=4, num_classes_cfg=3),
                       [_aoi()])
        assert not r.is_valid

    def test_class_names_count_mismatch(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(), _config(n_names=3), [_aoi()])
        assert not r.is_valid
        assert any("names" in i for i in r.issues)

    def test_class_colors_count_mismatch(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(), _config(n_colors=3), [_aoi()])
        assert not r.is_valid
        assert any("colors" in i for i in r.issues)

    def test_patch_size_mismatch_detected(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(), _config(pg_patch=256, inf_patch=512), [_aoi()])
        assert not r.is_valid
        assert any("patch_size" in i for i in r.issues)

    def test_incomplete_aoi_detected(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(mode="full"), _config(), [_aoi(complete=False)])
        assert not r.is_valid
        assert any("null" in i or "coordinate" in i for i in r.issues)

    def test_invalid_lon_range_detected(self):
        v    = PipelineValidator()
        bad  = AOIConfig("a1", 88.0, 26.0, 87.0, 26.5, "2023-01-01", "2023-06-30")
        r    = v.validate(_pcfg(mode="full"), _config(), [bad])
        assert not r.is_valid
        assert any("min_lon" in i or "max_lon" in i for i in r.issues)

    def test_invalid_lat_range_detected(self):
        v   = PipelineValidator()
        bad = AOIConfig("a1", 87.0, 27.0, 87.5, 26.0, "2023-01-01", "2023-06-30")
        r   = v.validate(_pcfg(mode="full"), _config(), [bad])
        assert not r.is_valid

    def test_date_start_after_end_detected(self):
        v   = PipelineValidator()
        aoi = _aoi(start="2023-12-01", end="2023-01-01")
        r   = v.validate(_pcfg(), _config(), [aoi])
        assert not r.is_valid
        assert any("date_range" in i or "start" in i for i in r.issues)

    def test_partial_dates_detected(self):
        v   = PipelineValidator()
        aoi = AOIConfig("a1", 87.0, 26.0, 87.5, 26.5, "2023-01-01", "")
        r   = v.validate(_pcfg(), _config(), [aoi])
        assert not r.is_valid

    def test_missing_resume_checkpoint_detected(self, tmp_path):
        v = PipelineValidator()
        r = v.validate(
            _pcfg(resume_from=str(tmp_path / "nonexistent.pt")),
            _config(), [_aoi()]
        )
        assert not r.is_valid
        assert any("resume_from" in i for i in r.issues)

    def test_tile_pixels_zero_is_warning(self):
        v = PipelineValidator()
        r = v.validate(_pcfg(), _config(max_tile_pixels=0), [_aoi()])
        assert any("max_tile_pixels" in w for w in r.warnings)

    def test_non_full_mode_skips_aoi_check(self):
        """Inference mode does not require a complete AOI."""
        v   = PipelineValidator()
        aoi = AOIConfig("a1")   # incomplete — no coordinates
        r   = v.validate(_pcfg(mode="inference"), _config(), [aoi])
        # Should not flag the incomplete AOI for non-GEE modes.
        aoi_issues = [i for i in r.issues if "coordinate" in i or "null" in i]
        assert len(aoi_issues) == 0

    def test_validation_result_issues_are_copy(self):
        r = PipelineValidationResult(["a"], [])
        r.issues.append("b")
        assert len(r.issues) == 1

    def test_valid_modes_constant(self):
        assert "full" in VALID_MODES
        assert "training" in VALID_MODES
        assert "inference" in VALID_MODES
        assert "reporting" in VALID_MODES
