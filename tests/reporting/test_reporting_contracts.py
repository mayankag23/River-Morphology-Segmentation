"""Tests for src/reporting/contracts.py"""
from __future__ import annotations
import json
import pytest
from src.reporting.contracts import (
    ArtifactRecord, ExperimentMetadata, ReportManifest,
    ReportingConfig, ReportResult,
)


class TestReportingConfig:
    def test_frozen(self):
        cfg = ReportingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.output_dir = "new"  # type: ignore[misc]

    def test_defaults(self):
        cfg = ReportingConfig()
        assert cfg.output_dir        == "reports"
        assert cfg.export_markdown   is True
        assert cfg.export_json       is True
        assert cfg.export_csv        is True
        assert cfg.export_pdf        is False
        assert cfg.report_version    == "1.0"

    def test_from_config_reads_values(self):
        class _R:
            output_dir="out"; report_name="rpt"; export_markdown=True
            export_json=False; export_csv=True; export_pdf=False
            include_figures=True; include_config=True; include_evaluation=True
            include_inference=True; include_morphology=True; include_visualization=True
            project_name="Test"; project_version="2.0"; author="Alice"
            institution="Uni"; report_version="1.0"; git_commit="abc123"
            experiment_id="exp_001"
        class _Cfg:
            reporting = _R()
        cfg = ReportingConfig.from_config(_Cfg())
        assert cfg.output_dir      == "out"
        assert cfg.export_json     is False
        assert cfg.project_version == "2.0"
        assert cfg.git_commit      == "abc123"
        assert cfg.experiment_id   == "exp_001"

    def test_from_config_no_section_returns_defaults(self):
        class _Cfg: pass
        assert ReportingConfig.from_config(_Cfg()) == ReportingConfig()


class TestArtifactRecord:
    def test_frozen(self):
        r = ArtifactRecord("id", "json", "/out/file.json")
        with pytest.raises((AttributeError, TypeError)):
            r.path = "/other"  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        r = ArtifactRecord("id1", "report", "/out/r.md", 1024, "Main report", "Module 19")
        d = r.as_dict()
        assert json.dumps(d)
        assert d["artifact_id"]   == "id1"
        assert d["size_bytes"]    == 1024

    def test_default_size_bytes(self):
        r = ArtifactRecord("id", "json", "/out/f.json")
        assert r.size_bytes == -1


class TestExperimentMetadata:
    def _make(self):
        return ExperimentMetadata(
            experiment_id="exp_001", run_timestamp="2024-01-01T00:00:00+00:00",
            architecture="unetplusplus", checkpoint_epoch=10,
            checkpoint_path="/ckpts/best.pt", num_classes=4,
            class_names=("bg","water","sand","veg"),
            git_commit="abc1234", project_name="Test", project_version="1.0",
            author="Alice", institution="Uni", total_duration_s=120.5,
            config_snapshot={"key": "value"},
        )

    def test_frozen(self):
        m = self._make()
        with pytest.raises((AttributeError, TypeError)):
            m.experiment_id = "new"  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_as_dict_fields(self):
        d = self._make().as_dict()
        assert d["experiment_id"]    == "exp_001"
        assert d["architecture"]     == "unetplusplus"
        assert d["checkpoint_epoch"] == 10
        assert d["class_names"]      == ["bg","water","sand","veg"]


class TestReportManifest:
    def _make(self):
        artifacts = (
            ArtifactRecord("a1", "report", "/out/r.md", 512),
            ArtifactRecord("a2", "json", "/out/r.json", 1024),
        )
        return ReportManifest(
            manifest_version="1.0", experiment_id="exp_001",
            report_timestamp="2024-01-01T00:00:00+00:00",
            artifacts=artifacts, num_artifacts=2,
            report_files=("/out/r.md", "/out/r.json"), total_size_bytes=1536,
        )

    def test_frozen(self):
        m = self._make()
        with pytest.raises((AttributeError, TypeError)):
            m.num_artifacts = 99  # type: ignore[misc]

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_as_dict_artifact_count(self):
        d = self._make().as_dict()
        assert d["num_artifacts"] == 2
        assert len(d["artifacts"]) == 2


class TestReportResult:
    def _make(self):
        from src.reporting.contracts import ExperimentMetadata, ReportManifest, ArtifactRecord
        exp = ExperimentMetadata(
            experiment_id="exp_001", run_timestamp="2024-01-01T00:00:00+00:00",
            architecture="unetplusplus", checkpoint_epoch=5,
            checkpoint_path="/ckpt.pt", num_classes=4,
            class_names=("bg","water","sand","veg"),
            git_commit="", project_name="Test", project_version="1.0",
            author="", institution="", total_duration_s=10.0,
            config_snapshot={},
        )
        manifest = ReportManifest(
            manifest_version="1.0", experiment_id="exp_001",
            report_timestamp="2024-01-01T00:00:00+00:00",
            artifacts=(), num_artifacts=0, report_files=(), total_size_bytes=0,
        )
        return ReportResult(
            experiment=exp, manifest=manifest, report_files=("/out/r.md",),
            num_report_files=1, num_artifacts=3, report_version="1.0",
            operations_log=("step1", "step2"), generation_time_s=0.5,
            evaluation_summary={"mean_iou": 0.82},
            inference_summary={"num_samples": 10},
            morphology_summary={"mean_water_fraction": 0.4},
            visualization_summary={"num_figures": 12},
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.num_artifacts = 99  # type: ignore[misc]

    def test_summary_lines_ascii(self):
        lines = self._make().summary_lines()
        assert len(lines) > 0
        assert all(ord(c) < 128 for l in lines for c in l)

    def test_as_dict_json_serialisable(self):
        assert json.dumps(self._make().as_dict())

    def test_as_dict_has_all_sections(self):
        d = self._make().as_dict()
        for key in ("experiment", "manifest", "evaluation_summary",
                    "inference_summary", "morphology_summary", "visualization_summary"):
            assert key in d