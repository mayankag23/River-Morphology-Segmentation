"""Tests for artifact.py, experiment.py, and manifest.py"""
from __future__ import annotations
import json
import types
import pytest
from pathlib import Path

from src.reporting.contracts import ArtifactRecord, ReportingConfig
from src.reporting.artifact import ArtifactManager, _file_size
from src.reporting.experiment import ExperimentManager, _generate_id, _read_git_commit
from src.reporting.manifest import ManifestManager


def _cfg(**kw) -> ReportingConfig:
    defaults = dict(
        output_dir="reports", report_name="test_report",
        export_markdown=True, export_json=True, export_csv=True, export_pdf=False,
        include_figures=True, include_config=True, include_evaluation=True,
        include_inference=True, include_morphology=True, include_visualization=True,
        project_name="Test Project", project_version="1.0.0",
        author="Test Author", institution="Test Uni",
        report_version="1.0", git_commit="", experiment_id="",
    )
    defaults.update(kw)
    return ReportingConfig(**defaults)


# ==============================================================================
# ArtifactManager
# ==============================================================================

class TestArtifactManager:
    def test_register_returns_artifact_record(self):
        mgr    = ArtifactManager("exp_001")
        record = mgr.register("/out/file.json", "json", "Test file", "Module 15")
        assert isinstance(record, ArtifactRecord)
        assert record.artifact_type == "json"
        assert record.path          == "/out/file.json"

    def test_register_increments_counter(self):
        mgr = ArtifactManager("exp")
        mgr.register("/a.json", "json")
        mgr.register("/b.md",   "markdown")
        assert mgr.num_artifacts() == 2

    def test_all_records_returns_tuple(self):
        mgr = ArtifactManager("exp")
        mgr.register("/a.json", "json")
        r = mgr.all_records()
        assert isinstance(r, tuple)
        assert len(r) == 1

    def test_by_type_filters_correctly(self):
        mgr = ArtifactManager("exp")
        mgr.register("/a.json", "json")
        mgr.register("/b.md",   "markdown")
        mgr.register("/c.json", "json")
        assert len(mgr.by_type("json"))     == 2
        assert len(mgr.by_type("markdown")) == 1
        assert len(mgr.by_type("figure"))   == 0

    def test_total_size_bytes_sums_existing(self, tmp_path):
        f1 = tmp_path / "a.txt"; f1.write_text("hello")
        f2 = tmp_path / "b.txt"; f2.write_text("world")
        mgr = ArtifactManager("exp")
        mgr.register(str(f1), "text")
        mgr.register(str(f2), "text")
        assert mgr.total_size_bytes() > 0

    def test_total_size_excludes_minus_one(self):
        mgr = ArtifactManager("exp")
        mgr.register("/nonexistent.json", "json")   # size_bytes = -1
        assert mgr.total_size_bytes() == 0

    def test_register_without_experiment_id(self):
        mgr    = ArtifactManager()
        record = mgr.register("/f.csv", "csv")
        assert record.artifact_type == "csv"

    def test_register_from_visualization_result_with_paths(self, tmp_path):
        f  = tmp_path / "fig.png"; f.write_bytes(b"png")
        spec = types.SimpleNamespace(
            figure_id="mask_p001", figure_type="mask",
            sample_id="p001", acquisition_date="2023-01-01",
            export_paths=[str(f)],
        )
        viz = types.SimpleNamespace(figures=(spec,))
        mgr = ArtifactManager("exp")
        mgr.register_from_visualization_result(viz)
        assert mgr.num_artifacts() == 1
        assert mgr.by_type("figure")[0].path == str(f)

    def test_register_from_visualization_result_empty(self):
        viz = types.SimpleNamespace(figures=())
        mgr = ArtifactManager("exp")
        mgr.register_from_visualization_result(viz)
        assert mgr.num_artifacts() == 0

    def test_register_from_inference_result_with_paths(self, tmp_path):
        f    = tmp_path / "pred.npy"; f.write_bytes(b"npy")
        pred = types.SimpleNamespace(sample_id="p001", exported_paths=[str(f)])
        inf  = types.SimpleNamespace(predictions=(pred,))
        mgr  = ArtifactManager("exp")
        mgr.register_from_inference_result(inf)
        assert mgr.num_artifacts() == 1
        assert mgr.by_type("prediction")[0].path == str(f)

    def test_register_from_inference_result_empty_predictions(self):
        inf = types.SimpleNamespace(predictions=())
        mgr = ArtifactManager("exp")
        mgr.register_from_inference_result(inf)
        assert mgr.num_artifacts() == 0

    def test_artifact_id_contains_type_and_counter(self):
        mgr    = ArtifactManager("myexp")
        record = mgr.register("/f.json", "json")
        assert "json" in record.artifact_id
        assert "myexp" in record.artifact_id

    def test_metadata_preserved(self):
        mgr    = ArtifactManager("exp")
        record = mgr.register("/f.json", "json", metadata={"key": "val"})
        assert record.metadata["key"] == "val"


class TestFileSizeHelper:
    def test_returns_size_for_existing_file(self, tmp_path):
        f = tmp_path / "t.txt"; f.write_text("hello")
        assert _file_size(str(f)) == 5

    def test_returns_minus_one_for_missing_file(self):
        assert _file_size("/nonexistent/file.txt") == -1


# ==============================================================================
# ExperimentManager
# ==============================================================================

class TestExperimentManager:
    def _inf_result(self):
        ckpt = types.SimpleNamespace(
            epoch=10, train_loss=0.3, val_loss=0.25,
            architecture="unetplusplus", num_classes=4, in_channels=12,
            checkpoint_path="/ckpts/best.pt",
        )
        inf_cfg = types.SimpleNamespace(
            device="cpu", batch_size=8, probability_mode="softmax",
            confidence_strategy="max_probability", checkpoint_strategy="best",
        )
        return types.SimpleNamespace(
            architecture="unetplusplus", class_names=("bg","water","sand","veg"),
            num_classes=4, num_samples=10, device_used="cpu",
            checkpoint_meta=ckpt, inference_config=inf_cfg,
            mean_confidence=0.8, total_inference_s=5.2,
        )

    def _eval_result(self):
        return types.SimpleNamespace(
            architecture="unetplusplus", split="test", ignore_index=255,
        )

    def test_build_returns_experiment_metadata(self):
        from src.reporting.contracts import ExperimentMetadata
        mgr = ExperimentManager(_cfg())
        exp = mgr.build()
        assert isinstance(exp, ExperimentMetadata)

    def test_experiment_id_auto_generated(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build()
        assert exp.experiment_id.startswith("exp_")

    def test_experiment_id_from_config(self):
        mgr = ExperimentManager(_cfg(experiment_id="my_exp"))
        exp = mgr.build()
        assert exp.experiment_id == "my_exp"

    def test_architecture_from_inference_result(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(inference_result=self._inf_result())
        assert exp.architecture == "unetplusplus"

    def test_architecture_from_eval_result_when_no_inference(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(evaluation_result=self._eval_result())
        assert exp.architecture == "unetplusplus"

    def test_checkpoint_epoch_from_inference(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(inference_result=self._inf_result())
        assert exp.checkpoint_epoch == 10

    def test_class_names_from_inference(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(inference_result=self._inf_result())
        assert exp.class_names == ("bg","water","sand","veg")

    def test_git_commit_from_config(self):
        mgr = ExperimentManager(_cfg(git_commit="abc123"))
        exp = mgr.build()
        assert exp.git_commit == "abc123"

    def test_total_duration_propagated(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(total_duration_s=99.5)
        assert exp.total_duration_s == pytest.approx(99.5)

    def test_project_info_from_config(self):
        mgr = ExperimentManager(_cfg(project_name="KosiRiver", author="Bob"))
        exp = mgr.build()
        assert exp.project_name == "KosiRiver"
        assert exp.author       == "Bob"

    def test_config_snapshot_non_empty(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build()
        assert isinstance(exp.config_snapshot, dict)
        assert len(exp.config_snapshot) > 0

    def test_config_snapshot_includes_inference_config(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(inference_result=self._inf_result())
        assert "inference" in exp.config_snapshot

    def test_config_snapshot_includes_evaluation(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build(evaluation_result=self._eval_result())
        assert "evaluation" in exp.config_snapshot

    def test_no_inference_result_defaults(self):
        mgr = ExperimentManager(_cfg())
        exp = mgr.build()
        assert exp.architecture     == ""
        assert exp.checkpoint_epoch == 0


class TestExperimentHelpers:
    def test_generate_id_starts_with_exp(self):
        assert _generate_id().startswith("exp_")

    def test_generate_id_unique(self):
        assert _generate_id() != _generate_id()

    def test_read_git_commit_returns_string(self):
        result = _read_git_commit()
        assert isinstance(result, str)   # "" or actual commit hash


# ==============================================================================
# ManifestManager
# ==============================================================================

class TestManifestManager:
    def _artifacts(self):
        return (
            ArtifactRecord("a1", "markdown", "/out/r.md",   512, "Main report", "Module 19"),
            ArtifactRecord("a2", "json",     "/out/r.json", 1024, "JSON metrics", "Module 19"),
        )

    def test_build_returns_report_manifest(self):
        from src.reporting.contracts import ReportManifest
        mgr  = ManifestManager(_cfg())
        man  = mgr.build("exp_001", self._artifacts(), ("/out/r.md",))
        assert isinstance(man, ReportManifest)

    def test_manifest_artifact_count(self):
        mgr = ManifestManager(_cfg())
        man = mgr.build("exp_001", self._artifacts(), ("/out/r.md",))
        assert man.num_artifacts == 2

    def test_manifest_total_size_bytes(self):
        mgr = ManifestManager(_cfg())
        man = mgr.build("exp_001", self._artifacts(), ("/out/r.md",))
        assert man.total_size_bytes == 1536

    def test_manifest_experiment_id(self):
        mgr = ManifestManager(_cfg())
        man = mgr.build("my_exp", self._artifacts(), ())
        assert man.experiment_id == "my_exp"

    def test_manifest_report_files(self):
        mgr = ManifestManager(_cfg())
        man = mgr.build("exp", self._artifacts(), ("/a.md", "/b.json"))
        assert "/a.md" in man.report_files

    def test_write_creates_file(self, tmp_path):
        mgr = ManifestManager(_cfg())
        man = mgr.build("exp_001", self._artifacts(), ())
        path = mgr.write(man, str(tmp_path))
        assert Path(path).exists()

    def test_write_creates_valid_json(self, tmp_path):
        mgr  = ManifestManager(_cfg())
        man  = mgr.build("exp_001", self._artifacts(), ())
        path = mgr.write(man, str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert data["experiment_id"] == "exp_001"
        assert data["num_artifacts"] == 2

    def test_write_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "reports"
        mgr     = ManifestManager(_cfg())
        man     = mgr.build("exp", self._artifacts(), ())
        mgr.write(man, str(new_dir))
        assert new_dir.exists()