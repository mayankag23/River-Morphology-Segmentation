"""
Report content generation for Module 19.

ReportGenerator reads pre-computed results from Modules 15-18 and assembles
structured report content (Markdown text, JSON dicts, CSV rows). It never
recomputes any metric — all numbers come from the immutable upstream contracts.

Report sections
---------------
1. Header          — project, author, timestamp, model info.
2. Evaluation      — EvaluationResult metrics table.
3. Inference       — InferenceResult summary.
4. Morphology      — RiverMorphologyResult temporal and class summaries.
5. Visualization   — VisualizationResult figure inventory.
6. Configuration   — config snapshot (when include_config=True).
"""

from __future__ import annotations

import logging
from typing import Any

from src.reporting.contracts import ExperimentMetadata, ReportingConfig

__all__ = ["ReportGenerator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Assembles report text and data from upstream result objects.

    Args:
        config:     ReportingConfig.
        experiment: ExperimentMetadata (assembled by ExperimentManager).
    """

    def __init__(
        self,
        config:     ReportingConfig,
        experiment: ExperimentMetadata,
    ) -> None:
        self._config     = config
        self._experiment = experiment

    # ------------------------------------------------------------------
    # Public: content assembly
    # ------------------------------------------------------------------

    def build_markdown(
        self,
        evaluation_result:  Any | None = None,
        inference_result:   Any | None = None,
        morphology_result:  Any | None = None,
        visualization_result: Any | None = None,
    ) -> str:
        """
        Build the full Markdown report as a single string.

        Args:
            evaluation_result:   EvaluationResult from Module 15.
            inference_result:    InferenceResult from Module 16.
            morphology_result:   RiverMorphologyResult from Module 17.
            visualization_result: VisualizationResult from Module 18.

        Returns:
            Complete Markdown string.
        """
        sections: list[str] = []
        sections.append(self._md_header())

        if self._config.include_evaluation and evaluation_result is not None:
            sections.append(self._md_evaluation(evaluation_result))

        if self._config.include_inference and inference_result is not None:
            sections.append(self._md_inference(inference_result))

        if self._config.include_morphology and morphology_result is not None:
            sections.append(self._md_morphology(morphology_result))

        if self._config.include_visualization and visualization_result is not None:
            sections.append(self._md_visualization(visualization_result))

        if self._config.include_config:
            sections.append(self._md_config())

        sections.append(self._md_footer())
        return "\n\n".join(sections)

    def build_json_summary(
        self,
        evaluation_result:   Any | None = None,
        inference_result:    Any | None = None,
        morphology_result:   Any | None = None,
        visualization_result: Any | None = None,
    ) -> dict:
        """
        Build a JSON-serializable summary dict.

        Returns:
            Dict suitable for json.dump().
        """
        summary: dict = {
            "experiment":  self._experiment.as_dict(),
            "report_version": self._config.report_version,
        }
        if self._config.include_evaluation and evaluation_result is not None:
            summary["evaluation"] = _extract_evaluation_summary(evaluation_result)
        if self._config.include_inference and inference_result is not None:
            summary["inference"] = _extract_inference_summary(inference_result)
        if self._config.include_morphology and morphology_result is not None:
            summary["morphology"] = _extract_morphology_summary(morphology_result)
        if self._config.include_visualization and visualization_result is not None:
            summary["visualization"] = _extract_visualization_summary(visualization_result)
        return summary

    def build_csv_rows(
        self,
        evaluation_result: Any | None = None,
        morphology_result: Any | None = None,
    ) -> list[dict]:
        """
        Build a list of per-class metric row dicts for CSV export.

        Returns:
            List of dicts with keys: class_name, precision, recall, f1, iou,
            dice, pixel_accuracy, area_fraction, mean_confidence.
        """
        rows: list[dict] = []
        class_names: tuple[str, ...] = ()

        if evaluation_result is not None:
            class_names = tuple(getattr(evaluation_result, "class_names", ()))
            per_class   = getattr(evaluation_result, "per_class", {})
            for cls_name in class_names:
                cm = per_class.get(cls_name)
                row: dict = {"class_name": cls_name}
                if cm is not None:
                    row["precision"]      = round(getattr(cm, "precision",      0.0), 6)
                    row["recall"]         = round(getattr(cm, "recall",         0.0), 6)
                    row["f1"]             = round(getattr(cm, "f1",             0.0), 6)
                    row["iou"]            = round(getattr(cm, "iou",            0.0), 6)
                    row["dice"]           = round(getattr(cm, "dice",           0.0), 6)
                    row["pixel_accuracy"] = round(getattr(cm, "pixel_accuracy", 0.0), 6)
                else:
                    for k in ("precision", "recall", "f1", "iou", "dice", "pixel_accuracy"):
                        row[k] = 0.0

                # Merge morphology area fraction if available.
                if morphology_result is not None:
                    for sa in getattr(morphology_result, "sample_analyses", ()):
                        cm_m = sa.class_metrics.get(cls_name)
                        if cm_m is not None:
                            row["area_fraction"]   = round(cm_m.area_fraction,  6)
                            row["mean_confidence"] = round(cm_m.mean_confidence, 6)
                            break
                rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Private: Markdown section builders
    # ------------------------------------------------------------------

    def _md_header(self) -> str:
        exp = self._experiment
        return (
            f"# {self._config.project_name}\n\n"
            f"**Version:** {self._config.project_version}  \n"
            f"**Report Version:** {self._config.report_version}  \n"
            f"**Experiment ID:** {exp.experiment_id}  \n"
            f"**Timestamp:** {exp.run_timestamp}  \n"
            f"**Architecture:** {exp.architecture}  \n"
            f"**Checkpoint Epoch:** {exp.checkpoint_epoch}  \n"
            f"**Classes:** {', '.join(exp.class_names)}  \n"
            + (f"**Author:** {exp.author}  \n" if exp.author else "")
            + (f"**Institution:** {exp.institution}  \n" if exp.institution else "")
            + (f"**Git Commit:** `{exp.git_commit}`  \n" if exp.git_commit else "")
        )

    def _md_evaluation(self, result: Any) -> str:
        s = _extract_evaluation_summary(result)
        lines = ["## Evaluation Metrics\n"]
        lines.append(f"- **Split:** {s.get('split', '')}")
        lines.append(f"- **Pixel Accuracy:** {s.get('pixel_accuracy', 0):.4f}")
        lines.append(f"- **Mean IoU:** {s.get('mean_iou', 0):.4f}")
        lines.append(f"- **Mean Dice:** {s.get('mean_dice', 0):.4f}")
        lines.append(f"- **Mean F1:** {s.get('mean_f1', 0):.4f}")
        lines.append(f"- **Cohen Kappa:** {s.get('kappa', 0):.4f}")
        lines.append(f"- **Balanced Accuracy:** {s.get('balanced_accuracy', 0):.4f}")
        lines.append(f"- **Total Samples:** {s.get('total_samples', 0)}")
        lines.append(f"- **Evaluation Time:** {s.get('evaluation_time_s', 0):.2f}s")
        return "\n".join(lines)

    def _md_inference(self, result: Any) -> str:
        s = _extract_inference_summary(result)
        lines = ["## Inference Summary\n"]
        lines.append(f"- **Architecture:** {s.get('architecture', '')}")
        lines.append(f"- **Device:** {s.get('device_used', '')}")
        lines.append(f"- **Samples:** {s.get('num_samples', 0)}")
        lines.append(f"- **Mean Confidence:** {s.get('mean_confidence', 0):.4f}")
        lines.append(f"- **Checkpoint Epoch:** {s.get('checkpoint_epoch', 0)}")
        lines.append(f"- **Inference Time:** {s.get('total_inference_s', 0):.2f}s")
        return "\n".join(lines)

    def _md_morphology(self, result: Any) -> str:
        s = _extract_morphology_summary(result)
        lines = ["## Morphology Summary\n"]
        lines.append(f"- **Samples Analysed:** {s.get('num_samples', 0)}")
        lines.append(f"- **Mean Water Fraction:** {s.get('mean_water_fraction', 0):.4f}")
        lines.append(f"- **Mean Sand Fraction:** {s.get('mean_sand_fraction', 0):.4f}")
        lines.append(f"- **Mean Vegetation Fraction:** {s.get('mean_veg_fraction', 0):.4f}")
        lines.append(f"- **Mean Confidence:** {s.get('mean_confidence', 0):.4f}")
        lines.append(f"- **Temporal Changes:** {s.get('num_temporal_changes', 0)}")
        lines.append(f"- **Seasonal Summaries:** {s.get('num_seasonal_summaries', 0)}")
        return "\n".join(lines)

    def _md_visualization(self, result: Any) -> str:
        s = _extract_visualization_summary(result)
        lines = ["## Visualization Summary\n"]
        lines.append(f"- **Figures Generated:** {s.get('num_figures', 0)}")
        lines.append(f"- **Files Exported:** {s.get('num_exported', 0)}")
        lines.append(f"- **Samples Visualised:** {s.get('num_samples', 0)}")
        lines.append(f"- **Render Time:** {s.get('visualization_time_s', 0):.2f}s")
        return "\n".join(lines)

    def _md_config(self) -> str:
        snap = self._experiment.config_snapshot
        lines = ["## Configuration Snapshot\n", "```json"]
        import json
        lines.append(json.dumps(snap, indent=2))
        lines.append("```")
        return "\n".join(lines)

    def _md_footer(self) -> str:
        return (
            f"---\n\n"
            f"*Generated by {self._config.project_name} "
            f"v{self._config.project_version} — "
            f"Report v{self._config.report_version}*"
        )


# ==============================================================================
# Module-level summary extractors (reused by ReportGenerator and engine)
# ==============================================================================

def _extract_evaluation_summary(result: Any) -> dict:
    """Extract key scalar metrics from EvaluationResult."""
    return {
        "split":             getattr(result, "split",             ""),
        "architecture":      getattr(result, "architecture",      ""),
        "num_classes":       getattr(result, "num_classes",       0),
        "num_samples":       getattr(result, "total_samples",     0),
        "total_pixels":      getattr(result, "total_pixels",      0),
        "pixel_accuracy":    round(getattr(result, "pixel_accuracy",      0.0), 6),
        "mean_pixel_accuracy": round(getattr(result, "mean_pixel_accuracy", 0.0), 6),
        "mean_iou":          round(getattr(result, "mean_iou",            0.0), 6),
        "fw_iou":            round(getattr(result, "fw_iou",              0.0), 6),
        "mean_dice":         round(getattr(result, "mean_dice",           0.0), 6),
        "mean_precision":    round(getattr(result, "mean_precision",      0.0), 6),
        "mean_recall":       round(getattr(result, "mean_recall",         0.0), 6),
        "mean_f1":           round(getattr(result, "mean_f1",             0.0), 6),
        "kappa":             round(getattr(result, "kappa",               0.0), 6),
        "balanced_accuracy": round(getattr(result, "balanced_accuracy",   0.0), 6),
        "evaluation_time_s": round(getattr(result, "evaluation_time_s",   0.0), 3),
        "class_names":       list(getattr(result, "class_names", ())),
    }


def _extract_inference_summary(result: Any) -> dict:
    """Extract key scalars from InferenceResult."""
    ckpt = getattr(result, "checkpoint_meta", None)
    return {
        "architecture":     getattr(result, "architecture",    ""),
        "num_samples":      getattr(result, "num_samples",     0),
        "num_classes":      getattr(result, "num_classes",     0),
        "device_used":      getattr(result, "device_used",     ""),
        "mean_confidence":  round(getattr(result, "mean_confidence",  0.0), 6),
        "total_inference_s": round(getattr(result, "total_inference_s", 0.0), 3),
        "per_sample_ms":    round(getattr(result, "per_sample_ms",    0.0), 3),
        "class_names":      list(getattr(result, "class_names", ())),
        "class_pixel_counts": dict(getattr(result, "class_pixel_counts", {})),
        "checkpoint_epoch": int(getattr(ckpt, "epoch", 0)) if ckpt else 0,
        "checkpoint_path":  str(getattr(ckpt, "checkpoint_path", "")) if ckpt else "",
        "val_loss":         round(float(getattr(ckpt, "val_loss", 0.0)), 6) if ckpt else 0.0,
    }


def _extract_morphology_summary(result: Any) -> dict:
    """Extract key scalars from RiverMorphologyResult."""
    return {
        "num_samples":            getattr(result, "num_samples",         0),
        "num_classes":            getattr(result, "num_classes",         0),
        "total_pixels":           getattr(result, "total_pixels",        0),
        "mean_water_fraction":    round(getattr(result, "mean_water_fraction", 0.0), 6),
        "mean_sand_fraction":     round(getattr(result, "mean_sand_fraction",  0.0), 6),
        "mean_veg_fraction":      round(getattr(result, "mean_veg_fraction",   0.0), 6),
        "mean_confidence":        round(getattr(result, "mean_confidence",     0.0), 6),
        "num_temporal_changes":   len(getattr(result, "temporal_changes",  ())),
        "num_seasonal_summaries": len(getattr(result, "seasonal_summaries", {})),
        "num_spatial_summaries":  len(getattr(result, "spatial_summaries",  {})),
        "class_names":            list(getattr(result, "class_names", ())),
        "architecture":           getattr(result, "architecture", ""),
        "analysis_time_s":        round(getattr(result, "analysis_time_s", 0.0), 3),
    }


def _extract_visualization_summary(result: Any) -> dict:
    """Extract key scalars from VisualizationResult."""
    return {
        "num_figures":          getattr(result, "num_figures",          0),
        "num_exported":         getattr(result, "num_exported",         0),
        "num_samples":          getattr(result, "num_samples",          0),
        "architecture":         getattr(result, "architecture",         ""),
        "output_dir":           getattr(result, "output_dir",           ""),
        "class_names":          list(getattr(result, "class_names", ())),
        "visualization_time_s": round(getattr(result, "visualization_time_s", 0.0), 3),
    }