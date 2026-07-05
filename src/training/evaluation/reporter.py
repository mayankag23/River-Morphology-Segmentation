"""
Structured evaluation report generation for Module 15.

EvaluationReporter writes EvaluationResult to JSON and/or CSV files.
No matplotlib, no plots — visualization belongs to a future module.

JSON output: full EvaluationResult.as_dict() (all metrics, confusion matrix,
             per-class details, statistics).
CSV output:  one row per class with all ClassMetrics fields.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.training.evaluation.contracts import EvaluationResult

__all__ = ["EvaluationReporter"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    Writes EvaluationResult to JSON and/or CSV files.

    Args:
        output_dir: Directory to write reports. Created if absent.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir).resolve()

    def save_json(self, result: EvaluationResult, filename: str = "evaluation.json") -> Path:
        """
        Write the full EvaluationResult as a JSON file.

        Args:
            result:   EvaluationResult to serialise.
            filename: Output filename (relative to output_dir).

        Returns:
            Absolute path of the written file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / filename
        data = result.as_dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=True)
        _LOGGER.info("EvaluationReporter: JSON saved -> %s", path)
        return path

    def save_csv(self, result: EvaluationResult, filename: str = "per_class_metrics.csv") -> Path:
        """
        Write per-class metrics as a CSV file (one row per class).

        Columns: class_id, class_name, precision, recall, f1, dice, iou,
                 pixel_accuracy, tp, fp, fn, tn, num_pixels, num_predicted, support.

        Args:
            result:   EvaluationResult.
            filename: Output filename.

        Returns:
            Absolute path of the written file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / filename

        fieldnames = [
            "class_id", "class_name", "precision", "recall", "f1", "dice",
            "iou", "pixel_accuracy", "tp", "fp", "fn", "tn",
            "num_pixels", "num_predicted", "support",
        ]

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for name in result.class_names:
                cm = result.per_class.get(name)
                if cm is None:
                    continue
                writer.writerow({
                    "class_id":       cm.class_id,
                    "class_name":     cm.class_name,
                    "precision":      round(cm.precision,      6),
                    "recall":         round(cm.recall,         6),
                    "f1":             round(cm.f1,             6),
                    "dice":           round(cm.dice,           6),
                    "iou":            round(cm.iou,            6),
                    "pixel_accuracy": round(cm.pixel_accuracy, 6),
                    "tp":             cm.tp,
                    "fp":             cm.fp,
                    "fn":             cm.fn,
                    "tn":             cm.tn,
                    "num_pixels":     cm.num_pixels,
                    "num_predicted":  cm.num_predicted,
                    "support":        cm.support,
                })

        _LOGGER.info("EvaluationReporter: CSV saved -> %s", path)
        return path

    def save_all(self, result: EvaluationResult) -> dict[str, Path]:
        """
        Write all configured report formats.

        Returns:
            Dict mapping format name to written path.
        """
        paths: dict[str, Path] = {}
        split = result.split
        paths["json"] = self.save_json(result, f"evaluation_{split}.json")
        paths["csv"]  = self.save_csv(result,  f"per_class_{split}.csv")
        return paths
