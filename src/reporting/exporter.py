"""
Report file export for Module 19.

ReportExporter writes report content to disk in the formats configured via
ReportingConfig. No duplication of Module 16 (PredictionExporter) or Module 18
(FigureExporter) — those export arrays and figures; this exports textual reports.

Supported formats:
    Markdown (.md)   — Human-readable summary with tables.
    JSON (.json)     — Machine-readable metrics summary.
    CSV (.csv)       — Per-class metrics table, importable by Excel / pandas.
    PDF (.pdf)       — Future-ready stub; requires weasyprint or reportlab;
                       gracefully skipped when neither is available.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.reporting.contracts import ReportingConfig

__all__ = ["ReportExporter"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ReportExporter:
    """
    Writes report content to disk in configured formats.

    Args:
        config: ReportingConfig with output_dir and format flags.
    """

    def __init__(self, config: ReportingConfig) -> None:
        self._config  = config
        self._out_dir = Path(config.output_dir).resolve()

    def export_markdown(self, content: str, filename: str | None = None) -> str | None:
        """
        Write a Markdown report to disk.

        Args:
            content:  Markdown string.
            filename: Output filename. Defaults to {report_name}.md.

        Returns:
            Absolute path of the written file, or None on failure.
        """
        if not self._config.export_markdown:
            return None
        name = filename or f"{self._config.report_name}.md"
        return self._write_text(content, name)

    def export_json(self, data: dict, filename: str | None = None) -> str | None:
        """
        Write a JSON metrics summary.

        Args:
            data:     JSON-serializable dict.
            filename: Output filename. Defaults to {report_name}.json.

        Returns:
            Absolute path, or None on failure.
        """
        if not self._config.export_json:
            return None
        name    = filename or f"{self._config.report_name}.json"
        content = json.dumps(data, indent=2, ensure_ascii=True)
        return self._write_text(content, name)

    def export_csv(self, rows: list[dict], filename: str | None = None) -> str | None:
        """
        Write per-class metrics as a CSV file.

        Args:
            rows:     List of dicts from ReportGenerator.build_csv_rows().
            filename: Output filename. Defaults to {report_name}_per_class.csv.

        Returns:
            Absolute path, or None on failure.
        """
        if not self._config.export_csv or not rows:
            return None
        name = filename or f"{self._config.report_name}_per_class.csv"
        path = self._prepare_path(name)
        if path is None:
            return None
        try:
            fieldnames = list(rows[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            _LOGGER.info("ReportExporter: CSV saved -> %s", path.name)
            return str(path)
        except Exception as exc:
            _LOGGER.warning("ReportExporter: CSV export failed: %s", exc)
            return None

    def export_pdf(self, markdown_content: str, filename: str | None = None) -> str | None:
        """
        Write a PDF report. Requires weasyprint; falls back to None when unavailable.

        Args:
            markdown_content: Markdown string (converted to HTML, then PDF).
            filename:         Output filename. Defaults to {report_name}.pdf.

        Returns:
            Absolute path, or None when PDF export is disabled or unavailable.
        """
        if not self._config.export_pdf:
            return None
        name = filename or f"{self._config.report_name}.pdf"
        path = self._prepare_path(name)
        if path is None:
            return None
        try:
            import weasyprint
            html = _markdown_to_html(markdown_content, self._config.project_name)
            weasyprint.HTML(string=html).write_pdf(str(path))
            _LOGGER.info("ReportExporter: PDF saved -> %s", path.name)
            return str(path)
        except ImportError:
            _LOGGER.warning(
                "ReportExporter: weasyprint not installed; PDF export skipped."
            )
            return None
        except Exception as exc:
            _LOGGER.warning("ReportExporter: PDF export failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_text(self, content: str, filename: str) -> str | None:
        """Write a text file. Returns path or None on failure."""
        path = self._prepare_path(filename)
        if path is None:
            return None
        try:
            path.write_text(content, encoding="utf-8")
            _LOGGER.info("ReportExporter: saved -> %s", path.name)
            return str(path)
        except Exception as exc:
            _LOGGER.warning("ReportExporter: write failed for %s: %s", filename, exc)
            return None

    def _prepare_path(self, filename: str) -> Path | None:
        """Create the output directory and return the full path."""
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            return self._out_dir / filename
        except Exception as exc:
            _LOGGER.warning("ReportExporter: cannot create output dir: %s", exc)
            return None


def _markdown_to_html(md: str, title: str) -> str:
    """Convert Markdown to a minimal HTML string for PDF generation."""
    # Minimal conversion: wrap in <pre> when markdown is not available.
    try:
        import markdown
        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        body = f"<pre>{md}</pre>"
    return (
        f"<!DOCTYPE html><html><head>"
        f"<meta charset='utf-8'><title>{title}</title>"
        f"<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:2em}}"
        f"pre{{background:#f4f4f4;padding:1em;overflow:auto}}</style>"
        f"</head><body>{body}</body></html>"
    )