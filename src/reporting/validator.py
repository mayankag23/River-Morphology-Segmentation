"""
Report validation for Module 19.

ReportValidator runs pre-flight checks before report generation. It verifies
that the ReportingConfig is consistent and that at least one upstream result
is available. Never raises; accumulates issues for the caller to decide.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.reporting.contracts import ReportingConfig

__all__ = ["ReportValidator", "ReportValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ReportValidationResult:
    """Result of one validation pass."""

    def __init__(self, issues: list[str]) -> None:
        self._issues = list(issues)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)


class ReportValidator:
    """Pre-flight validation for the report engine."""

    def validate(
        self,
        config:               ReportingConfig,
        evaluation_result:    Any | None = None,
        inference_result:     Any | None = None,
        morphology_result:    Any | None = None,
        visualization_result: Any | None = None,
    ) -> ReportValidationResult:
        """
        Validate ReportingConfig and upstream result availability.

        Args:
            config:               ReportingConfig.
            evaluation_result:    EvaluationResult from Module 15 (optional).
            inference_result:     InferenceResult from Module 16 (optional).
            morphology_result:    RiverMorphologyResult from Module 17 (optional).
            visualization_result: VisualizationResult from Module 18 (optional).

        Returns:
            ReportValidationResult with any detected issues.
        """
        issues: list[str] = []

        # At least one result must be provided.
        if all(r is None for r in (
            evaluation_result, inference_result,
            morphology_result, visualization_result,
        )):
            issues.append(
                "ReportEngine: no upstream results provided; nothing to report."
            )

        # Output directory must be non-empty.
        if not config.output_dir:
            issues.append("output_dir is empty; reports cannot be written.")

        # Output path must be a valid filesystem path.
        if config.output_dir:
            try:
                Path(config.output_dir)
            except Exception:
                issues.append(
                    f"output_dir='{config.output_dir}' is an invalid path."
                )

        # At least one export format must be enabled when output is configured.
        if config.output_dir and not any([
            config.export_markdown,
            config.export_json,
            config.export_csv,
            config.export_pdf,
        ]):
            issues.append(
                "No export format is enabled "
                "(export_markdown, export_json, export_csv, export_pdf are all False)."
            )

        # report_name must not be empty.
        if not config.report_name:
            issues.append("report_name must not be empty.")

        return ReportValidationResult(issues)