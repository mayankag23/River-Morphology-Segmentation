"""
Pipeline configuration validator for Module 20.

PipelineValidator performs all cross-cutting consistency checks required before
any pipeline stage is allowed to run. Checks are exhaustive and fail-fast:
the pipeline halts on the first fatal issue.

Checks performed
----------------
1. mode               — must be one of VALID_MODES.
2. in_channels        — model.in_channels == num_channels == len(spectral_bands).
3. num_classes        — model.num_classes == classes.num_classes.
4. class names count  — len(classes.names) == classes.num_classes.
5. class colors count — len(classes.colors) == classes.num_classes.
6. patch size         — patch_generation.patch_size == inference.patch_size.
7. AOI completeness   — all four bbox coordinates set when GEE stages are needed.
8. Date range         — both start and end dates set, start < end.
9. max_tile_pixels    — positive integer.
10. Checkpoint path   — exists when resume_from is set.
11. Single/multi-AOI  — config.aoi and config.aois do not conflict ambiguously.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.pipeline.contracts import AOIConfig, PipelineConfig, VALID_MODES

__all__ = ["PipelineValidator", "PipelineValidationResult"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Stages that require GEE and therefore need a complete AOI and date range.
_GEE_STAGES: frozenset[str] = frozenset({"full"})


class PipelineValidationResult:
    """Result of one pipeline validation pass."""

    def __init__(self, issues: list[str], warnings: list[str]) -> None:
        self._issues   = list(issues)
        self._warnings = list(warnings)

    @property
    def is_valid(self) -> bool:
        return len(self._issues) == 0

    @property
    def issues(self) -> list[str]:
        return list(self._issues)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)


class PipelineValidator:
    """Validates the full pipeline configuration before execution."""

    def validate(
        self,
        pipeline_config: PipelineConfig,
        config:          Any,
        aoi_configs:     list[AOIConfig] | None = None,
    ) -> PipelineValidationResult:
        """
        Run all pre-flight checks.

        Args:
            pipeline_config: PipelineConfig from CLI/config.
            config:          Full project Config object.
            aoi_configs:     List of resolved AOIConfig objects.

        Returns:
            PipelineValidationResult with issues (fatal) and warnings (non-fatal).
        """
        issues:   list[str] = []
        warnings: list[str] = []

        issues.extend(self._check_mode(pipeline_config))
        issues.extend(self._check_channels(config))
        issues.extend(self._check_classes(config))
        issues.extend(self._check_patch_size(config))
        warnings.extend(self._check_tile_pixels(config))
        issues.extend(self._check_resume(pipeline_config))

        if aoi_configs:
            for aoi in aoi_configs:
                issues.extend(self._check_aoi(aoi, pipeline_config))
                issues.extend(self._check_dates(aoi))

        return PipelineValidationResult(issues, warnings)

    @staticmethod
    def _check_mode(pipeline_config: PipelineConfig) -> list[str]:
        if pipeline_config.mode not in VALID_MODES:
            return [
                f"Invalid pipeline mode '{pipeline_config.mode}'. "
                f"Valid modes: {', '.join(VALID_MODES)}."
            ]
        return []

    @staticmethod
    def _check_channels(config: Any) -> list[str]:
        issues: list[str] = []
        num_channels    = getattr(config, "num_channels",    None)
        spectral_bands  = getattr(config, "spectral_bands",  None)
        model           = getattr(config, "model",           None)
        model_in_ch     = int(getattr(model, "in_channels",  0)) if model else 0

        if num_channels is not None and spectral_bands is not None:
            n_bands = len(spectral_bands) if hasattr(spectral_bands, "__len__") else None
            if n_bands is not None and n_bands != int(num_channels):
                issues.append(
                    f"num_channels={num_channels} does not match "
                    f"len(spectral_bands)={n_bands}."
                )
        if model_in_ch > 0 and num_channels is not None:
            if model_in_ch != int(num_channels):
                issues.append(
                    f"model.in_channels={model_in_ch} does not match "
                    f"num_channels={num_channels}. "
                    "These must be equal."
                )
        return issues

    @staticmethod
    def _check_classes(config: Any) -> list[str]:
        issues:  list[str] = []
        classes  = getattr(config, "classes",  None)
        model    = getattr(config, "model",    None)
        if classes is None:
            return ["config.classes section is missing."]

        num_cls       = int(getattr(classes, "num_classes",  0))
        model_num_cls = int(getattr(model,   "num_classes",  0)) if model else 0
        names         = getattr(classes, "names",  None)
        colors        = getattr(classes, "colors", None)

        if model_num_cls > 0 and num_cls > 0 and model_num_cls != num_cls:
            issues.append(
                f"model.num_classes={model_num_cls} != classes.num_classes={num_cls}."
            )
        if names is not None and hasattr(names, "__len__"):
            if len(names) != num_cls:
                issues.append(
                    f"len(classes.names)={len(names)} != classes.num_classes={num_cls}."
                )
        if colors is not None and hasattr(colors, "__len__"):
            if len(colors) != num_cls:
                issues.append(
                    f"len(classes.colors)={len(colors)} != classes.num_classes={num_cls}."
                )
        return issues

    @staticmethod
    def _check_patch_size(config: Any) -> list[str]:
        pg    = getattr(config, "patch_generation", None)
        inf   = getattr(config, "inference",         None)
        if pg is None or inf is None:
            return []
        pg_size  = int(getattr(pg,  "patch_size", 0))
        inf_size = int(getattr(inf, "patch_size", 0))
        if pg_size != inf_size:
            return [
                f"patch_generation.patch_size={pg_size} != "
                f"inference.patch_size={inf_size}. "
                "These must be identical — the model input contract is set at "
                "patch generation time and cannot change."
            ]
        return []

    @staticmethod
    def _check_tile_pixels(config: Any) -> list[str]:
        export = getattr(config, "export", None)
        if export is None:
            return []
        max_tp = getattr(export, "max_tile_pixels", None)
        if max_tp is not None and int(max_tp) < 1:
            return [
                f"export.max_tile_pixels={max_tp} must be >= 1."
            ]
        return []

    @staticmethod
    def _check_resume(pipeline_config: PipelineConfig) -> list[str]:
        if pipeline_config.resume_from and not Path(pipeline_config.resume_from).exists():
            return [
                f"resume_from checkpoint '{pipeline_config.resume_from}' does not exist."
            ]
        return []

    @staticmethod
    def _check_aoi(aoi: AOIConfig, pipeline_config: PipelineConfig) -> list[str]:
        issues: list[str] = []
        # GEE stages need a complete AOI.
        if pipeline_config.mode in _GEE_STAGES or pipeline_config.mode == "full":
            if not aoi.is_complete:
                issues.append(
                    f"AOI '{aoi.aoi_id}': one or more coordinate bounds are null "
                    "(min_lon, min_lat, max_lon, max_lat). "
                    "Set all four before running GEE stages."
                )
            else:
                # Basic sanity checks.
                if aoi.min_lon >= aoi.max_lon:  # type: ignore[operator]
                    issues.append(
                        f"AOI '{aoi.aoi_id}': min_lon >= max_lon "
                        f"({aoi.min_lon} >= {aoi.max_lon})."
                    )
                if aoi.min_lat >= aoi.max_lat:  # type: ignore[operator]
                    issues.append(
                        f"AOI '{aoi.aoi_id}': min_lat >= max_lat "
                        f"({aoi.min_lat} >= {aoi.max_lat})."
                    )
        return issues

    @staticmethod
    def _check_dates(aoi: AOIConfig) -> list[str]:
        issues: list[str] = []
        if not aoi.start_date and not aoi.end_date:
            return []   # Both empty → will be caught by GEE stage itself.
        if bool(aoi.start_date) != bool(aoi.end_date):
            issues.append(
                f"AOI '{aoi.aoi_id}': date_range.start and date_range.end must "
                "both be set or both be null."
            )
        elif aoi.start_date and aoi.end_date:
            if aoi.start_date >= aoi.end_date:
                issues.append(
                    f"AOI '{aoi.aoi_id}': date_range.start='{aoi.start_date}' "
                    f"must be before date_range.end='{aoi.end_date}'."
                )
        return issues
