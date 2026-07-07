"""
Pipeline factory for Module 20.

PipelineFactory resolves the list of AOIConfigs from the project config,
builds a deterministic run_id, and wires up all the engine/factory
instantiation so the orchestrator only has to call `runner.run(stage, fn)`.

Design rule: PipelineFactory never executes any module logic. It only
instantiates and configures the engine objects that the orchestrator will call.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.contracts import AOIConfig, PipelineConfig

__all__ = ["PipelineFactory"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class PipelineFactory:
    """Assembles the execution context for one pipeline run."""

    @classmethod
    def resolve_aoi_configs(
        cls,
        pipeline_config: PipelineConfig,
        config:          Any,
    ) -> list[AOIConfig]:
        """
        Build the ordered list of AOIConfig objects for this run.

        Resolution order:
        1. config.aois list (multi-AOI batch mode).
        2. pipeline_config.aoi_ids filtered against config.aoi (single-AOI mode).
        3. Bare config.aoi section (single AOI, no explicit ID).

        Returns:
            Ordered list of AOIConfig; deterministic (sorted by aoi_id).
        """
        aoi_configs: list[AOIConfig] = []

        # Multi-AOI: config.aois is a list of AOI dicts.
        raw_aois = getattr(config, "aois", None)
        if raw_aois and hasattr(raw_aois, "__iter__"):
            for entry in raw_aois:
                aoi_id = str(getattr(entry, "id", getattr(entry, "aoi_id", "unknown")))
                # Filter by requested aoi_ids if specified.
                if pipeline_config.aoi_ids and aoi_id not in pipeline_config.aoi_ids:
                    continue
                aoi_configs.append(cls._build_aoi_config(aoi_id, entry, config))
            if aoi_configs:
                return sorted(aoi_configs, key=lambda a: a.aoi_id)

        # Single-AOI: fall back to config.aoi.
        aoi_section = getattr(config, "aoi", None)
        if aoi_section is not None:
            aoi_id = str(getattr(aoi_section, "id", "default"))
            aoi_configs.append(cls._build_aoi_config(aoi_id, aoi_section, config))

        return aoi_configs

    @staticmethod
    def _build_aoi_config(
        aoi_id:      str,
        aoi_section: Any,
        config:      Any,
    ) -> AOIConfig:
        """Build one AOIConfig from an AOI config section."""
        date_range  = getattr(config, "date_range", None)
        start_date  = str(getattr(aoi_section, "start", None) or
                          getattr(date_range,  "start", None) or "")
        end_date    = str(getattr(aoi_section, "end",   None) or
                          getattr(date_range,  "end",   None) or "")

        return AOIConfig(
            aoi_id     = aoi_id,
            min_lon    = getattr(aoi_section, "min_lon", None),
            min_lat    = getattr(aoi_section, "min_lat", None),
            max_lon    = getattr(aoi_section, "max_lon", None),
            max_lat    = getattr(aoi_section, "max_lat", None),
            start_date = start_date,
            end_date   = end_date,
        )

    @classmethod
    def make_run_id(cls, pipeline_config: PipelineConfig) -> str:
        """Return a deterministic, collision-safe run identifier."""
        if pipeline_config.run_id:
            return pipeline_config.run_id
        ts    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        short = uuid.uuid4().hex[:6]
        return f"{pipeline_config.mode}_{ts}_{short}"

    @classmethod
    def make_output_dir(
        cls,
        base_output_dir: str,
        run_id:          str,
        aoi_id:          str,
    ) -> str:
        """
        Build a per-AOI output directory path.

        Pattern: {base_output_dir}/{run_id}/{aoi_id}/
        Collision-safe because run_id is unique and aoi_id is explicit.
        """
        path = Path(base_output_dir) / run_id / aoi_id
        return str(path)

    @classmethod
    def resolve_device(cls, config: Any, cli_override: str = "") -> str:
        """Resolve the compute device string."""
        if cli_override:
            return cli_override
        device_section = getattr(config, "device", None)
        raw = str(getattr(device_section, "device", "auto") if device_section else "auto")
        if raw.lower() == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return raw
