"""
Figure export for Module 18.

FigureExporter writes matplotlib Figure objects to disk in the formats
configured via VisualizationConfig (PNG, SVG, PDF). DPI is configurable.

Design rules
------------
- Never imports matplotlib at module level.
- Gracefully skips export when matplotlib is unavailable or when output_dir
  is empty (returns an empty list for that figure).
- Does not duplicate any export logic from Module 16 (PredictionExporter):
  Module 16 exports numpy arrays and GeoTIFFs; this module exports figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.visualization.contracts import FigureSpec, VisualizationConfig

__all__ = ["FigureExporter"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class FigureExporter:
    """
    Exports matplotlib Figure objects to PNG, SVG, and/or PDF.

    Args:
        config: VisualizationConfig with output_dir and format flags.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self._config  = config
        self._out_dir = Path(config.output_dir).resolve() if config.output_dir else None

    def export(self, spec: FigureSpec) -> list[str]:
        """
        Export one FigureSpec to all configured formats.

        Args:
            spec: FigureSpec with a non-None figure attribute.

        Returns:
            List of absolute path strings for all written files.
        """
        if self._out_dir is None or spec.figure is None:
            return []

        self._out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        fid   = _sanitise(spec.figure_id)

        if self._config.export_png:
            p = self._save(spec.figure, fid, "png")
            if p:
                paths.append(p)

        if self._config.export_svg:
            p = self._save(spec.figure, fid, "svg")
            if p:
                paths.append(p)

        if self._config.export_pdf:
            p = self._save(spec.figure, fid, "pdf")
            if p:
                paths.append(p)

        return paths

    def export_all(self, specs: list[FigureSpec]) -> list[FigureSpec]:
        """
        Export all FigureSpec objects and append paths to their export_paths list.

        Args:
            specs: List of FigureSpec objects.

        Returns:
            The same list (mutated in place) for chaining.
        """
        for spec in specs:
            paths = self.export(spec)
            spec.export_paths.extend(paths)
        return specs

    def _save(self, figure: Any, fid: str, fmt: str) -> str | None:
        """Save one figure in the given format. Returns path or None on failure."""
        path = self._out_dir / f"{fid}.{fmt}"  # type: ignore[operator]
        try:
            kw: dict = {"format": fmt, "bbox_inches": "tight"}
            if fmt == "png":
                kw["dpi"] = self._config.dpi
            figure.savefig(str(path), **kw)
            _LOGGER.debug("FigureExporter: saved %s -> %s", fmt.upper(), path.name)
            return str(path)
        except Exception as exc:
            _LOGGER.warning("FigureExporter: failed to save %s: %s", fmt, exc)
            return None


def _sanitise(fid: str) -> str:
    """Replace filesystem-unsafe characters in figure IDs."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in fid) or "figure"