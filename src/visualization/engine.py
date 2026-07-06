"""
VisualizationEngine — the single public interface for Module 18.

Usage
-----
    from src.visualization import VisualizationEngine, VisualizationConfig

    config = VisualizationConfig(
        output_dir   = "figures",
        export_png   = True,
        dpi          = 150,
        alpha_overlay = 0.5,
        class_colors = {
            "background": (0.2, 0.2, 0.2),
            "water":      (0.1, 0.5, 0.8),
            "sand":       (0.9, 0.8, 0.5),
            "vegetation": (0.2, 0.6, 0.2),
        },
    )
    engine = VisualizationEngine(config)
    result = engine.visualize(morphology_result)

    print(result.num_figures)
    for spec in result.figures:
        print(spec.figure_id, spec.export_paths)

VisualizationEngine orchestrates:
    VisualizationValidator    -> pre-flight checks
    VisualizationFactory      -> builds all renderers
    MaskRenderer              -> mask, per-class, confidence, uncertainty figures
    OverlayRenderer           -> class overlay, confidence overlay
    TimelineRenderer          -> temporal timeline, seasonal, change charts
    ComparisonRenderer        -> side-by-side, change map, class diff
    FigureExporter            -> PNG/SVG/PDF export
    VisualizationResult       -> immutable public output
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.visualization.contracts import (
    FigureSpec,
    VisualizationConfig,
    VisualizationResult,
)
from src.visualization.factory import VisualizationFactory
from src.visualization.validator import VisualizationValidator

__all__ = ["VisualizationEngine"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Orchestrates a complete visualization run.

    Args:
        config: VisualizationConfig or project Config object.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, VisualizationConfig):
            self._config = config
        else:
            self._config = VisualizationConfig.from_config(config)
        self._validator = VisualizationValidator()

    def visualize(
        self,
        morphology_result: Any,
        prediction_masks:  dict[str, np.ndarray] | None = None,
        confidence_maps:   dict[str, np.ndarray] | None = None,
        source_images:     dict[str, np.ndarray] | None = None,
    ) -> VisualizationResult:
        """
        Run a complete visualization pass.

        Args:
            morphology_result: RiverMorphologyResult from Module 17.
            prediction_masks:  Optional dict sample_id -> (H, W) uint8 mask.
                               When provided, enables per-sample figure rendering.
                               When None, only aggregate figures are generated.
            confidence_maps:   Optional dict sample_id -> (H, W) float32 confidence.
            source_images:     Optional dict sample_id -> (H, W[, C]) source image
                               for overlay rendering.

        Returns:
            Frozen VisualizationResult with all generated FigureSpec objects.
        """
        ops: list[str] = []
        t0             = time.perf_counter()

        # Step 1: Pre-flight validation.
        validation = self._validator.validate(self._config, morphology_result)
        for issue in validation.issues:
            _LOGGER.warning("VisualizationEngine pre-flight: %s", issue)
        ops.append(f"validation: {len(validation.issues)} issue(s)")

        # Step 2: Build components.
        context     = VisualizationFactory.build(self._config, morphology_result)
        class_names = context["class_names"]
        ops.append(f"classes: {len(class_names)}")

        mask_rend   = context["mask_renderer"]
        ovl_rend    = context["overlay_renderer"]
        tl_rend     = context["timeline_renderer"]
        cmp_rend    = context["comparison_renderer"]
        exporter    = context["exporter"]

        all_specs: list[FigureSpec] = []

        # Step 3: Per-sample figures (require prediction_masks).
        analyses = list(getattr(morphology_result, "sample_analyses", ()))
        max_s    = self._config.max_samples
        if max_s > 0:
            analyses = analyses[:max_s]

        n_rendered = 0
        if prediction_masks:
            for analysis in analyses:
                sid  = analysis.sample_id
                mask = prediction_masks.get(sid)
                if mask is None:
                    continue
                conf = (confidence_maps or {}).get(sid)
                img  = (source_images  or {}).get(sid)

                if self._config.render_masks:
                    all_specs.append(mask_rend.render_mask(analysis, mask))

                if self._config.render_per_class:
                    all_specs.append(
                        mask_rend.render_per_class(analysis, mask, class_names)
                    )

                if self._config.render_confidence and conf is not None:
                    all_specs.append(mask_rend.render_confidence(analysis, conf))
                    all_specs.append(
                        mask_rend.render_uncertainty(
                            analysis, conf,
                            morphology_result.analytics_config.low_confidence_threshold,
                        )
                    )

                if self._config.render_overlays:
                    all_specs.append(
                        ovl_rend.render_overlay(analysis, mask, img)
                    )
                    if conf is not None:
                        all_specs.append(
                            ovl_rend.render_confidence_overlay(
                                analysis, mask, conf,
                                morphology_result.analytics_config.low_confidence_threshold,
                            )
                        )
                n_rendered += 1
        ops.append(f"per-sample: {n_rendered} samples rendered")

        # Step 4: Temporal / seasonal aggregate figures.
        if self._config.render_timeline:
            all_specs.append(tl_rend.render_timeline(morphology_result))
            all_specs.append(tl_rend.render_change_chart(morphology_result))

        if self._config.render_seasonal:
            all_specs.append(tl_rend.render_seasonal(morphology_result))

        # Step 5: Comparison figures (first two samples).
        if (
            self._config.render_comparison
            and prediction_masks
            and len(analyses) >= 2
        ):
            a_a = analyses[0]
            a_b = analyses[1]
            m_a = prediction_masks.get(a_a.sample_id)
            m_b = prediction_masks.get(a_b.sample_id)
            if m_a is not None and m_b is not None:
                all_specs.append(cmp_rend.render_side_by_side(a_a, a_b, m_a, m_b))
                all_specs.append(cmp_rend.render_change_map(a_a, a_b, m_a, m_b))
                all_specs.append(cmp_rend.render_class_diff(a_a, a_b, class_names))

        ops.append(f"total figures: {len(all_specs)}")

        # Step 6: Export figures to disk.
        if self._config.output_dir:
            exporter.export_all(all_specs)
        num_exported = sum(len(s.export_paths) for s in all_specs)
        ops.append(f"exported: {num_exported} files")

        elapsed = time.perf_counter() - t0
        ops.append(f"total_time: {elapsed:.3f}s")

        # Step 7: Assemble VisualizationResult.
        result = VisualizationResult(
            figures              = tuple(all_specs),
            num_figures          = len(all_specs),
            num_exported         = num_exported,
            visualization_config = self._config,
            architecture         = str(getattr(morphology_result, "architecture", "")),
            num_samples          = len(analyses),
            class_names          = class_names,
            output_dir           = str(Path(self._config.output_dir).resolve())
                                   if self._config.output_dir else "",
            operations_log       = tuple(ops),
            visualization_time_s = elapsed,
        )

        for line in result.summary_lines():
            _LOGGER.info(line)

        return result