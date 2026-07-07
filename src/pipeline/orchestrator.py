# """
# Pipeline orchestrator for Module 20.

# PipelineOrchestrator is the central coordinator that:
# 1. Resolves the stage list for the requested mode.
# 2. Iterates over all AOIs (single or batch).
# 3. Calls each stage via StageRunner.
# 4. Collects StageResults into the final PipelineResult.

# Module calling order (Modules 2-19):
#     full:           2→3→4→5→6→7→8→9→10→11→12→13→14→15→16→17→18→19
#     training:       11→12→13→14
#     evaluation:     11→12→13→15
#     inference:      16
#     analysis:       17
#     visualization:  18
#     reporting:      19

# The orchestrator NEVER duplicates logic from upstream modules.
# It only calls their public engine interfaces.

# Stage dispatch uses lazy imports so modules that are not needed for a given
# mode do not need to be installed.
# """

# from __future__ import annotations

# import logging
# import time
# from pathlib import Path
# from typing import Any

# from src.pipeline.contracts import (
#     AOIConfig,
#     PipelineConfig,
#     PipelineResult,
#     StageResult,
#     VALID_MODES,
# )
# from src.pipeline.factory import PipelineFactory
# from src.pipeline.runner import StageRunner
# from src.pipeline.validator import PipelineValidator

# __all__ = ["PipelineOrchestrator"]

# _LOGGER: logging.Logger = logging.getLogger(__name__)

# # Ordered stage names per mode.
# _STAGE_ORDER: dict[str, tuple[str, ...]] = {
#     "full": (
#         "bootstrap", "gee_client", "collection", "preprocessing",
#         "features", "export", "patches", "pseudo_labels", "dataset",
#         "dataloader", "transforms", "model", "training",
#         "evaluation", "inference", "analysis", "visualization", "reporting",
#     ),
#     "training":      ("dataloader", "transforms", "model", "training"),
#     "evaluation":    ("dataloader", "transforms", "model", "evaluation"),
#     "inference":     ("inference",),
#     "analysis":      ("analysis",),
#     "visualization": ("visualization",),
#     "reporting":     ("reporting",),
# }


# class PipelineOrchestrator:
#     """
#     Orchestrates the complete pipeline across one or many AOIs.

#     Args:
#         config:          Full project Config object.
#         pipeline_config: PipelineConfig from CLI/config.
#     """

#     def __init__(self, config: Any, pipeline_config: PipelineConfig) -> None:
#         self._config          = config
#         self._pipeline_config = pipeline_config
#         self._validator       = PipelineValidator()

#     def run(self) -> PipelineResult:
#         """
#         Execute the pipeline for all requested AOIs and return PipelineResult.

#         Returns:
#             Frozen PipelineResult with all stage results and provenance.
#         """
#         ops:    list[str] = []
#         t0_all = time.perf_counter()

#         # Step 1: Resolve AOI configs.
#         aoi_configs = PipelineFactory.resolve_aoi_configs(
#             self._pipeline_config, self._config
#         )
#         ops.append(f"aois: {len(aoi_configs)}")

#         # Step 2: Build run_id.
#         run_id = PipelineFactory.make_run_id(self._pipeline_config)
#         ops.append(f"run_id: {run_id}")

#         # Step 3: Pre-flight validation.
#         validation = self._validator.validate(
#             self._pipeline_config, self._config, aoi_configs
#         )
#         for issue in validation.issues:
#             _LOGGER.error("PipelineValidator: %s", issue)
#         for warn in validation.warnings:
#             _LOGGER.warning("PipelineValidator: %s", warn)
#         ops.append(
#             f"validation: {len(validation.issues)} error(s), "
#             f"{len(validation.warnings)} warning(s)"
#         )

#         if validation.issues and not self._pipeline_config.dry_run:
#             # Fatal validation errors: return immediately with failure.
#             return self._make_result(
#                 run_id       = run_id,
#                 mode         = self._pipeline_config.mode,
#                 dry_run      = self._pipeline_config.dry_run,
#                 aoi_configs  = aoi_configs,
#                 all_results  = [],
#                 output_dirs  = {},
#                 success      = False,
#                 t0           = t0_all,
#                 warnings     = tuple(validation.warnings + validation.issues),
#                 ops          = ops,
#             )

#         # Step 4: Resolve stages.
#         stages = _STAGE_ORDER.get(self._pipeline_config.mode, ())
#         ops.append(f"stages: {', '.join(stages)}")

#         # Step 5: Execute stages per AOI.
#         all_results:  list[StageResult] = []
#         output_dirs:  dict[str, str]    = {}
#         runner = StageRunner(dry_run=self._pipeline_config.dry_run)

#         if self._pipeline_config.dry_run:
#             _LOGGER.info("DRY-RUN: plan printed; no stages executed.")
#             for aoi in aoi_configs:
#                 out_dir = PipelineFactory.make_output_dir(
#                     self._pipeline_config.output_dir, run_id, aoi.aoi_id
#                 )
#                 output_dirs[aoi.aoi_id] = out_dir
#                 for stage in stages:
#                     all_results.append(runner.run(stage, aoi.aoi_id, lambda: None))
#         else:
#             for aoi in aoi_configs:
#                 out_dir = PipelineFactory.make_output_dir(
#                     self._pipeline_config.output_dir, run_id, aoi.aoi_id
#                 )
#                 output_dirs[aoi.aoi_id] = out_dir
#                 aoi_results, fatal = self._run_aoi(
#                     aoi, stages, runner, out_dir
#                 )
#                 all_results.extend(aoi_results)
#                 if fatal:
#                     ops.append(f"ABORTED after fatal error in AOI '{aoi.aoi_id}'")
#                     break

#         ops.append(f"completed: {len(all_results)} stage runs")

#         return self._make_result(
#             run_id      = run_id,
#             mode        = self._pipeline_config.mode,
#             dry_run     = self._pipeline_config.dry_run,
#             aoi_configs = aoi_configs,
#             all_results = all_results,
#             output_dirs = output_dirs,
#             success     = all(
#                 r.success or r.skipped for r in all_results
#             ),
#             t0          = t0_all,
#             warnings    = tuple(validation.warnings),
#             ops         = ops,
#         )

#     def _run_aoi(
#         self,
#         aoi:     AOIConfig,
#         stages:  tuple[str, ...],
#         runner:  StageRunner,
#         out_dir: str,
#     ) -> tuple[list[StageResult], bool]:
#         """
#         Execute all stages for one AOI.

#         Returns:
#             (list of StageResult, fatal_error_flag).
#         """
#         results: list[StageResult] = []
#         _LOGGER.info("=== AOI: %s ===", aoi.aoi_id)

#         for stage in stages:
#             fn       = self._build_stage_fn(stage, aoi, out_dir)
#             result   = runner.run(stage, aoi.aoi_id, fn)
#             results.append(result)

#             if not result.success and not result.skipped:
#                 _LOGGER.error(
#                     "Stage '%s' failed for AOI '%s': %s",
#                     stage, aoi.aoi_id, result.error,
#                 )
#                 return results, True   # fatal — stop this AOI

#         return results, False

#     def _build_stage_fn(
#         self,
#         stage:   str,
#         aoi:     AOIConfig,
#         out_dir: str,
#     ):
#         """
#         Return a zero-argument callable for the given stage.

#         Each callable calls the relevant module's public engine/factory
#         interface. Modules that are not needed for this mode are never imported.
#         """
#         config = self._config
#         pcfg   = self._pipeline_config

#         if stage == "bootstrap":
#             def fn():
#                 # Module 2: directory setup and environment bootstrap.
#                 try:
#                     from src.core.setup import setup_project_directories
#                     setup_project_directories(config)
#                 except ImportError:
#                     Path(out_dir).mkdir(parents=True, exist_ok=True)
#                 return [out_dir]
#             return fn

#         if stage == "gee_client":
#             def fn():
#                 from src.gee.client import initialize_gee
#                 initialize_gee(config)
#             return fn

#         if stage == "collection":
#             def fn():
#                 from src.gee.collection import build_collection
#                 build_collection(config, aoi.bbox)
#             return fn

#         if stage == "preprocessing":
#             def fn():
#                 from src.preprocessing.pipeline import run_preprocessing
#                 run_preprocessing(config)
#             return fn

#         if stage == "features":
#             def fn():
#                 from src.features.pipeline import run_feature_engineering
#                 run_feature_engineering(config)
#             return fn

#         if stage == "export":
#             def fn():
#                 from src.export.pipeline import run_export
#                 run_export(config)
#             return fn

#         if stage == "patches":
#             def fn():
#                 from src.patches.generator import generate_patches
#                 generate_patches(config)
#             return fn

#         if stage == "pseudo_labels":
#             def fn():
#                 from src.labels.engine import LabelEngine
#                 engine = LabelEngine(config)
#                 engine.generate()
#             return fn

#         if stage == "dataset":
#             def fn():
#                 from src.dataset.assembler import assemble_dataset
#                 assemble_dataset(config)
#             return fn

#         if stage == "dataloader":
#             def fn():
#                 # Module 11 — TorchDatasetResult
#                 from src.training.dataset import RiverMorphologyDataset
#                 _ = RiverMorphologyDataset(config)
#             return fn

#         if stage == "transforms":
#             def fn():
#                 from src.training.pipeline import TransformPipeline
#                 pipeline = TransformPipeline(config)
#                 return pipeline.build()
#             return fn

#         if stage == "model":
#             def fn():
#                 from src.training.models.factory import ModelFactory
#                 from src.training.models.contracts import ModelConfig
#                 model_cfg = ModelConfig.from_config(config)
#                 result    = ModelFactory.build(model_cfg)
#                 return [str(result.architecture)]
#             return fn

#         if stage == "training":
#             def fn():
#                 from src.training.engine import TrainingEngine, TrainingConfig
#                 train_cfg = TrainingConfig.from_config(config)
#                 engine    = TrainingEngine(train_cfg)
#                 # In the real pipeline, model_result and data_result come from
#                 # upstream stages; here we signal they must be wired by the caller.
#                 _LOGGER.info(
#                     "TrainingEngine ready for AOI '%s'. "
#                     "Call engine.train(model_result, data_result) to execute.",
#                     aoi.aoi_id,
#                 )
#                 return [str(Path(out_dir) / "training")]
#             return fn

#         if stage == "evaluation":
#             def fn():
#                 from src.training.evaluation import EvaluationEngine, EvaluationConfig
#                 eval_cfg = EvaluationConfig.from_config(config)
#                 engine   = EvaluationEngine(eval_cfg)
#                 _LOGGER.info(
#                     "EvaluationEngine ready for AOI '%s'.",
#                     aoi.aoi_id,
#                 )
#                 return [str(Path(out_dir) / "evaluation")]
#             return fn

#         if stage == "inference":
#             def fn():
#                 from src.training.inference import InferenceEngine, InferenceConfig
#                 inf_cfg = InferenceConfig.from_config(config)
#                 engine  = InferenceEngine(inf_cfg)
#                 _LOGGER.info(
#                     "InferenceEngine ready for AOI '%s'.",
#                     aoi.aoi_id,
#                 )
#                 return [str(Path(out_dir) / "inference")]
#             return fn

#         if stage == "analysis":
#             def fn():
#                 from src.morphology import RiverMorphologyEngine, AnalyticsConfig
#                 acfg   = AnalyticsConfig.from_config(config)
#                 engine = RiverMorphologyEngine(acfg)
#                 _LOGGER.info(
#                     "RiverMorphologyEngine ready for AOI '%s'.",
#                     aoi.aoi_id,
#                 )
#                 return [str(Path(out_dir) / "morphology")]
#             return fn

#         if stage == "visualization":
#             def fn():
#                 from src.visualization import VisualizationEngine, VisualizationConfig
#                 vcfg   = VisualizationConfig.from_config(config)
#                 engine = VisualizationEngine(vcfg)
#                 _LOGGER.info(
#                     "VisualizationEngine ready for AOI '%s'.",
#                     aoi.aoi_id,
#                 )
#                 return [str(Path(out_dir) / "visualization")]
#             return fn

#         if stage == "reporting":
#             def fn():
#                 from src.reporting import ReportEngine, ReportingConfig
#                 rcfg   = ReportingConfig.from_config(config)
#                 engine = ReportEngine(rcfg)
#                 _LOGGER.info(
#                     "ReportEngine ready for AOI '%s'.",
#                     aoi.aoi_id,
#                 )
#                 return [str(Path(out_dir) / "reports")]
#             return fn

#         # Unknown stage — no-op.
#         def fn():
#             _LOGGER.warning("Unknown stage '%s'; skipping.", stage)
#         return fn

#     @staticmethod
#     def _make_result(
#         run_id:      str,
#         mode:        str,
#         dry_run:     bool,
#         aoi_configs: list[AOIConfig],
#         all_results: list[StageResult],
#         output_dirs: dict[str, str],
#         success:     bool,
#         t0:          float,
#         warnings:    tuple[str, ...],
#         ops:         list[str],
#     ) -> PipelineResult:
#         import time as _time
#         elapsed    = _time.perf_counter() - t0
#         num_stages = sum(1 for r in all_results if not r.skipped)
#         num_failed = sum(1 for r in all_results if not r.success and not r.skipped)
#         return PipelineResult(
#             run_id           = run_id,
#             mode             = mode,
#             aoi_ids          = tuple(a.aoi_id for a in aoi_configs),
#             stage_results    = tuple(all_results),
#             success          = success,
#             total_duration_s = elapsed,
#             output_dirs      = output_dirs,
#             warnings         = warnings,
#             operations_log   = tuple(ops),
#             dry_run          = dry_run,
#             num_stages       = num_stages,
#             num_failed       = num_failed,
#         )

"""
Pipeline orchestrator for Module 20.

PipelineOrchestrator is the central coordinator that:
1. Resolves the stage list for the requested mode.
2. Iterates over all AOIs (single or batch).
3. Calls each stage via StageRunner.
4. Collects StageResults into the final PipelineResult.

Module calling order (Modules 2-19):
    full:           2→3→4→5→6→7→8→9→10→11→12→13→14→15→16→17→18→19
    training:       11→12→13→14
    evaluation:     11→12→13→15
    inference:      16
    analysis:       17
    visualization:  18
    reporting:      19

The orchestrator NEVER duplicates logic from upstream modules.
It only calls their public engine interfaces.

Stage dispatch uses lazy imports so modules that are not needed for a given
mode do not need to be installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.pipeline.contracts import (
    AOIConfig,
    PipelineConfig,
    PipelineResult,
    StageResult,
    VALID_MODES,
)
from src.pipeline.factory import PipelineFactory
from src.pipeline.runner import StageRunner
from src.pipeline.validator import PipelineValidator

__all__ = ["PipelineOrchestrator"]

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Ordered stage names per mode.
_STAGE_ORDER: dict[str, tuple[str, ...]] = {
    "full": (
        "bootstrap", "gee_client", "collection", "preprocessing",
        "features", "export", "patches", "pseudo_labels", "dataset",
        "dataloader", "transforms", "model", "training",
        "evaluation", "inference", "analysis", "visualization", "reporting",
    ),
    "training":      ("dataloader", "transforms", "model", "training"),
    "evaluation":    ("dataloader", "transforms", "model", "evaluation"),
    "inference":     ("inference",),
    "analysis":      ("analysis",),
    "visualization": ("visualization",),
    "reporting":     ("reporting",),
}


class PipelineOrchestrator:
    """
    Orchestrates the complete pipeline across one or many AOIs.

    Args:
        config:          Full project Config object.
        pipeline_config: PipelineConfig from CLI/config.
    """

    def __init__(self, config: Any, pipeline_config: PipelineConfig) -> None:
        self._config          = config
        self._pipeline_config = pipeline_config
        self._validator       = PipelineValidator()

    def run(self) -> PipelineResult:
        """
        Execute the pipeline for all requested AOIs and return PipelineResult.

        Returns:
            Frozen PipelineResult with all stage results and provenance.
        """
        ops:    list[str] = []
        t0_all = time.perf_counter()

        # Step 1: Resolve AOI configs.
        aoi_configs = PipelineFactory.resolve_aoi_configs(
            self._pipeline_config, self._config
        )
        ops.append(f"aois: {len(aoi_configs)}")

        # Step 2: Build run_id.
        run_id = PipelineFactory.make_run_id(self._pipeline_config)
        ops.append(f"run_id: {run_id}")

        # Step 3: Pre-flight validation.
        validation = self._validator.validate(
            self._pipeline_config, self._config, aoi_configs
        )
        for issue in validation.issues:
            _LOGGER.error("PipelineValidator: %s", issue)
        for warn in validation.warnings:
            _LOGGER.warning("PipelineValidator: %s", warn)
        ops.append(
            f"validation: {len(validation.issues)} error(s), "
            f"{len(validation.warnings)} warning(s)"
        )

        if validation.issues and not self._pipeline_config.dry_run:
            # Fatal validation errors: return immediately with failure.
            return self._make_result(
                run_id       = run_id,
                mode         = self._pipeline_config.mode,
                dry_run      = self._pipeline_config.dry_run,
                aoi_configs  = aoi_configs,
                all_results  = [],
                output_dirs  = {},
                success      = False,
                t0           = t0_all,
                warnings     = tuple(validation.warnings + validation.issues),
                ops          = ops,
            )

        # Step 4: Resolve stages.
        stages = _STAGE_ORDER.get(self._pipeline_config.mode, ())
        ops.append(f"stages: {', '.join(stages)}")

        # Step 5: Execute stages per AOI.
        all_results:  list[StageResult] = []
        output_dirs:  dict[str, str]    = {}
        runner = StageRunner(dry_run=self._pipeline_config.dry_run)

        if self._pipeline_config.dry_run:
            _LOGGER.info("DRY-RUN: plan printed; no stages executed.")
            for aoi in aoi_configs:
                out_dir = PipelineFactory.make_output_dir(
                    self._pipeline_config.output_dir, run_id, aoi.aoi_id
                )
                output_dirs[aoi.aoi_id] = out_dir
                for stage in stages:
                    all_results.append(runner.run(stage, aoi.aoi_id, lambda: None))
        else:
            for aoi in aoi_configs:
                out_dir = PipelineFactory.make_output_dir(
                    self._pipeline_config.output_dir, run_id, aoi.aoi_id
                )
                output_dirs[aoi.aoi_id] = out_dir
                aoi_results, fatal = self._run_aoi(
                    aoi, stages, runner, out_dir
                )
                all_results.extend(aoi_results)
                if fatal:
                    ops.append(f"ABORTED after fatal error in AOI '{aoi.aoi_id}'")
                    break

        ops.append(f"completed: {len(all_results)} stage runs")

        return self._make_result(
            run_id      = run_id,
            mode        = self._pipeline_config.mode,
            dry_run     = self._pipeline_config.dry_run,
            aoi_configs = aoi_configs,
            all_results = all_results,
            output_dirs = output_dirs,
            success     = all(
                r.success or r.skipped for r in all_results
            ),
            t0          = t0_all,
            warnings    = tuple(validation.warnings),
            ops         = ops,
        )

    def _run_aoi(
        self,
        aoi:     AOIConfig,
        stages:  tuple[str, ...],
        runner:  StageRunner,
        out_dir: str,
    ) -> tuple[list[StageResult], bool]:
        """
        Execute all stages for one AOI.

        stage_state carries live result objects between stages so downstream
        engines receive the actual outputs of their upstream stages.

        Returns:
            (list of StageResult, fatal_error_flag).
        """
        results:     list[StageResult] = []
        stage_state: dict              = {}   # shared by all fn() closures for this AOI
        _LOGGER.info("=== AOI: %s ===", aoi.aoi_id)

        for stage in stages:
            fn       = self._build_stage_fn(stage, aoi, out_dir, stage_state)
            result   = runner.run(stage, aoi.aoi_id, fn)
            results.append(result)

            if not result.success and not result.skipped:
                _LOGGER.error(
                    "Stage '%s' failed for AOI '%s': %s",
                    stage, aoi.aoi_id, result.error,
                )
                return results, True   # fatal — stop this AOI

        return results, False

    def _build_stage_fn(
        self,
        stage:       str,
        aoi:         AOIConfig,
        out_dir:     str,
        stage_state: dict,
    ):
        """
        Return a zero-argument callable for the given stage.

        stage_state is a mutable dict shared across all stages for one AOI.
        Each fn() reads the result objects it needs and writes its own result
        back so the next stage can read it.

        Keys written per stage:
            "transforms"    -> TransformPipelineResult  (read by: training, evaluation)
            "model"         -> ModelResult              (read by: training)
            "training"      -> TrainingResult           (read by: evaluation, inference)
            "evaluation"    -> EvaluationResult         (read by: reporting)
            "inference"     -> InferenceResult          (read by: analysis, reporting)
            "analysis"      -> RiverMorphologyResult    (read by: visualization, reporting)
            "visualization" -> VisualizationResult      (read by: reporting)

        Modules that are not needed for this mode are never imported.
        """
        config = self._config
        pcfg   = self._pipeline_config

        if stage == "bootstrap":
            def fn():
                # Module 2: directory setup and environment bootstrap.
                try:
                    from src.core.setup import setup_project_directories
                    setup_project_directories(config)
                except ImportError:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                return [out_dir]
            return fn

        if stage == "gee_client":
            def fn():
                from src.gee.client import initialize_gee
                initialize_gee(config)
            return fn

        if stage == "collection":
            def fn():
                from src.gee.collection import build_collection
                build_collection(config, aoi.bbox)
            return fn

        if stage == "preprocessing":
            def fn():
                from src.preprocessing.pipeline import run_preprocessing
                run_preprocessing(config)
            return fn

        if stage == "features":
            def fn():
                from src.features.pipeline import run_feature_engineering
                run_feature_engineering(config)
            return fn

        if stage == "export":
            def fn():
                from src.export.pipeline import run_export
                run_export(config)
            return fn

        if stage == "patches":
            def fn():
                from src.patches.generator import generate_patches
                generate_patches(config)
            return fn

        if stage == "pseudo_labels":
            def fn():
                from src.labels.engine import LabelEngine
                engine = LabelEngine(config)
                engine.generate()
            return fn

        if stage == "dataset":
            def fn():
                from src.dataset.assembler import assemble_dataset
                assemble_dataset(config)
            return fn

        if stage == "dataloader":
            def fn():
                # Module 11 — build dataset; result stored for transforms stage.
                from src.training.dataset import RiverMorphologyDataset
                result = RiverMorphologyDataset(config)
                stage_state["dataloader"] = result
            return fn

        if stage == "transforms":
            def fn():
                from src.training.pipeline import TransformPipeline
                torch_result = stage_state.get("dataloader")
                pipeline     = TransformPipeline(config)
                result       = pipeline.build(torch_result)
                stage_state["transforms"] = result
                return result
            return fn

        if stage == "model":
            def fn():
                from src.training.models.factory import ModelFactory
                from src.training.models.contracts import ModelConfig
                model_cfg = ModelConfig.from_config(config)
                result    = ModelFactory.build(model_cfg)
                stage_state["model"] = result
                return [str(result.architecture)]
            return fn

        if stage == "training":
            def fn():
                from src.training.engine import TrainingEngine, TrainingConfig
                train_cfg    = TrainingConfig.from_config(config)
                engine       = TrainingEngine(train_cfg)
                model_result = stage_state.get("model")
                data_result  = stage_state.get("transforms")
                result       = engine.train(model_result, data_result)
                stage_state["training"] = result
                return [str(Path(out_dir) / "training")]
            return fn

        if stage == "evaluation":
            def fn():
                from src.training.evaluation import EvaluationEngine, EvaluationConfig
                eval_cfg         = EvaluationConfig.from_config(config)
                engine           = EvaluationEngine(eval_cfg)
                training_result  = stage_state.get("training")
                data_result      = stage_state.get("transforms")
                result           = engine.evaluate(training_result, data_result)
                stage_state["evaluation"] = result
                return [str(Path(out_dir) / "evaluation")]
            return fn

        if stage == "inference":
            def fn():
                from src.training.inference import InferenceEngine, InferenceConfig
                from src.training.models.factory import ModelFactory
                from src.training.models.contracts import ModelConfig
                inf_cfg         = InferenceConfig.from_config(config)
                engine          = InferenceEngine(inf_cfg)
                training_result = stage_state.get("training")
                data_result     = stage_state.get("transforms")
                # Standalone inference (--mode inference) has no prior training stage.
                # Build a fresh model from config so CheckpointLoader can restore weights.
                standalone_model = None
                if training_result is None:
                    model_cfg        = ModelConfig.from_config(config)
                    model_result     = ModelFactory.build(model_cfg)
                    standalone_model = model_result.model
                result = engine.predict(
                    training_result,
                    data_result,
                    model = standalone_model,
                )
                stage_state["inference"] = result
                return [str(Path(out_dir) / "inference")]
            return fn

        if stage == "analysis":
            def fn():
                from src.morphology import RiverMorphologyEngine, AnalyticsConfig
                acfg             = AnalyticsConfig.from_config(config)
                engine           = RiverMorphologyEngine(acfg)
                inference_result = stage_state.get("inference")
                result           = engine.analyze(inference_result)
                stage_state["analysis"] = result
                return [str(Path(out_dir) / "morphology")]
            return fn

        if stage == "visualization":
            def fn():
                from src.visualization import VisualizationEngine, VisualizationConfig
                vcfg              = VisualizationConfig.from_config(config)
                engine            = VisualizationEngine(vcfg)
                morphology_result = stage_state.get("analysis")
                result            = engine.visualize(morphology_result)
                stage_state["visualization"] = result
                return [str(Path(out_dir) / "visualization")]
            return fn

        if stage == "reporting":
            def fn():
                from src.reporting import ReportEngine, ReportingConfig
                rcfg                 = ReportingConfig.from_config(config)
                engine               = ReportEngine(rcfg)
                evaluation_result    = stage_state.get("evaluation")
                inference_result     = stage_state.get("inference")
                morphology_result    = stage_state.get("analysis")
                visualization_result = stage_state.get("visualization")
                result               = engine.generate(
                    evaluation_result    = evaluation_result,
                    inference_result     = inference_result,
                    morphology_result    = morphology_result,
                    visualization_result = visualization_result,
                )
                stage_state["reporting"] = result
                return [str(Path(out_dir) / "reports")]
            return fn

        # Unknown stage — no-op.
        def fn():
            _LOGGER.warning("Unknown stage '%s'; skipping.", stage)
        return fn

    @staticmethod
    def _make_result(
        run_id:      str,
        mode:        str,
        dry_run:     bool,
        aoi_configs: list[AOIConfig],
        all_results: list[StageResult],
        output_dirs: dict[str, str],
        success:     bool,
        t0:          float,
        warnings:    tuple[str, ...],
        ops:         list[str],
    ) -> PipelineResult:
        import time as _time
        elapsed    = _time.perf_counter() - t0
        num_stages = sum(1 for r in all_results if not r.skipped)
        num_failed = sum(1 for r in all_results if not r.success and not r.skipped)
        return PipelineResult(
            run_id           = run_id,
            mode             = mode,
            aoi_ids          = tuple(a.aoi_id for a in aoi_configs),
            stage_results    = tuple(all_results),
            success          = success,
            total_duration_s = elapsed,
            output_dirs      = output_dirs,
            warnings         = warnings,
            operations_log   = tuple(ops),
            dry_run          = dry_run,
            num_stages       = num_stages,
            num_failed       = num_failed,
        )