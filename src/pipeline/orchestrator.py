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
    evaluation:     11→12→13→14→15
    inference:      16
    analysis:       17
    visualization:  18
    reporting:      19

Contract flow:
    Config
    → (M3) EarthEngineClient    → authenticated GEE session
    → (M4) LandsatCollectionBuilder → CollectionResult
    → (M5) LandsatPreprocessor + LandsatCompositor → ProcessedCollectionResult / CompositeResult
    → (M6) FeatureStackBuilder  → FeatureStackResult
    → (M7) DatasetExporter      → DatasetExportResult
    → (M8) PatchGenerator       → PatchDatasetResult
    → (M9) LabelManager         → LabelDatasetResult
    → (M10) DatasetAssembler    → TrainingDatasetResult
    → (M11) DataLoaderFactory   → DataLoaderBundle
    → (M12) TransformPipeline   → TransformPipelineResult
    → (M13) ModelFactory        → ModelResult
    → (M14) TrainingEngine      → TrainingResult
    → (M15) EvaluationEngine    → EvaluationResult
    → (M16) InferenceEngine     → InferenceResult
    → (M17) RiverMorphologyEngine → RiverMorphologyResult
    → (M18) VisualizationEngine → VisualizationResult
    → (M19) ReportEngine        → ReportResult
    → (M20) PipelineOrchestrator → PipelineResult

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
# from unittest import result

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

# ---------------------------------------------------------------------------
# Ordered stage names per mode.
# Each stage name maps to one _build_stage_fn branch below.
# ---------------------------------------------------------------------------
_STAGE_ORDER: dict[str, tuple[str, ...]] = {
    "full": (
        "bootstrap",      # M2  — bootstrap + directory creation
        "gee_client",     # M3  — EarthEngineClient authentication
        "collection",     # M4  — LandsatCollectionBuilder → CollectionResult
        "preprocessing",  # M5  — LandsatPreprocessor + LandsatCompositor
        "features",       # M6  — FeatureStackBuilder → FeatureStackResult
        "export",         # M7  — DatasetExporter → DatasetExportResult
        "patches",        # M8  — PatchGenerator → PatchDatasetResult
        "pseudo_labels",  # M9  — LabelManager → LabelDatasetResult
        "dataset",        # M10 — DatasetAssembler → TrainingDatasetResult
        "dataloader",     # M11 — DataLoaderFactory → DataLoaderBundle
        "transforms",     # M12 — TransformPipeline → TransformPipelineResult
        "model",          # M13 — ModelFactory → ModelResult
        "training",       # M14 — TrainingEngine → TrainingResult
        "evaluation",     # M15 — EvaluationEngine → EvaluationResult
        "inference",      # M16 — InferenceEngine → InferenceResult
        "analysis",       # M17 — RiverMorphologyEngine → RiverMorphologyResult
        "visualization",  # M18 — VisualizationEngine → VisualizationResult
        "reporting",      # M19 — ReportEngine → ReportResult
    ),
    # training mode: load data → augment → build model → train
    "training":      ("dataloader", "transforms", "model", "training"),
    # evaluation mode: load data → augment → build model → train → evaluate
    "evaluation":    ("dataloader", "transforms", "model", "training", "evaluation"),
    # remaining modes are single-stage; they read from checkpoint/disk
    "inference":     ("inference",),
    "analysis":      ("analysis",),
    "visualization": ("visualization",),
    "reporting":     ("reporting",),
}


class PipelineOrchestrator:
    """
    Orchestrates the complete pipeline across one or many AOIs.

    Args:
        config:          Full project Config object (from src.core.config).
        pipeline_config: PipelineConfig from CLI / config.pipeline.
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
            return self._make_result(
                run_id      = run_id,
                mode        = self._pipeline_config.mode,
                dry_run     = self._pipeline_config.dry_run,
                aoi_configs = aoi_configs,
                all_results = [],
                output_dirs = {},
                success     = False,
                t0          = t0_all,
                warnings    = tuple(validation.warnings + validation.issues),
                ops         = ops,
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
                aoi_results, fatal = self._run_aoi(aoi, stages, runner, out_dir)
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
            success     = all(r.success or r.skipped for r in all_results),
            t0          = t0_all,
            warnings    = tuple(validation.warnings),
            ops         = ops,
        )

    # ------------------------------------------------------------------
    # AOI execution
    # ------------------------------------------------------------------

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
        engines receive the actual typed contracts from their upstream stages.

        Returns:
            (list of StageResult, fatal_error_flag).
        """
        results:     list[StageResult] = []
        stage_state: dict              = {}   # shared by all fn() closures for this AOI
        _LOGGER.info("=== AOI: %s ===", aoi.aoi_id)

        for stage in stages:
            fn     = self._build_stage_fn(stage, aoi, out_dir, stage_state)
            result = runner.run(stage, aoi.aoi_id, fn)
            results.append(result)

            if not result.success and not result.skipped:
                _LOGGER.error(
                    "Stage '%s' failed for AOI '%s': %s",
                    stage, aoi.aoi_id, result.error,
                )
                return results, True   # fatal — stop this AOI

        return results, False

    # ------------------------------------------------------------------
    # Stage function builder
    # ------------------------------------------------------------------

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
        Each fn() reads the typed contract it needs and writes its own result back.

        stage_state keys (contract types):
            "gee_client"    -> EarthEngineClient          (read by: collection)
            "collection"    -> CollectionResult           (read by: preprocessing)
            "preprocessing" -> CompositeResult            (read by: features)
            "features"      -> FeatureStackResult         (read by: export)
            "export"        -> DatasetExportResult        (read by: patches)
            "patches"       -> PatchDatasetResult         (read by: pseudo_labels, dataset)
            "pseudo_labels" -> LabelDatasetResult         (read by: dataset)
            "dataset"       -> TrainingDatasetResult      (read by: dataloader)
            "dataloader"    -> DataLoaderBundle           (read by: transforms)
            "transforms"    -> TransformPipelineResult    (read by: training, evaluation)
            "model"         -> ModelResult                (read by: training)
            "training"      -> TrainingResult             (read by: evaluation, inference)
            "evaluation"    -> EvaluationResult           (read by: reporting)
            "inference"     -> InferenceResult            (read by: analysis, reporting)
            "analysis"      -> RiverMorphologyResult      (read by: visualization, reporting)
            "visualization" -> VisualizationResult        (read by: reporting)

        Modules not needed for the current mode are never imported.
        """
        config = self._config

        # ------------------------------------------------------------------ M2
        if stage == "bootstrap":
            def fn():
                # Module 2: bootstrap() validates environment + initialises logging.
                # DirectoryManager.create_all() creates the project directory tree.
                from src.core.bootstrap import bootstrap
                from src.core.directories import DirectoryManager
                bootstrap(
                    config_path  = getattr(config, "_config_path", "config/config.yaml"),
                    env_file     = None,
                    strict_env   = False,
                    print_summary= False,
                )
                manager = DirectoryManager(config)
                manager.create_all()
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                return [out_dir]
            return fn

        # ------------------------------------------------------------------ M3
        if stage == "gee_client":
            def fn():
                # Module 3: EarthEngineClient encapsulates all GEE authentication.
                # No other module may import earthengine-api directly.
                from src.gee.client import EarthEngineClient
                client = EarthEngineClient(config)
                client.initialize()
                stage_state["gee_client"] = client
            return fn

        # ------------------------------------------------------------------ M4
        if stage == "collection":
            def fn():
                # Module 4: LandsatCollectionBuilder → CollectionResult
                from src.gee.collections import LandsatCollectionBuilder
                client = stage_state.get("gee_client")
                result = (
                    LandsatCollectionBuilder(client, config)
                    .with_aoi_from_config()
                    .with_date_range_from_config()
                    .with_cloud_cover(config.satellite.max_cloud_cover_percent)
                    .with_auto_sensors()
                    .build()
                )
                stage_state["collection"] = result
            return fn

        # ------------------------------------------------------------------ M5
        if stage == "preprocessing":
            def fn():
                # Module 5: LandsatPreprocessor → ProcessedCollectionResult
                #           LandsatCompositor   → CompositeResult  (passed downstream)
                from src.gee.preprocessing import LandsatPreprocessor
                from src.gee.composite import LandsatCompositor
                collection_result = stage_state.get("collection")
                client = stage_state.get("gee_client")
                preprocessed = LandsatPreprocessor(client = client, config = config).process(collection_result)
                composite    = LandsatCompositor(client = client, config = config).build_composite(preprocessed)
                # Store the CompositeResult; FeatureStackBuilder consumes it.
                stage_state["preprocessing"] = composite
            return fn

        # ------------------------------------------------------------------ M6
        # if stage == "features":
        #     def fn():
        #         # Module 6: FeatureStackBuilder → FeatureStackResult
        #         from src.gee.features import FeatureStackBuilder
        #         composite_result = stage_state.get("preprocessing")
        #         client = stage_state.get("gee_client")
        #         result = FeatureStackBuilder(client = client, config = config).build(composite_result)
        #         stage_state["features"] = result
        #     return fn
        # ------------------------------------------------------------------ M6
        if stage == "features":
                def fn():
                    # Module 6: SpectralFeatureGenerator → FeatureStackResult
                    from src.gee.features import SpectralFeatureGenerator
                    composite_result = stage_state.get("preprocessing")
                    client = stage_state.get("gee_client")
                    result = (
                        SpectralFeatureGenerator(
                            client=client,
                            config=config,
                        )
                        .generate(composite_result)
                    )
                    stage_state["features"] = result
                return fn

        # ------------------------------------------------------------------ M7
        # if stage == "export":
        #     def fn():
        #         # Module 7: DatasetExporter → DatasetExportResult
        #         # Public entry: src/export/exporter.py :: DatasetExporter
        #         from src.export.exporter import DatasetExporter
        #         feature_stack_result = stage_state.get("features")
        #         exporter = DatasetExporter(config)
        #         result   = exporter.export(feature_stack_result)
        #         stage_state["export"] = result
        #     return fn
        # ------------------------------------------------------------------ M7
        if stage == "export":
            def fn():
                # Module 7: DatasetExporter → DatasetExportResult
                from pathlib import Path
                from src.export.exporter import DatasetExporter
                feature_stack_result = stage_state.get("features")
                client = stage_state.get("gee_client")
                exporter = DatasetExporter(
                    client=client,
                    config=config,
                )
                result = exporter.export(
                    feature_stack_result=feature_stack_result,
                    output_dir=Path(config.paths.processed_dir),
                )
                stage_state["export"] = result
            return fn

        # ------------------------------------------------------------------ M8
        if stage == "patches":
            def fn():
                # Module 8: PatchGenerator → PatchDatasetResult
                # Public entry: src/patches/generator.py :: PatchGenerator
                from src.patches.generator import PatchGenerator
                export_result = stage_state.get("export")
                from pathlib import Path
                generator = PatchGenerator(config)
                # result    = generator.generate(export_result, Path(config.paths.patches_dir ))
                result = generator.generate(
                    export_result,
                    config.paths.patches_dir,
                )
                stage_state["patches"] = result
            return fn

        # ------------------------------------------------------------------ M9
        # if stage == "pseudo_labels":
        #     def fn():
        #         # Module 9: LabelManager → LabelDatasetResult
        #         # Public entry: src/labels/manager.py :: LabelManager
        #         # NOT: src/labels/engine.py (does not exist)
        #         from src.labels.manager import LabelManager
        #         patch_result = stage_state.get("patches")
        #         manager = LabelManager(config)
        #         result  = manager.generate(patch_result)
        #         stage_state["pseudo_labels"] = result
        #     return fn
        # ------------------------------------------------------------------ M9
        if stage == "pseudo_labels":
            def fn():
                from src.labels.manager import LabelManager
                patch_result = stage_state.get("patches")
                export_result = stage_state.get("export")

                manager = LabelManager(config)

                result = manager.generate(
                    patch_dataset_result=patch_result,
                    scene_metadata=export_result.scene_metadata,
                    output_dir=export_result.dataset_root,
                    aoi_id=aoi.aoi_id,
                )
                stage_state["pseudo_labels"] = result
            return fn

        # ------------------------------------------------------------------ M10
        if stage == "dataset":
            def fn():
                # Module 10: DatasetAssembler → TrainingDatasetResult
                # Public entry: src/dataset/assembler.py :: DatasetAssembler
                # Input contracts: PatchDatasetResult + LabelDatasetResult
                from src.dataset.assembler import DatasetAssembler
                patch_result = stage_state.get("patches")
                label_result = stage_state.get("pseudo_labels")
                assembler = DatasetAssembler(config)
                # result    = assembler.assemble(patch_result, label_result)
                result = assembler.assemble(
                    patch_results=[patch_result],
                    label_results=[label_result],
                    output_dir=config.paths.processed_dir,
                )
                stage_state["dataset"] = result
            return fn

        # ------------------------------------------------------------------ M11
        if stage == "dataloader":
            def fn():
                # Module 11: DataLoaderFactory → DataLoaderBundle
                # Public entry: src/training/dataloader.py :: DataLoaderFactory
                # Input contract: TrainingDatasetResult (from M10)
                # Output contract: DataLoaderBundle (train_loader, val_loader, test_loader)
                from src.training.dataloader import DataLoaderFactory
                from src.labels.schema import ClassSchema

                training_dataset_result = stage_state.get("dataset")
                class_schema = ClassSchema.from_config(config)

                # factory = DataLoaderFactory(config)
                # result  = factory.create(training_dataset_result)
                factory = DataLoaderFactory(
                    config=config,
                    class_schema=class_schema,
                )
                output_dir = Path(config.paths.processed_dir)
                result = factory.build(
                    training_dataset_result,
                    output_dir,
                )
                # result = factory.build(
                #     training_result=training_dataset_result,
                #     output_dir=training_dataset_result.output_dir,
                # )
                stage_state["dataloader"] = result
                return result
            return fn

        # ------------------------------------------------------------------ M12
        if stage == "transforms":
            def fn():
                # Module 12: TransformPipeline → TransformPipelineResult
                # Public entry: src/training/pipeline.py :: TransformPipeline
                # Input contract: DataLoaderBundle (from M11)
                from src.training.pipeline import TransformPipeline
                dataloader_bundle = stage_state.get("dataloader")
                pipeline = TransformPipeline(config)
                result   = pipeline.build(dataloader_bundle)
                stage_state["transforms"] = result
                return result
            return fn

        # ------------------------------------------------------------------ M13
        if stage == "model":
            def fn():
                # Module 13: ModelFactory → ModelResult
                # Public entry: src/training/models/factory.py :: ModelFactory
                from src.training.models.factory import ModelFactory
                from src.training.models.contracts import ModelConfig

                model_cfg = ModelConfig.from_config(config)

                # result    = ModelFactory.build(model_cfg)
                result = ModelFactory.build_from_model_config(model_cfg)
                stage_state["model"] = result
                return [str(result.architecture)]
            return fn

        # ------------------------------------------------------------------ M14
        if stage == "training":
            def fn():
                # Module 14: TrainingEngine → TrainingResult
                # Inputs: ModelResult (M13) + TransformPipelineResult (M12)
                from src.training.engine import TrainingEngine, TrainingConfig
                train_cfg    = TrainingConfig.from_config(config)
                engine       = TrainingEngine(train_cfg)
                model_result = stage_state.get("model")
                data_result  = stage_state.get("transforms")
                result       = engine.train(model_result, data_result)
                stage_state["training"] = result
                return [str(Path(out_dir) / "training")]
            return fn

        # ------------------------------------------------------------------ M15
        if stage == "evaluation":
            def fn():
                # Module 15: EvaluationEngine → EvaluationResult
                # Inputs: TrainingResult (M14) + TransformPipelineResult (M12)
                from src.training.evaluation import EvaluationEngine, EvaluationConfig
                eval_cfg        = EvaluationConfig.from_config(config)
                engine          = EvaluationEngine(eval_cfg)
                training_result = stage_state.get("training")
                data_result     = stage_state.get("transforms")
                result          = engine.evaluate(training_result, data_result)
                stage_state["evaluation"] = result
                return [str(Path(out_dir) / "evaluation")]
            return fn

        # ------------------------------------------------------------------ M16
        if stage == "inference":
            def fn():
                # Module 16: InferenceEngine → InferenceResult
                # In full/evaluation mode: uses TrainingResult from stage_state.
                # In standalone inference mode (--mode inference): no prior training
                # stage runs, so we reconstruct the model from ModelFactory and
                # let CheckpointLoader restore weights from checkpoint_dir.
                from src.training.inference import InferenceEngine, InferenceConfig
                from src.training.models.factory import ModelFactory
                from src.training.models.contracts import ModelConfig
                inf_cfg         = InferenceConfig.from_config(config)
                engine          = InferenceEngine(inf_cfg)
                training_result = stage_state.get("training")
                data_result     = stage_state.get("transforms")
                standalone_model = None
                if training_result is None:
                    # Standalone: build a fresh model architecture for checkpoint loading.
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

        # ------------------------------------------------------------------ M17
        if stage == "analysis":
            def fn():
                # Module 17: RiverMorphologyEngine → RiverMorphologyResult
                # Input: InferenceResult (M16)
                from src.morphology import RiverMorphologyEngine, AnalyticsConfig
                acfg             = AnalyticsConfig.from_config(config)
                engine           = RiverMorphologyEngine(acfg)
                inference_result = stage_state.get("inference")
                result           = engine.analyze(inference_result)
                stage_state["analysis"] = result
                return [str(Path(out_dir) / "morphology")]
            return fn

        # ------------------------------------------------------------------ M18
        if stage == "visualization":
            def fn():
                # Module 18: VisualizationEngine → VisualizationResult
                # Input: RiverMorphologyResult (M17)
                from src.visualization import VisualizationEngine, VisualizationConfig
                vcfg              = VisualizationConfig.from_config(config)
                engine            = VisualizationEngine(vcfg)
                morphology_result = stage_state.get("analysis")
                result            = engine.visualize(morphology_result)
                stage_state["visualization"] = result
                return [str(Path(out_dir) / "visualization")]
            return fn

        # ------------------------------------------------------------------ M19
        if stage == "reporting":
            def fn():
                # Module 19: ReportEngine → ReportResult
                # Inputs: EvaluationResult + InferenceResult +
                #         RiverMorphologyResult + VisualizationResult
                from src.reporting import ReportEngine, ReportingConfig
                rcfg                 = ReportingConfig.from_config(config)
                engine               = ReportEngine(rcfg)
                evaluation_result    = stage_state.get("evaluation")
                inference_result     = stage_state.get("inference")
                morphology_result    = stage_state.get("analysis")
                visualization_result = stage_state.get("visualization")
                result = engine.generate(
                    evaluation_result    = evaluation_result,
                    inference_result     = inference_result,
                    morphology_result    = morphology_result,
                    visualization_result = visualization_result,
                )
                stage_state["reporting"] = result
                return [str(Path(out_dir) / "reports")]
            return fn

        # Unknown stage — no-op with warning.
        def fn():
            _LOGGER.warning("Unknown stage '%s'; skipping.", stage)
        return fn

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

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

