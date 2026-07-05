# Project Architecture
## Module 3:

EarthEngineClient  (src/gee/client.py)
    |
    +-- AuthManager        (src/gee/auth.py)   <- handles ee.Authenticate + ee.Initialize
    |       |
    |       +-- detect_runtime()               <- Colab vs Local detection
    |
    +-- HealthChecker      (src/gee/health.py) <- structured connectivity verification
    |       |
    |       +-- HealthReport / HealthCheckItem <- structured results
    |
    +-- RetryConfig        (src/gee/client.py) <- exponential backoff configuration
    |
    +-- _is_transient_error()                  <- classify retryable vs fatal errors

GEE Exceptions     (src/gee/__init__.py)       <- defined FIRST, imported by submodules

## Module 4:
LandsatCollectionBuilder    (collections.py)
    |
    +-- with_date_range()            -- pure Python, sets state
    +-- with_date_range_from_config()-- reads Config
    +-- with_aoi()                   -- accepts pre-built ee.Geometry
    +-- with_aoi_from_config()       -- calls client.get_aoi_geometry()
    +-- with_cloud_cover()           -- pure Python
    +-- with_auto_sensors()          -- defers to build()
    +-- with_sensors()               -- manual sensor list
    +-- build()                      -- assembles + filters collection
         |
         +-- _resolve_sensors()          <- SensorAvailabilityPeriod overlap check
         +-- _build_merged_collection()  <- ee.ImageCollection + merge
         +-- _apply_all_filters()
               |
               +-- filter_by_date()    (filters.py)
               +-- filter_by_bounds()  (filters.py)
               +-- filter_by_cloud_cover() (filters.py)
         +-- returns CollectionResult

MetadataExtractor           (metadata.py)
    +-- get_image_count()         -- size().getInfo()
    +-- get_band_names()          -- first().bandNames().getInfo()
    +-- get_image_ids()           -- aggregate_array().getInfo()
    +-- get_acquisition_dates()   -- aggregate_array().getInfo()
    +-- get_spacecraft_ids()      -- aggregate_array().getInfo()
    +-- get_temporal_coverage()   -- derived from dates
    +-- get_crs_and_scale()       -- projection().getInfo()
    +-- extract_all()             -- all of the above
    +-- returns CollectionMetadata

## Module 5:
LandsatPreprocessor          (preprocessing.py)
    |
    +-- _apply_scaling()          <- closure: _make_scale_function()
    |       |
    |       +-- image.select('SR_B.*').multiply().add()  [server-side]
    |       +-- image.select('ST_B.*').multiply().add()  [server-side]
    |
    +-- _apply_masking()          <- LandsatQAMasker.apply_to_collection()
    |       |                        (masking.py)
    |       +-- QAMaskConfig      <- frozen dataclass (configurable flags)
    |       +-- _build_qa_mask()  <- bitwiseAnd chains  [server-side]
    |
    +-- _apply_harmonization()    <- BandHarmonizer.harmonize_collection()
    |       |                        (harmonization.py)
    |       +-- ee.Algorithms.If(is_oli, rename_oli, rename_tm_etm)  [server-side]
    |
    +-- returns ProcessedCollectionResult

LandsatCompositor            (composite.py)
    +-- build_composite(ProcessedCollectionResult, method) -> CompositeResult
         |
         +-- _median()     collection.median()                [server-side]
         +-- _mean()       collection.mean()                  [server-side]
         +-- _mosaic()     collection.mosaic()                [server-side]
         +-- _medoid()     qualityMosaic(-distance)           [server-side]
         +-- _percentile() collection.reduce(Reducer.pct)    [server-side]

## Module 6:
FeatureStackResult (from Module 6)
        |
        v
DatasetExporter.export()          <-- orchestrator only
        |
        +-- EarthEngineDownloader.download()    -> DownloadResult
        +-- GeoTiffWriter.write()               -> GeoTiffWriteResult
        +-- MetadataWriter.generate() + save()  -> SceneMetadata
        +-- GeoTiffValidator.validate()         -> GeoTiffValidationResult
        +-- DatasetVersionManager.generate()    -> VersionInfo
        +-- DatasetManifestManager.add/save()   -> DatasetManifest
        |
        v
DatasetExportResult                <-- typed output contract

Output layout:
{output_dir}/
    version.json
    manifest.csv
    manifest.json
    scenes/
        {scene_id}/
            image.tif
            metadata.json

## Module 8
PatchGenerator                  (generator.py)   <-- orchestrator only
    |
    +-- PatchTiler.compute_windows()        (tiler.py)     <-- pure index arithmetic
    |
    +-- PatchReader.read_window()           (reader.py)    <-- rasterio window read
    |
    +-- PatchValidator.validate()           (validator.py) <-- NoData ratio check
    |
    +-- GeoTiffWriter.write()               (REUSED from src.export.geotiff, Module 7)
    |
    +-- PatchManifestManager.add_entry()/save()  (manifest.py)
    |
    v
PatchDatasetResult               <-- typed output contract

## Module 9
PatchDatasetResult + SceneMetadata
        │
        ▼
LabelManager.generate()          <-- orchestrator only (updated)
        │
        ├── SpectralBandReader   (classifier.py)  reads patch GeoTIFF
        ├── RuleEngine           (rules.py)        applies spectral rules
        ├── ConflictResolver     (conflicts.py)    resolves class conflicts
        ├── MorphologyProcessor  (morphology.py)   cleans mask geometry
        ├── QualityAssessment    (quality.py)      scores mask quality
        ├── ConfidenceEstimator  (confidence.py)   pixel/mask confidence
        ├── PseudoLabelGenerator (generator.py)    orchestrates + saves mask
        ├── LabelValidator       (validator.py)    REUSED unchanged
        ├── LabelStatisticsCalculator (statistics.py) REUSED unchanged
        ├── TemporalMetadataBuilder   (temporal.py)   REUSED unchanged
        └── LabelManifestManager  (manifest.py)    REUSED unchanged
        │
        ▼
LabelDatasetResult               <-- identical public contract

<!-- LabelSource (abstract)                    (source.py)
    |
    +-- FilesystemLabelSource              <-- discovers masks on disk
    |   (future: SamAnnotationLabelSource, GisLabelSource, etc.
    |    plug in without changing LabelManager)
    v
LabelManager.generate()                    (manager.py)   <-- orchestrator only
    |
    +-- ClassSchema.from_config()          (schema.py)
    +-- SeasonResolver.from_config()       (temporal.py)
    +-- HydrologicalYearResolver.from_config() (temporal.py)
    +-- TemporalMetadataBuilder.build()    (temporal.py)
    +-- LabelSource.discover()             (source.py)
    +-- LabelValidator.validate()          (validator.py)
    +-- LabelStatisticsCalculator          (statistics.py)
    +-- LabelManifestManager               (manifest.py)
    |
    v
LabelDatasetResult                          <-- typed output contract -->

###############################################################################################
PatchDatasetResult + SceneMetadata (+ optional ClassificationContext)
        │
        ▼
LabelManager.generate()                              (manager.py — UNCHANGED)
        │
        ▼
PseudoLabelGenerator.generate(context=None)          (generator.py — updated)
        │
        ├── SpectralClassificationEngine
        │       .classify(patch_path, context=None)  (classifier.py — updated)
        │           │
        │           └── RuleRegistry                 (rules.py — NEW)
        │                   └── registered rules (discovered automatically)
        │                       WaterRule   (evidence-based sigmoid scoring)
        │                       VegetationRule  (evidence-based sigmoid scoring)
        │                       SandRule    (extended: +NDBI,NDMI,SAVI)
        │                       BackgroundRule  (fallback)
        │                       [Future: ShadowRule, MudRule, SnowRule ...]
        │
        ├── ConflictResolver.resolve()                (conflicts.py — UNCHANGED)
        │
        ├── MorphologyProcessor.process()             (morphology.py — updated)
        │       New order:
        │       Opening → Small-object removal → Closing
        │       → Hole filling → Boundary smoothing (channel-preserving)
        │
        ├── QualityAssessment.assess()                (quality.py — updated)
        │       Pluggable QualityMetric list:
        │       ValidPixelRatioMetric      (core, weight=0.50)
        │       UnclassifiedRatioMetric    (core, weight=0.30)
        │       ClassCoverageMetric        (core, weight=0.20)
        │       WaterContinuityMetric      (optional, river-specific)
        │       FragmentationMetric        (optional, river-specific)
        │       [Future metrics plug in without changing QualityAssessment]
        │
        ├── ConfidenceEstimator.estimate(             (confidence.py — updated)
        │       classification, class_map,
        │       quality_result=None)
        │       Pluggable ConfidenceComponent list:
        │       SpectralEvidenceComponent   (weight from config)
        │       RuleAgreementComponent      (weight from config)
        │       NeighborhoodConsistencyComponent (weight from config)
        │       QualityWeightComponent      (weight from config)
        │       [Future components plug in]
        │
        ├── ReproducibilityMetadata (generated)      (contracts.py — NEW class)
        │
        └── PseudoLabelResult (+ reproducibility)    (contracts.py — updated)

        ▼
LabelDatasetResult                                   (public contract — UNCHANGED)
## Module 10
DatasetAssembler              (assembler.py)    <-- orchestrator only
    |
    +-- DatasetValidator           (validator.py)   -- QC checks
    +-- DatasetSplitter            (splitter.py)    -- train/val/test split
    +-- DataLeakageDetector        (leakage.py)     -- leakage prevention
    +-- DatasetStatisticsCalculator (statistics.py) -- class distribution
    +-- DatasetQualityAnalyzer     (quality.py)     -- quality report
    +-- DatasetManifestManager     (manifest.py)    -- all file outputs
    +-- DatasetVersionManager      (version.py)     -- version.json
    |
    v
TrainingDatasetResult           <-- frozen typed output contract

# Module 11
DataLoaderFactory                 (dataloader.py)  <-- orchestrator only
    |
    +-- DatasetNormalizer              (normalizer.py)
    |       +-- compute(train_samples) -> NormalizationStats
    |       +-- apply(data)            -> normalized tensor
    |
    +-- AugmentationPipeline           (transforms.py)
    |       +-- build_train_transform() -> albumentations.Compose
    |       +-- build_eval_transform()  -> albumentations.Compose
    |
    +-- RiverMorphologyDataset         (dataset.py)
    |       +-- __getitem__() reads patch.tif + mask.tif via rasterio
    |       +-- returns (image_tensor, mask_tensor, SampleMetadata)
    |
    +-- TemporalSampler                (sampler.py)  (optional)
    |       +-- WeightedRandomSampler by season/year
    |
    v
DataLoaderBundle                       <-- typed output
    +-- train_loader  (DataLoader)
    +-- val_loader    (DataLoader)
    +-- test_loader   (DataLoader)
    +-- norm_stats    (NormalizationStats)
    +-- class_weights (ClassWeights)

# Module 12
