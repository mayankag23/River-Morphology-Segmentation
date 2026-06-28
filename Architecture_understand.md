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
