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
