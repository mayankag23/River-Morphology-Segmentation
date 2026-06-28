Module 3:

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

