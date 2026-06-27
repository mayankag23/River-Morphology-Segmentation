"""
Custom exception hierarchy for the River Morphology Segmentation System.

All project exceptions inherit from RiverMorphologyError, which inherits
from the built-in Exception. This allows callers to catch either the
specific exception type for fine-grained handling, or RiverMorphologyError
to catch any project-related failure in a single except clause.

Hierarchy:

    RiverMorphologyError
    ├── ConfigurationError
    │   ├── MissingFieldError
    │   ├── InvalidValueError
    │   └── TypeMismatchError
    ├── EnvironmentValidationError
    ├── GEECredentialError
    ├── GEEDownloadError
    └── PipelineError
        ├── PreprocessingError
        └── InferenceError

Usage:

    from src.core.exceptions import InvalidValueError, MissingFieldError

    if value < 0:
        raise InvalidValueError(
            field="optimizer.learning_rate",
            value=value,
            reason="must be a positive number",
        )

    # Catch any project exception:
    try:
        config = Config("config/config.yaml")
    except RiverMorphologyError as exc:
        logger.error("Project error: %s", exc)
"""

from __future__ import annotations

__all__ = [
    "RiverMorphologyError",
    "ConfigurationError",
    "MissingFieldError",
    "InvalidValueError",
    "TypeMismatchError",
    "EnvironmentValidationError",
    "GEECredentialError",
    "GEEDownloadError",
    "PipelineError",
    "PreprocessingError",
    "InferenceError",
]


# ==============================================================================
# Base Exception
# ==============================================================================

class RiverMorphologyError(Exception):
    """
    Base class for all project-specific exceptions.

    Catching this class catches any error raised intentionally by the
    river morphology system (as opposed to unexpected Python errors).
    """


# ==============================================================================
# Configuration Exceptions
# ==============================================================================

class ConfigurationError(RiverMorphologyError):
    """
    Raised when the configuration is invalid, incomplete, or inconsistent.

    Use the more specific subclasses where possible to allow callers to
    distinguish between missing fields, wrong values, and wrong types.
    """


class MissingFieldError(ConfigurationError):
    """
    Raised when a required configuration field is absent or null when a
    non-null value is expected.

    Args:
        field:   Dot-notation path to the missing field, e.g. "aoi.min_lon".
        context: Optional additional context explaining what the field is used
                 for and how to set it.

    Example:
        raise MissingFieldError(
            field="aoi.min_lon",
            context="Set all four AOI coordinates in config.yaml before running GEE download.",
        )
    """

    def __init__(self, field: str, context: str = "") -> None:
        self.field = field
        self.context = context
        message = f"Required configuration field is missing or null: '{field}'"
        if context:
            message = f"{message}\n{context}"
        super().__init__(message)


class InvalidValueError(ConfigurationError):
    """
    Raised when a configuration field is present but has a semantically
    invalid value (e.g., negative learning rate, coordinates out of range).

    Args:
        field:   Dot-notation path to the invalid field.
        value:   The invalid value that was found.
        reason:  Human-readable explanation of why the value is invalid.

    Example:
        raise InvalidValueError(
            field="aoi.min_lon",
            value=190.0,
            reason="longitude must be in the range [-180, 180]",
        )
    """

    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid value for '{field}': {value!r}\nReason: {reason}"
        )


class TypeMismatchError(ConfigurationError):
    """
    Raised when a configuration field has the correct key but an unexpected
    Python type (e.g., a string where an integer is required).

    Args:
        field:         Dot-notation path to the field with the wrong type.
        expected_type: Human-readable description of the expected type.
        actual_type:   The Python type that was actually found.

    Example:
        raise TypeMismatchError(
            field="training.batch_size",
            expected_type="int",
            actual_type=type(value).__name__,
        )
    """

    def __init__(
        self, field: str, expected_type: str, actual_type: str
    ) -> None:
        self.field = field
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"Type mismatch for '{field}': "
            f"expected {expected_type}, got {actual_type}"
        )


# ==============================================================================
# Environment Exceptions
# ==============================================================================

class EnvironmentValidationError(RiverMorphologyError):
    """
    Raised when the runtime environment fails a validation check.

    Used by src/core/environment.py when strict=True and a required
    dependency is missing, incompatible, or incorrectly configured.

    Args:
        check:   Name of the check that failed, e.g. "python_version".
        details: Human-readable explanation and remediation instructions.

    Example:
        raise EnvironmentValidationError(
            check="python_version",
            details="Python 3.11+ required. Found: 3.9.7.\nUpgrade Python.",
        )
    """

    def __init__(self, check: str, details: str) -> None:
        self.check = check
        self.details = details
        super().__init__(
            f"Environment validation failed [{check}]\n{details}"
        )


# ==============================================================================
# GEE Exceptions
# ==============================================================================

class GEECredentialError(RiverMorphologyError):
    """
    Raised when Google Earth Engine credentials are missing, invalid, or
    cannot be resolved from the expected environment variables.

    Environment variables expected:
        GEE_PROJECT_ID          — required
        GEE_SERVICE_ACCOUNT_KEY — optional (path to service account JSON)

    Example:
        raise GEECredentialError(
            "GEE_PROJECT_ID environment variable is not set.\n"
            "Add it to your .env file: GEE_PROJECT_ID=your-project-id"
        )
    """


class GEEDownloadError(RiverMorphologyError):
    """
    Raised when a Google Earth Engine image download fails.

    Args:
        url_or_asset: The GEE asset path or download URL that failed.
        reason:       Human-readable error detail from the GEE response.

    Example:
        raise GEEDownloadError(
            url_or_asset="LANDSAT/LC08/C02/T1_L2/LC08_145040_20231101",
            reason="Computation timed out after 300 seconds.",
        )
    """

    def __init__(self, url_or_asset: str, reason: str) -> None:
        self.url_or_asset = url_or_asset
        self.reason = reason
        super().__init__(
            f"GEE download failed for '{url_or_asset}'\nReason: {reason}"
        )


# ==============================================================================
# Pipeline Exceptions
# ==============================================================================

class PipelineError(RiverMorphologyError):
    """
    Raised when a pipeline stage fails for a reason not covered by a more
    specific exception subclass.

    Use PreprocessingError or InferenceError where applicable.
    """


class PreprocessingError(PipelineError):
    """
    Raised during data preprocessing failures — index computation,
    normalization, patch generation, or label generation.
    """


class InferenceError(PipelineError):
    """
    Raised during model inference failures — prediction, tile merging,
    post-processing, or export.
    """