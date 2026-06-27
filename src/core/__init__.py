"""
Core infrastructure package for the River Morphology Segmentation System.

Provides:
    Config          — Master configuration manager with validation
    ConfigNode      — Dot-notation namespace for config values
    exceptions      — Project-specific exception hierarchy
    environment     — Runtime environment validation utilities
"""

from src.core.config import Config, ConfigNode
from src.core.exceptions import (
    ConfigurationError,
    EnvironmentValidationError,
    GEECredentialError,
    GEEDownloadError,
    InferenceError,
    InvalidValueError,
    MissingFieldError,
    PipelineError,
    PreprocessingError,
    RiverMorphologyError,
    TypeMismatchError,
)

__all__ = [
    "Config",
    "ConfigNode",
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