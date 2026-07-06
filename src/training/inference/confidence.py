"""
Confidence estimation strategies for Module 16.

ConfidenceStrategy is an abstract strategy that converts a probability map
(C, H, W) into a scalar confidence map (H, W). Two strategies are registered:

    max_probability  -- confidence = max probability across classes.
                        High when model is certain about one class.
                        Low when probability mass is spread across classes.

    entropy          -- confidence = 1 - normalised_entropy.
                        normalised_entropy = H(p) / log(C)
                        High when entropy is low (certain prediction).
                        Low when entropy is high (uncertain prediction).

Both return values in [0, 1] where 1 means fully confident.

Adding a future strategy (TTA confidence, ensemble disagreement, etc.):
    1. Subclass ConfidenceStrategy.
    2. Decorate with @ConfidenceRegistry.register("my_strategy").
    3. Set config.inference.confidence_strategy = "my_strategy".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = [
    "ConfidenceStrategy",
    "ConfidenceRegistry",
    "MaxProbabilityStrategy",
    "EntropyStrategy",
]

_EPS: float = 1e-10


# ==============================================================================
# Abstract interface
# ==============================================================================

class ConfidenceStrategy(ABC):
    """
    Abstract confidence estimation strategy.

    Accepts probability maps (C, H, W) float32, returns (H, W) float32.
    All values are guaranteed in [0, 1].
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry key for this strategy."""

    @abstractmethod
    def compute(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel confidence.

        Args:
            probabilities: (C, H, W) float32 probability map (rows sum to 1).

        Returns:
            (H, W) float32 confidence map in [0, 1].
        """


# ==============================================================================
# ConfidenceRegistry
# ==============================================================================

class ConfidenceRegistry:
    """Registry mapping strategy names to ConfidenceStrategy classes."""

    _registered: dict[str, type[ConfidenceStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator. Registers by name string."""
        def decorator(klass: type[ConfidenceStrategy]) -> type[ConfidenceStrategy]:
            cls._registered[name.lower()] = klass
            return klass
        return decorator

    @classmethod
    def build(cls, name: str) -> ConfidenceStrategy:
        """
        Instantiate the strategy registered under name.

        Raises:
            KeyError: name is not registered.
        """
        n = name.lower().strip()
        if n not in cls._registered:
            raise KeyError(
                f"ConfidenceRegistry: '{name}' is not registered. "
                f"Available: {sorted(cls._registered)}"
            )
        return cls._registered[n]()

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registered.keys()))


# ==============================================================================
# MaxProbabilityStrategy
# ==============================================================================

@ConfidenceRegistry.register("max_probability")
class MaxProbabilityStrategy(ConfidenceStrategy):
    """
    Confidence = maximum softmax probability across all classes.

    Interpretation: 1.0 means the model assigns 100% probability to a single
    class; values near 1/C indicate maximum uncertainty (uniform distribution).

    This is the fastest confidence estimator (single argmax over the class dim).
    It is the standard confidence metric in river morphology segmentation
    literature and is directly comparable across checkpoints.
    """

    @property
    def name(self) -> str:
        return "max_probability"

    def compute(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Args:
            probabilities: (C, H, W) float32 in [0, 1].
        Returns:
            (H, W) float32 max probability per pixel.
        """
        return probabilities.max(axis=0).astype(np.float32)


# ==============================================================================
# EntropyStrategy
# ==============================================================================

@ConfidenceRegistry.register("entropy")
class EntropyStrategy(ConfidenceStrategy):
    """
    Confidence = 1 - normalised Shannon entropy.

    normalised_entropy(p) = -sum(p * log(p)) / log(C)

    Normalisation by log(C) maps entropy to [0, 1] regardless of the number
    of classes. A value of 0 means maximum uncertainty; 1 means certain.

    Entropy-based confidence is more sensitive to multi-modal predictive
    distributions than max-probability (where two classes sharing 50% each
    would yield confidence=0.5 rather than 0.0).

    Future-ready: TTA confidence and ensemble disagreement metrics can be
    implemented as additional strategies without changing the public API.
    """

    @property
    def name(self) -> str:
        return "entropy"

    def compute(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Args:
            probabilities: (C, H, W) float32 in [0, 1].
        Returns:
            (H, W) float32 entropy-based confidence in [0, 1].
        """
        C = probabilities.shape[0]
        # Clip to avoid log(0).
        p        = np.clip(probabilities.astype(np.float64), _EPS, 1.0)
        entropy  = -(p * np.log(p)).sum(axis=0)
        max_ent  = np.log(C) if C > 1 else 1.0
        norm_ent = entropy / max_ent
        confidence = (1.0 - norm_ent).clip(0.0, 1.0)
        return confidence.astype(np.float32)
