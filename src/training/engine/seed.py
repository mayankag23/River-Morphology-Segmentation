"""
Deterministic seeding and CuDNN configuration for Module 14.

SeedManager seeds torch, numpy, and the Python random module from a single
integer, and optionally configures PyTorch's deterministic algorithm mode
and cuDNN autotuner. This guarantees reproducible training runs when the
same config is used.
"""

from __future__ import annotations

import logging
import random
import torch

import numpy as np

__all__ = ["SeedManager"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class SeedManager:
    """
    Seeds all RNGs required for deterministic training.

    Call SeedManager.seed(n) once before building DataLoaders and the model.
    Call SeedManager.seed_worker(worker_id) as the DataLoader worker_init_fn
    to ensure each worker uses a unique but deterministic seed.
    """

    @staticmethod
    def seed(
        seed:              int,
        deterministic:     bool = False,
        cudnn_benchmark:   bool = False,
    ) -> None:
        """
        Seed torch, numpy, and Python random.

        Args:
            seed:             Integer seed.
            deterministic:    Enable torch.use_deterministic_algorithms(True).
                              Slower but fully reproducible on GPU.
            cudnn_benchmark:  Enable cuDNN autotuner. Set False for
                              reproducibility, True for speed.
        """
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.use_deterministic_algorithms(True)
                _LOGGER.info("SeedManager: deterministic algorithms enabled.")
            try:
                torch.backends.cudnn.benchmark    = cudnn_benchmark
                torch.backends.cudnn.deterministic = deterministic
            except AttributeError:
                pass
        except ImportError:
            _LOGGER.warning("SeedManager: torch not available; only numpy/random seeded.")

        _LOGGER.debug(
            "SeedManager: seeded with %d (deterministic=%s, cudnn_benchmark=%s).",
            seed, deterministic, cudnn_benchmark,
        )

    @staticmethod
    def seed_worker(worker_id: int) -> None:
        """
        DataLoader worker_init_fn for per-worker deterministic seeding.

        Each worker receives a unique seed derived from the base seed so
        that workers produce different (but reproducible) augmentations.

        Args:
            worker_id: Integer worker index provided by DataLoader.
        """
        # worker_seed = (np.random.get_state()[1][0] + worker_id) % (2 ** 32)

        base_seed = int(np.random.get_state()[1][0])
        worker_seed = (base_seed + worker_id) % (2**32)

        # np.random.seed(worker_seed)
        # random.seed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    @staticmethod
    def get_rng_state() -> dict:
        """
        Capture current RNG state for checkpoint persistence.

        Returns:
            Dict with keys "python", "numpy", "torch" (and "torch_cuda" if CUDA).
        """
        state: dict = {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
        }
        try:
            import torch
            state["torch"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        return state

    @staticmethod
    def restore_rng_state(state: dict) -> None:
        """
        Restore RNG state from a checkpoint dict.

        Args:
            state: Dict as returned by get_rng_state().
        """
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        try:
            import torch
            if "torch" in state:
                torch.set_rng_state(state["torch"])
            if "torch_cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except ImportError:
            pass
