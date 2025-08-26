"""Seed control utilities for deterministic and performance modes."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("glide_finetune.seed_utils")


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    warn: bool = True,
) -> None:
    """Set seeds for reproducibility or performance.

    Args:
        seed: Random seed value. Use 0 for performance mode (non-deterministic).
        deterministic: Enable deterministic algorithms (slower but reproducible).
        benchmark: Enable cudnn benchmark mode (faster but non-deterministic).
        warn: Print warnings about determinism settings.
    """
    if seed == 0:
        # Performance mode - don't set seeds, enable optimizations
        if warn:
            logger.info("Running in performance mode (seed=0):")
            logger.info("  - Non-deterministic algorithms enabled")
            logger.info("  - TF32 operations enabled")
            logger.info("  - cuDNN benchmark enabled")

        # Enable performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Use different random seed each run
        actual_seed = random.randint(1, 2**31 - 1)  # noqa: S311 - Pseudorandom appropriate for ML seed generation
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)
    else:
        # Deterministic mode - set seeds and disable non-deterministic algorithms
        if warn:
            logger.info(f"Running in deterministic mode (seed={seed}):")
            logger.info("  - Deterministic algorithms enabled (slower)")
            logger.info("  - cuDNN benchmark disabled")
            if not deterministic:
                logger.info("  - Warning: deterministic=False but seed is set")

        # Set all seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Configure deterministic behavior
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

            # Set environment variable for some operations
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif benchmark:
            # Benchmark mode - faster but non-deterministic even with seed
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


def get_random_seed() -> int:
    """Generate a random seed value.

    Returns:
        Random seed between 1 and 2^31-1.
    """
    return random.randint(1, 2**31 - 1)  # noqa: S311 - Pseudorandom appropriate for ML seed generation


def worker_init_fn(worker_id: int, base_seed: int | None = None) -> None:
    """Initialize worker seed for DataLoader workers.

    This ensures each DataLoader worker has a different seed for
    proper randomization in multi-worker scenarios.

    Args:
        worker_id: Worker ID from DataLoader.
        base_seed: Base seed to derive worker seeds from.
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % (2**31)

    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_generator(seed: int | None = None) -> torch.Generator:
    """Create a torch Generator with optional seed.

    Args:
        seed: Optional seed for the generator.

    Returns:
        Configured torch.Generator instance.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


class SeedManager:
    """Manager for controlling seeds across different components."""

    def __init__(
        self,
        base_seed: int = 0,
        deterministic: bool = True,
        benchmark: bool = False,
    ):
        """Initialize seed manager.

        Args:
            base_seed: Base seed value (0 for performance mode).
            deterministic: Enable deterministic algorithms.
            benchmark: Enable cudnn benchmark mode.
        """
        self.base_seed = base_seed
        self.deterministic = deterministic
        self.benchmark = benchmark
        self.is_performance_mode = (base_seed == 0)

        # Set initial seed
        set_seed(base_seed, deterministic, benchmark)

    def get_component_seed(self, component: str) -> int:
        """Get a deterministic seed for a specific component.

        Args:
            component: Component name (e.g., 'dataloader', 'augmentation').

        Returns:
            Seed value for the component.
        """
        if self.is_performance_mode:
            return get_random_seed()

        # Use hash to generate component-specific seed
        import hashlib
        hash_obj = hashlib.md5(f"{self.base_seed}_{component}".encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        return hash_int % (2**31)

    def get_epoch_seed(self, epoch: int) -> int:
        """Get seed for a specific epoch.

        Args:
            epoch: Epoch number.

        Returns:
            Seed value for the epoch.
        """
        if self.is_performance_mode:
            return get_random_seed()

        return self.base_seed + epoch

    def get_generator(self, component: str | None = None) -> torch.Generator:
        """Get a torch Generator for a component.

        Args:
            component: Optional component name.

        Returns:
            Configured torch.Generator instance.
        """
        if component is None or self.is_performance_mode:
            return get_generator()

        seed = self.get_component_seed(component)
        return get_generator(seed)

    def reset(self) -> None:
        """Reset to base seed configuration."""
        set_seed(self.base_seed, self.deterministic, self.benchmark, warn=False)


def validate_seed(seed: int) -> int:
    """Validate and normalize seed value.

    Args:
        seed: Seed value to validate.

    Returns:
        Valid seed value.

    Raises:
        ValueError: If seed is invalid.
    """
    if not isinstance(seed, int | np.integer):
        msg = f"Seed must be an integer, got {type(seed)}"
        raise ValueError(msg)

    if seed < 0:
        msg = f"Seed must be non-negative, got {seed}"
        raise ValueError(msg)

    if seed >= 2**31:
        msg = f"Seed must be less than 2^31, got {seed}"
        raise ValueError(msg)

    return int(seed)


def log_seed_info() -> None:
    """Log current seed and determinism settings."""
    import logging
    logger = logging.getLogger("glide_finetune")

    logger.info("Seed Configuration:")
    logger.info(f"  Random seed: {random.getstate()[1][0]}")
    logger.info(f"  NumPy seed: {np.random.get_state()[1][0]}")
    logger.info(f"  PyTorch seed: {torch.initial_seed()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"  CUDA benchmark: {torch.backends.cudnn.benchmark}")
        logger.info(f"  TF32 allowed: {torch.backends.cuda.matmul.allow_tf32}")
