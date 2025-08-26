"""Device management utilities for GLIDE training.

Centralized CUDA/device handling with mixed precision support.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.device_utils")


@dataclass
class DeviceConfig:
    """Configuration for device management."""

    device_type: str = "cuda"  # "cuda", "cpu", "mps"
    device_id: int = 0  # GPU device ID for CUDA
    mixed_precision: bool = False  # Enable mixed precision
    memory_fraction: float = 0.95  # Fraction of GPU memory to use
    allow_tf32: bool = True  # Allow TF32 operations
    cudnn_benchmark: bool = True  # Enable cuDNN benchmark
    enable_cuda_graphs: bool = False  # Enable CUDA graphs (experimental)


class DeviceManager:
    """Manager for device operations and memory."""

    def __init__(self, config: DeviceConfig | None = None) -> None:
        """Initialize device manager.
        
        Args:
            config: Device configuration settings
        """
        self.config = config or DeviceConfig()
        self._device: torch.device | None = None
        self._setup_device()
        self._configure_backends()

    def _setup_device(self) -> None:
        """Setup the compute device."""
        if self.config.device_type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.config.device_type = "cpu"
            else:
                # Set memory fraction if specified
                if self.config.memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.config.memory_fraction,
                        device=self.config.device_id
                    )

                # Log GPU info
                device_name = torch.cuda.get_device_name(self.config.device_id)
                memory_gb = torch.cuda.get_device_properties(
                    self.config.device_id
                ).total_memory / 1024**3
                logger.info(f"Using GPU: {device_name} ({memory_gb:.1f} GB)")

        elif self.config.device_type == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                self.config.device_type = "cpu"
            else:
                logger.info("Using Apple Metal Performance Shaders (MPS)")

        # Create device object
        if self.config.device_type == "cuda":
            self._device = torch.device("cuda", self.config.device_id)
        else:
            self._device = torch.device(self.config.device_type)

        logger.info(f"Device initialized: {self._device}")

    def _configure_backends(self) -> None:
        """Configure backend settings for performance."""
        if self.config.device_type == "cuda":
            # Configure cuDNN
            torch.backends.cudnn.benchmark = self.config.cudnn_benchmark

            # Configure TF32
            torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32

            if self.config.allow_tf32:
                logger.info("TF32 operations enabled for better performance")

            # Enable CUDA graphs if requested (experimental)
            if self.config.enable_cuda_graphs:
                logger.info("CUDA graphs enabled (experimental)")

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            msg = "Device not initialized"
            raise RuntimeError(msg)
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self.config.device_type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if using MPS."""
        return self.config.device_type == "mps"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self.config.device_type == "cpu"

    def to_device(
        self,
        tensor_or_module: torch.Tensor | nn.Module,
        non_blocking: bool = False,
    ) -> torch.Tensor | nn.Module:
        """Move tensor or module to device.
        
        Args:
            tensor_or_module: Tensor or module to move
            non_blocking: Use non-blocking transfer
            
        Returns:
            Moved tensor or module
        """
        return tensor_or_module.to(self.device, non_blocking=non_blocking)

    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elif self.is_mps:
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Empty device memory cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()

        # Also run Python garbage collection
        gc.collect()

    def get_memory_stats(self) -> dict[str, float]:
        """Get device memory statistics.
        
        Returns:
            Dictionary with memory stats in GB
        """
        stats = {}

        if self.is_cuda:
            stats["allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1024**3
            stats["free_gb"] = (
                torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                - stats["allocated_gb"]
            )
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
        elif self.is_mps:
            # MPS doesn't provide detailed memory stats yet
            stats["device"] = "mps"
        else:
            stats["device"] = "cpu"

        return stats

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def set_device(self) -> None:
        """Set this as the current device."""
        if self.is_cuda:
            torch.cuda.set_device(self.device)

    def get_device_capability(self) -> tuple[int, int] | None:
        """Get CUDA compute capability.
        
        Returns:
            Tuple of (major, minor) compute capability or None
        """
        if self.is_cuda:
            return torch.cuda.get_device_capability(self.device)
        return None

    def supports_bfloat16(self) -> bool:
        """Check if device supports bfloat16.
        
        Returns:
            True if bfloat16 is supported
        """
        if self.is_cuda:
            capability = self.get_device_capability()
            if capability:
                # bfloat16 requires compute capability >= 8.0 (Ampere)
                return capability[0] >= 8
        elif self.is_mps:
            # MPS supports bfloat16 on M1 and later
            return True
        return False

    def get_amp_dtype(self) -> torch.dtype:
        """Get recommended dtype for automatic mixed precision.
        
        Returns:
            Recommended dtype (float16 or bfloat16)
        """
        if self.supports_bfloat16():
            return torch.bfloat16
        return torch.float16

    def create_amp_scaler(self) -> torch.cuda.amp.GradScaler | None:
        """Create gradient scaler for mixed precision.
        
        Returns:
            GradScaler for CUDA or None for other devices
        """
        if self.is_cuda and self.config.mixed_precision:
            # Only use GradScaler with float16, not bfloat16
            if self.get_amp_dtype() == torch.float16:
                return torch.cuda.amp.GradScaler()
        return None


def get_device(
    device_str: str | None = None,
    device_id: int = 0,
) -> torch.device:
    """Get a torch device from string specification.
    
    Args:
        device_str: Device string ("cuda", "cpu", "mps", "cuda:0", etc.)
        device_id: Default device ID if not specified in string
        
    Returns:
        torch.device object
    """
    if device_str is None:
        # Auto-select best available device
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    # Parse device string
    if device_str.startswith("cuda"):
        if ":" in device_str:
            # Device ID specified in string
            return torch.device(device_str)
        # Use provided device_id
        return torch.device("cuda", device_id)
    return torch.device(device_str)


def get_optimal_device() -> torch.device:
    """Get the optimal available device.
    
    Returns:
        Best available device (CUDA > MPS > CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(
    data: Any,
    device: str | torch.device,
    non_blocking: bool = False,
) -> Any:
    """Recursively move data to device.
    
    Handles tensors, modules, lists, tuples, and dicts.
    
    Args:
        data: Data to move
        device: Target device
        non_blocking: Use non-blocking transfer
        
    Returns:
        Data moved to device
    """
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(data, torch.Tensor | nn.Module):
        return data.to(device, non_blocking=non_blocking)
    if isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(item, device, non_blocking) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in data)
    return data


def log_device_info() -> None:
    """Log information about available devices."""
    logger.info("=== Device Information ===")

    # CUDA devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available: {num_gpus} GPU(s)")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            logger.info(
                f"  GPU {i}: {props.name} "
                f"({memory_gb:.1f} GB, "
                f"compute {props.major}.{props.minor})"
            )
    else:
        logger.info("CUDA not available")

    # MPS device
    if torch.backends.mps.is_available():
        logger.info("Apple MPS available")

    # CPU info
    logger.info(f"CPU threads: {torch.get_num_threads()}")


def configure_multi_gpu(
    device_ids: list[int] | None = None,
    find_unused_parameters: bool = False,
) -> tuple[torch.device, nn.Module | None]:
    """Configure multi-GPU training with DataParallel or DistributedDataParallel.
    
    Args:
        device_ids: List of GPU device IDs to use
        find_unused_parameters: For DDP, whether to find unused parameters
        
    Returns:
        Tuple of (primary device, optional DDP/DP wrapper class)
    """
    if not torch.cuda.is_available():
        logger.warning("Multi-GPU requested but CUDA not available")
        return torch.device("cpu"), None

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        logger.info("Single GPU available, multi-GPU not needed")
        return torch.device("cuda"), None

    if device_ids is None:
        device_ids = list(range(num_gpus))

    logger.info(f"Configuring multi-GPU with devices: {device_ids}")

    # Check if we're in a distributed setting
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed Data Parallel
        logger.info("Using DistributedDataParallel")
        from torch.nn.parallel import DistributedDataParallel as DDP

        rank = int(os.environ["RANK"])
        device = torch.device("cuda", device_ids[rank % len(device_ids)])

        return device, DDP
    # Data Parallel
    logger.info("Using DataParallel")
    from torch.nn import DataParallel as DP

    primary_device = torch.device("cuda", device_ids[0])
    return primary_device, DP


def benchmark_device(
    device: str | torch.device,
    size: int = 1024,
    iterations: int = 100,
) -> dict[str, float]:
    """Benchmark device performance.
    
    Args:
        device: Device to benchmark
        size: Matrix size for benchmark
        iterations: Number of iterations
        
    Returns:
        Benchmark results dictionary
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Create test matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Time matrix multiplication
    import time
    start = time.perf_counter()

    for _ in range(iterations):
        torch.matmul(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    # Calculate metrics
    flops = 2 * size**3 * iterations  # Matrix multiply FLOPs
    tflops = flops / elapsed / 1e12

    return {
        "device": str(device),
        "matrix_size": size,
        "iterations": iterations,
        "total_time_s": elapsed,
        "time_per_iter_ms": elapsed / iterations * 1000,
        "tflops": tflops,
    }


# Convenience functions
def cuda_is_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def mps_is_available() -> bool:
    """Check if MPS is available."""
    return torch.backends.mps.is_available()


def get_device_count() -> int:
    """Get number of available CUDA devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def set_cuda_device(device_id: int) -> None:
    """Set current CUDA device.
    
    Args:
        device_id: GPU device ID
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    else:
        logger.warning("Cannot set CUDA device - CUDA not available")
