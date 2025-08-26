"""
Memory management utilities for GLIDE fine-tuning with VRAM constraints.
Handles safe loading/unloading of models on GPU with comprehensive memory tracking.
"""

import gc
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil
import torch as th

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("glide_finetune.memory_manager")


class GPUMemoryMonitor:
    """Monitor and track GPU memory usage."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.initial_memory = None
        self.peak_memory = 0
        self.memory_history = []

    def get_memory_info(self) -> dict[str, float]:
        """Get current GPU memory information in MB."""
        if not th.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

        allocated = th.cuda.memory_allocated(self.device) / 1024**2
        reserved = th.cuda.memory_reserved(self.device) / 1024**2
        total = th.cuda.get_device_properties(self.device).total_memory / 1024**2
        free = total - reserved

        return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}

    def reset_peak_memory_stats(self):
        """Reset peak memory tracking."""
        th.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0

    def log_memory_snapshot(self, label: str = ""):
        """Log current memory state with optional label."""
        info = self.get_memory_info()
        peak_allocated = th.cuda.max_memory_allocated(self.device) / 1024**2
        peak_reserved = th.cuda.max_memory_reserved(self.device) / 1024**2

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "allocated": info["allocated"],
            "reserved": info["reserved"],
            "free": info["free"],
            "total": info["total"],
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "memory_usage_pct": (info["reserved"] / info["total"]) * 100,
        }

        self.memory_history.append(snapshot)

        logger.info(
            f"🔍 GPU Memory [{label}]: "
            f"Allocated: {info['allocated']:.1f}MB | "
            f"Reserved: {info['reserved']:.1f}MB | "
            f"Free: {info['free']:.1f}MB | "
            f"Usage: {snapshot['memory_usage_pct']:.1f}%"
        )

        return snapshot

    def check_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available."""
        info = self.get_memory_info()
        available = info["free"]
        return available >= required_mb

    def estimate_model_memory(self, model: th.nn.Module) -> float:
        """Estimate memory usage of a model in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024**2


class ModelMemoryManager:
    """Manages loading and unloading of models with memory constraints."""

    def __init__(self, device: str = "cuda", max_memory_usage_pct: float = 90.0):
        self.device = device
        self.max_memory_usage_pct = max_memory_usage_pct
        self.monitor = GPUMemoryMonitor(device)
        self.loaded_models = {}
        self.model_checksums = {}

    def clear_gpu_cache(self):
        """Aggressively clear GPU memory cache."""
        if th.cuda.is_available():
            th.cuda.empty_cache()
            th.cuda.synchronize()
        gc.collect()

    def safe_delete_model(self, model: th.nn.Module, model_name: str = "model"):
        """Safely delete a model and clear its memory."""
        logger.info(f"🗑️  Unloading {model_name}...")

        # Move to CPU first to free GPU memory
        if hasattr(model, "cpu"):
            model.cpu()

        # Clear any cached computations
        if hasattr(model, "del_cache"):
            model.del_cache()

        # Remove from tracked models
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

        # Delete the model
        del model

        # Force garbage collection and cache clearing
        self.clear_gpu_cache()

        self.monitor.log_memory_snapshot(f"After unloading {model_name}")

    def load_model_safely(self, loader_func, model_name: str, *args, **kwargs) -> tuple[Any, ...]:
        """
        Load a model safely with memory checking.

        Args:
            loader_func: Function to load the model
            model_name: Name for tracking
            *args, **kwargs: Arguments for loader_func

        Returns:
            Tuple of loaded model components
        """
        logger.info(f"📥 Loading {model_name}...")

        # Check initial memory
        self.monitor.log_memory_snapshot(f"Before loading {model_name}")

        # Load model
        model_components = loader_func(*args, **kwargs)

        # Move to device if needed
        if isinstance(model_components, tuple | list):
            model = model_components[0]  # Assume first component is the model
        else:
            model = model_components

        if hasattr(model, "to"):
            model.to(self.device)

        # Track the loaded model
        self.loaded_models[model_name] = {
            "model": model,
            "components": model_components,
            "load_time": time.time(),
        }

        # Check final memory
        final_snapshot = self.monitor.log_memory_snapshot(f"After loading {model_name}")

        # Verify we haven't exceeded memory limits
        if final_snapshot["memory_usage_pct"] > self.max_memory_usage_pct:
            logger.info(
                f"⚠️  Warning: Memory usage ({final_snapshot['memory_usage_pct']:.1f}%) "
                f"exceeds limit ({self.max_memory_usage_pct}%)"
            )

        return model_components

    def get_loaded_models(self) -> dict[str, dict]:
        """Get information about currently loaded models."""
        return self.loaded_models.copy()

    def memory_report(self) -> str:
        """Generate a comprehensive memory usage report."""
        info = self.monitor.get_memory_info()

        report = [
            "=" * 60,
            "🔍 GPU MEMORY REPORT",
            "=" * 60,
            f"Device: {self.device}",
            f"Total Memory: {info['total']:.1f} MB",
            f"Allocated: {info['allocated']:.1f} MB ({info['allocated'] / info['total'] * 100:.1f}%)",
            f"Reserved: {info['reserved']:.1f} MB ({info['reserved'] / info['total'] * 100:.1f}%)",
            f"Free: {info['free']:.1f} MB ({info['free'] / info['total'] * 100:.1f}%)",
            "",
            f"Loaded Models: {len(self.loaded_models)}",
        ]

        for model_name, model_info in self.loaded_models.items():
            load_time = time.time() - model_info["load_time"]
            report.append(f"  - {model_name}: loaded {load_time:.1f}s ago")

        if len(self.monitor.memory_history) > 0:
            report.extend(
                [
                    "",
                    "Recent Memory Snapshots:",
                ]
            )
            for snapshot in self.monitor.memory_history[-5:]:  # Last 5 snapshots
                report.append(f"  {snapshot['label']}: {snapshot['memory_usage_pct']:.1f}% usage")

        report.append("=" * 60)
        return "\n".join(report)


@contextmanager
def temporary_model_load(
    memory_manager: ModelMemoryManager, loader_func, model_name: str, *args, **kwargs
):
    """
    Context manager for temporarily loading a model.
    Automatically unloads when exiting context.
    """
    model_components = None
    try:
        model_components = memory_manager.load_model_safely(
            loader_func, model_name, *args, **kwargs
        )
        yield model_components
    finally:
        if model_components is not None:
            if isinstance(model_components, tuple | list):
                model = model_components[0]
            else:
                model = model_components
            memory_manager.safe_delete_model(model, model_name)


class ModelStateManager:
    """Manages saving and restoring model states for memory-efficient evaluation."""

    def __init__(self, temp_dir: str = "/tmp/glide_eval_states"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.saved_states = {}

    def save_model_state(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer | None,
        state_name: str,
        include_optimizer: bool = True,
    ) -> str:
        """
        Save model and optimizer state to disk.

        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            state_name: Unique name for this state
            include_optimizer: Whether to save optimizer state

        Returns:
            Path to saved state file
        """
        state_path = self.temp_dir / f"{state_name}.pt"

        logger.info(f"💾 Saving model state: {state_name}")

        state_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "save_time": time.time(),
        }

        if include_optimizer and optimizer is not None:
            state_dict["optimizer_state_dict"] = optimizer.state_dict()
            state_dict["optimizer_class"] = optimizer.__class__.__name__

        th.save(state_dict, state_path)

        self.saved_states[state_name] = {
            "path": str(state_path),
            "save_time": time.time(),
            "size_mb": state_path.stat().st_size / 1024**2,
        }

        logger.info(f"✅ Saved {state_name} ({self.saved_states[state_name]['size_mb']:.1f} MB)")
        return str(state_path)

    def restore_model_state(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer | None,
        state_name: str,
        device: str = "cuda",
    ) -> bool:
        """
        Restore model and optimizer state from disk.

        Args:
            model: Model to restore into
            optimizer: Optimizer to restore into (optional)
            state_name: Name of saved state
            device: Device to load tensors to

        Returns:
            True if successful, False otherwise
        """
        if state_name not in self.saved_states:
            logger.info(f"❌ State {state_name} not found")
            return False

        state_path = self.saved_states[state_name]["path"]

        logger.info(f"📂 Restoring model state: {state_name}")

        try:
            checkpoint = th.load(state_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info(f"✅ Restored {state_name}")
            return True

        except Exception as e:
            logger.info(f"❌ Failed to restore {state_name}: {e}")
            return False

    def cleanup_states(self, keep_latest: int = 1):
        """Clean up old saved states, keeping only the most recent."""
        if len(self.saved_states) <= keep_latest:
            return

        # Sort by save time
        sorted_states = sorted(
            self.saved_states.items(), key=lambda x: x[1]["save_time"], reverse=True
        )

        # Remove old states
        for state_name, info in sorted_states[keep_latest:]:
            try:
                os.remove(info["path"])
                del self.saved_states[state_name]
                logger.info(f"🗑️  Cleaned up old state: {state_name}")
            except Exception as e:
                logger.info(f"⚠️  Failed to cleanup {state_name}: {e}")

    def get_state_info(self) -> dict[str, dict]:
        """Get information about saved states."""
        return self.saved_states.copy()


def get_system_memory_info() -> dict[str, float]:
    """Get system memory information."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / 1024**3,
        "available_gb": memory.available / 1024**3,
        "used_gb": memory.used / 1024**3,
        "usage_pct": memory.percent,
    }
