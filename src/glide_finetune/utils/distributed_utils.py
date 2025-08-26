"""Utilities for distributed training with HuggingFace Accelerate.

Clean, type-safe integration with Accelerate for distributed training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import (
    DistributedType,
    wait_for_everyone,
)
from torch import nn

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.distributed_utils")


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: str = "nccl"  # "nccl", "gloo", "mps"
    mixed_precision: str = "no"  # "no", "fp16", "bf16", "fp8"
    gradient_accumulation_steps: int = 1
    cpu_offload: bool = False
    split_batches: bool = False
    dispatch_batches: bool = False
    even_batches: bool = True
    use_seedable_sampler: bool = True
    log_with: str | None = None  # "wandb", "tensorboard", etc.
    project_dir: str = "."


class DistributedManager:
    """Manager for distributed training with Accelerate."""

    def __init__(
        self,
        config: DistributedConfig | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Initialize distributed manager.
        
        Args:
            config: Distributed training configuration
            accelerator: Existing Accelerator instance (if None, creates new one)
        """
        self.config = config or DistributedConfig()

        if accelerator is None:
            self.accelerator = Accelerator(
                mixed_precision=self.config.mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                cpu=self.config.cpu_offload,
                split_batches=self.config.split_batches,
                dispatch_batches=self.config.dispatch_batches,
                even_batches=self.config.even_batches,
                use_seedable_sampler=self.config.use_seedable_sampler,
                log_with=self.config.log_with,
                project_dir=self.config.project_dir,
            )
        else:
            self.accelerator = accelerator

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup distributed logging."""
        if self.is_main_process:
            logger.info("Distributed training initialized:")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Distributed type: {self.distributed_type}")
            logger.info(f"  Mixed precision: {self.mixed_precision}")
            logger.info(f"  Num processes: {self.num_processes}")
            logger.info(f"  Process index: {self.process_index}")

    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self.accelerator.device

    @property
    def distributed_type(self) -> DistributedType:
        """Get distributed training type."""
        return self.accelerator.distributed_type

    @property
    def num_processes(self) -> int:
        """Get number of processes."""
        return self.accelerator.num_processes

    @property
    def process_index(self) -> int:
        """Get current process index."""
        return self.accelerator.process_index

    @property
    def local_process_index(self) -> int:
        """Get local process index (within node)."""
        return self.accelerator.local_process_index

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self) -> bool:
        """Check if this is the local main process (within node)."""
        return self.accelerator.is_local_main_process

    @property
    def mixed_precision(self) -> str:
        """Get mixed precision type."""
        return self.accelerator.mixed_precision

    def prepare(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, torch.optim.lr_scheduler._LRScheduler | None]:
        """Prepare model, optimizer, and dataloader for distributed training.
        
        Args:
            model: Model to prepare
            optimizer: Optimizer to prepare
            dataloader: DataLoader to prepare
            scheduler: Optional learning rate scheduler
            
        Returns:
            Prepared (model, optimizer, dataloader, scheduler)
        """
        if scheduler is not None:
            model, optimizer, dataloader, scheduler = self.accelerator.prepare(
                model, optimizer, dataloader, scheduler
            )
            return model, optimizer, dataloader, scheduler
        model, optimizer, dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
        return model, optimizer, dataloader, None

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient accumulation.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        self.accelerator.backward(loss)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor from all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            Gathered tensor
        """
        return self.accelerator.gather(tensor)

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor for metrics computation.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            Gathered tensor (duplicates removed)
        """
        return self.accelerator.gather_for_metrics(tensor)

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce tensor across all processes.
        
        Args:
            tensor: Tensor to reduce
            reduction: Type of reduction ("mean", "sum")
            
        Returns:
            Reduced tensor
        """
        return self.accelerator.reduce(tensor, reduction)

    def pad_across_processes(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        pad_index: int = 0,
        pad_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad tensor to have same shape across processes.
        
        Args:
            tensor: Tensor to pad
            dim: Dimension to pad along
            pad_index: Value to use for padding
            pad_first: Whether to pad at beginning or end
            
        Returns:
            Tuple of (padded tensor, mask tensor indicating real values)
        """
        return self.accelerator.pad_across_processes(
            tensor, dim=dim, pad_index=pad_index, pad_first=pad_first
        )

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap model from any distributed wrapper.
        
        Args:
            model: Possibly wrapped model
            
        Returns:
            Unwrapped model
        """
        return self.accelerator.unwrap_model(model)

    def save_model(
        self,
        model: nn.Module,
        save_directory: str | Path,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
    ) -> None:
        """Save model with proper distributed handling.
        
        Args:
            model: Model to save
            save_directory: Directory to save to
            max_shard_size: Maximum shard size for large models
            safe_serialization: Use safetensors format
        """
        self.accelerator.save_model(
            model,
            save_directory,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
        )

    def save_state(self, output_dir: str | Path) -> None:
        """Save complete training state.
        
        Args:
            output_dir: Directory to save state to
        """
        self.accelerator.save_state(output_dir)

    def load_state(
        self,
        input_dir: str | Path,
        strict: bool = True,
    ) -> None:
        """Load complete training state.
        
        Args:
            input_dir: Directory to load state from
            strict: Whether to enforce strict state dict loading
        """
        self.accelerator.load_state(input_dir, strict=strict)

    def wait_for_everyone(self) -> None:
        """Wait for all processes to reach this point."""
        self.accelerator.wait_for_everyone()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only from main process."""
        self.accelerator.print(*args, **kwargs)

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        """Log values (only from main process).
        
        Args:
            values: Dictionary of values to log
            step: Optional step number
        """
        self.accelerator.log(values, step=step)

    def init_trackers(
        self,
        project_name: str,
        config: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tracking libraries.
        
        Args:
            project_name: Name of the project
            config: Configuration to log
            init_kwargs: Additional kwargs for tracker initialization
        """
        self.accelerator.init_trackers(
            project_name,
            config=config,
            init_kwargs=init_kwargs,
        )

    def end_training(self) -> None:
        """End training and clean up trackers."""
        self.accelerator.end_training()

    def get_tracker(self, name: str) -> Any:
        """Get a specific tracker.
        
        Args:
            name: Name of the tracker (e.g., "wandb", "tensorboard")
            
        Returns:
            Tracker instance
        """
        return self.accelerator.get_tracker(name)

    def on_main_process(self, function: callable) -> Any:
        """Execute function only on main process.
        
        Args:
            function: Function to execute
            
        Returns:
            Function result on main process, None on others
        """
        if self.is_main_process:
            return function()
        return None

    def on_local_main_process(self, function: callable) -> Any:
        """Execute function only on local main process.
        
        Args:
            function: Function to execute
            
        Returns:
            Function result on local main process, None on others
        """
        if self.is_local_main_process:
            return function()
        return None


# Standalone utility functions (backward compatibility)

def is_distributed() -> bool:
    """Check if we're running in distributed mode."""
    state = AcceleratorState()
    return state.distributed_type != DistributedType.NO


def get_world_size() -> int:
    """Get number of processes in distributed training."""
    if is_distributed() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process."""
    state = AcceleratorState()
    return state.is_main_process


def is_local_main_process() -> bool:
    """Check if this is the local main process."""
    state = AcceleratorState()
    return state.is_local_main_process


def wait_for_everyone() -> None:
    """Wait for all processes to reach this point."""
    if is_distributed():
        wait_for_everyone()


def print_once(*args: Any, **kwargs: Any) -> None:
    """Print only from the main process."""
    if is_main_process():
        logger.info(*args, **kwargs)


def gather_tensor(
    tensor: torch.Tensor,
    accelerator: Accelerator | None = None,
) -> torch.Tensor:
    """Gather tensor from all processes.
    
    Args:
        tensor: Tensor to gather
        accelerator: Optional Accelerator instance
        
    Returns:
        Gathered tensor (concatenated along batch dimension)
    """
    if accelerator is not None:
        return accelerator.gather(tensor)

    if not is_distributed() or not dist.is_initialized():
        return tensor

    # Manual gathering if no accelerator
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def reduce_tensor(
    tensor: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        reduction: Type of reduction ("mean", "sum", "max", "min")
        
    Returns:
        Reduced tensor
    """
    if not is_distributed() or not dist.is_initialized():
        return tensor

    # Clone to avoid modifying input
    tensor = tensor.clone()

    if reduction == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    elif reduction == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif reduction == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif reduction == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        msg = f"Unknown reduction: {reduction}"
        raise ValueError(msg)

    return tensor


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed() and dist.is_initialized():
        dist.barrier()


class DistributedMetricsTracker:
    """Metrics tracker that handles distributed training."""

    def __init__(
        self,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Initialize metrics tracker.
        
        Args:
            accelerator: Optional Accelerator instance
        """
        self.accelerator = accelerator
        self.metrics: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, key: str, value: float, count: int = 1) -> None:
        """Update a metric.
        
        Args:
            key: Metric key
            value: Metric value
            count: Number of samples
        """
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0

        self.metrics[key] += value * count
        self.counts[key] += count

    def get_average(self, key: str) -> float:
        """Get average value of a metric.
        
        Args:
            key: Metric key
            
        Returns:
            Average value
        """
        if key not in self.metrics or self.counts[key] == 0:
            return 0.0

        return self.metrics[key] / self.counts[key]

    def get_global_average(self, key: str) -> float:
        """Get average value across all processes.
        
        Args:
            key: Metric key
            
        Returns:
            Global average value
        """
        if not is_distributed():
            return self.get_average(key)

        # Determine the correct device to use
        if self.accelerator is not None:
            device = self.accelerator.device
        elif torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")

        # Gather metrics from all processes
        total_value = torch.tensor(
            self.metrics.get(key, 0.0),
            device=device,
            dtype=torch.float32,
        )
        total_count = torch.tensor(
            self.counts.get(key, 0),
            device=device,
            dtype=torch.float32,
        )

        if self.accelerator is not None:
            total_value = self.accelerator.reduce(total_value, "sum")
            total_count = self.accelerator.reduce(total_count, "sum")
        else:
            total_value = reduce_tensor(total_value, "sum")
            total_count = reduce_tensor(total_count, "sum")

        if total_count.item() == 0:
            return 0.0

        return (total_value / total_count).item()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def get_all_averages(self) -> dict[str, float]:
        """Get all average metrics.
        
        Returns:
            Dictionary of average metrics
        """
        return {key: self.get_average(key) for key in self.metrics}

    def get_all_global_averages(self) -> dict[str, float]:
        """Get all average metrics across all processes.
        
        Returns:
            Dictionary of global average metrics
        """
        return {key: self.get_global_average(key) for key in self.metrics}

    def log_metrics(
        self,
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to accelerator.
        
        Args:
            step: Optional step number
            prefix: Prefix for metric names
        """
        if self.accelerator is not None and self.accelerator.is_main_process:
            metrics = self.get_all_global_averages()
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            self.accelerator.log(metrics, step=step)


def setup_distributed_seed(seed: int) -> int:
    """Setup seeds for distributed training.
    
    Each process gets a different seed based on its rank.
    
    Args:
        seed: Base seed value
        
    Returns:
        Base seed used
    """
    import random

    if seed == 0:
        seed = random.randint(1, 2**31 - 1)  # noqa: S311 - Pseudorandom appropriate for ML seed generation

    # Each process gets a unique seed
    rank = get_rank()
    process_seed = seed + rank

    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)

    # Print seed info
    print_once(f"Base seed: {seed}, Process seeds: {seed} to {seed + get_world_size() - 1}")

    return seed


class DistributedSampler:
    """Simple distributed sampler for datasets.
    
    Ensures each process gets different samples.
    """

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize distributed sampler.
        
        Args:
            dataset_len: Length of dataset
            batch_size: Batch size per process
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling
        """
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for proper shuffling.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch

    def get_indices(self) -> list[int]:
        """Get indices for current process with balanced distribution.
        
        Returns:
            List of indices for this process
        """
        # Create indices
        indices = list(range(self.dataset_len))

        # Shuffle if needed
        if self.shuffle:
            # Use epoch as part of seed for different shuffling each epoch
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=rng).tolist()

        # Ensure balanced distribution across processes
        # Pad indices to make them evenly divisible
        total_size = ((len(indices) + self.world_size - 1) // self.world_size) * self.world_size
        indices += indices[: (total_size - len(indices))]  # Repeat some indices if needed

        # Now distribute evenly
        indices_per_process = len(indices) // self.world_size
        start_idx = self.rank * indices_per_process
        end_idx = start_idx + indices_per_process

        return indices[start_idx:end_idx]

    def __len__(self) -> int:
        """Get number of samples for this process.
        
        Returns:
            Number of samples
        """
        return len(self.get_indices())


def save_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_path: str | Path,
    accelerator: Accelerator | None = None,
    additional_state: dict[str, Any] | None = None,
) -> None:
    """Save checkpoint in distributed training.
    
    Only saves from main process unless using FSDP.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current step
        checkpoint_path: Path to save checkpoint
        accelerator: Optional Accelerator instance
        additional_state: Additional state to save
    """
    checkpoint_path = Path(checkpoint_path)

    if accelerator is not None:
        # Use Accelerate's save method
        accelerator.save_state(checkpoint_path)

        # Save additional metadata
        if accelerator.is_main_process:
            metadata = {
                "epoch": epoch,
                "step": step,
                **(additional_state or {}),
            }
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    elif is_main_process():
        # Manual save from main process only
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            **(additional_state or {}),
        }

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Wait for save to complete
    barrier()


def load_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: str | Path,
    accelerator: Accelerator | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load checkpoint in distributed training.
    
    Args:
        model: Model to load into
        optimizer: Optional optimizer to load into
        checkpoint_path: Path to checkpoint
        accelerator: Optional Accelerator instance
        strict: Whether to enforce strict state dict loading
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if accelerator is not None:
        # Use Accelerate's load method
        accelerator.load_state(checkpoint_path, strict=strict)

        # Try to load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {"epoch": 0, "step": 0}
    # Manual load
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
    }


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast an object from source rank to all other ranks.
    
    Args:
        obj: Object to broadcast (only needed on source rank)
        src: Source rank
        
    Returns:
        The broadcasted object on all ranks
    """
    if not is_distributed() or not dist.is_initialized():
        return obj

    # Use broadcast_object_list for complex objects
    objects = [obj] if get_rank() == src else [None]

    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(obj: Any) -> list[Any]:
    """Gather objects from all processes.
    
    Args:
        obj: Object from this process
        
    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered
