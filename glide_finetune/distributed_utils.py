"""Utilities for distributed training with Accelerate."""

import os
from typing import Optional, Any, Dict, List
import torch as th
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object


def is_distributed() -> bool:
    """Check if we're running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get number of processes in distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_rank() == 0


def print_once(*args, **kwargs):
    """Print only from the main process."""
    if is_main_process():
        print(*args, **kwargs)


def gather_tensor(tensor: th.Tensor, accelerator: Optional[Accelerator] = None) -> th.Tensor:
    """
    Gather tensor from all processes.
    
    Args:
        tensor: Tensor to gather
        accelerator: Optional Accelerator instance
        
    Returns:
        Gathered tensor (concatenated along batch dimension)
    """
    if accelerator is not None:
        return accelerator.gather(tensor)
    
    if not is_distributed():
        return tensor
    
    # Manual gathering if no accelerator
    world_size = get_world_size()
    gathered = [th.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return th.cat(gathered, dim=0)


def reduce_tensor(tensor: th.Tensor, reduction: str = "mean") -> th.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        reduction: Type of reduction ("mean", "sum", "max", "min")
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
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
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return tensor


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


class DistributedMetricsTracker:
    """Metrics tracker that handles distributed training."""
    
    def __init__(self, accelerator: Optional[Accelerator] = None):
        self.accelerator = accelerator
        self.metrics = {}
        self.counts = {}
    
    def update(self, key: str, value: float, count: int = 1):
        """Update a metric."""
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0
        
        self.metrics[key] += value * count
        self.counts[key] += count
    
    def get_average(self, key: str) -> float:
        """Get average value of a metric."""
        if key not in self.metrics:
            return 0.0
        
        if self.counts[key] == 0:
            return 0.0
        
        return self.metrics[key] / self.counts[key]
    
    def get_global_average(self, key: str) -> float:
        """Get average value across all processes."""
        if not is_distributed():
            return self.get_average(key)
        
        # Determine the correct device to use
        if self.accelerator is not None:
            device = self.accelerator.device
        elif th.cuda.is_available():
            device = th.device("cuda", th.cuda.current_device())
        else:
            device = th.device("cpu")
        
        # Gather metrics from all processes - use proper device placement
        total_value = th.as_tensor(self.metrics.get(key, 0.0), device=device, dtype=th.float32)
        total_count = th.as_tensor(self.counts.get(key, 0), device=device, dtype=th.float32)
        
        total_value = reduce_tensor(total_value, "sum")
        total_count = reduce_tensor(total_count, "sum")
        
        if total_count.item() == 0:
            return 0.0
        
        return (total_value / total_count).item()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all average metrics."""
        return {key: self.get_average(key) for key in self.metrics}
    
    def get_all_global_averages(self) -> Dict[str, float]:
        """Get all average metrics across all processes."""
        return {key: self.get_global_average(key) for key in self.metrics}


def setup_distributed_seed(seed: int):
    """
    Setup seeds for distributed training.
    Each process gets a different seed based on its rank.
    """
    import random
    import numpy as np
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Each process gets a unique seed
    rank = get_rank()
    process_seed = seed + rank
    
    random.seed(process_seed)
    np.random.seed(process_seed)
    th.manual_seed(process_seed)
    th.cuda.manual_seed(process_seed)
    th.cuda.manual_seed_all(process_seed)
    
    # Print seed info
    print_once(f"Base seed: {seed}, Process seeds: {seed} to {seed + get_world_size() - 1}")
    
    return seed


def distributed_cleanup():
    """
    Clean up distributed training.
    
    WARNING: Only call this for raw torch.distributed training.
    Do NOT call this when using Accelerate, as it manages its own cleanup.
    """
    if is_distributed():
        dist.destroy_process_group()


class DistributedSampler:
    """
    Simple distributed sampler for datasets.
    Ensures each process gets different samples.
    """
    
    def __init__(self, dataset_len: int, batch_size: int, shuffle: bool = True):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling."""
        self.epoch = epoch
    
    def get_indices(self) -> List[int]:
        """Get indices for current process with balanced distribution."""
        # Create indices
        indices = list(range(self.dataset_len))
        
        # Shuffle if needed
        if self.shuffle:
            # Use epoch as part of seed for different shuffling each epoch
            rng = th.Generator()
            rng.manual_seed(42 + self.epoch)  # Fixed base seed + epoch for reproducibility
            indices = th.randperm(len(indices), generator=rng).tolist()
        
        # Ensure balanced distribution across processes
        # Pad indices to make them evenly divisible
        total_size = ((len(indices) + self.world_size - 1) // self.world_size) * self.world_size
        indices += indices[:(total_size - len(indices))]  # Repeat some indices if needed
        
        # Now distribute evenly
        indices_per_process = len(indices) // self.world_size
        start_idx = self.rank * indices_per_process
        end_idx = start_idx + indices_per_process
        
        return indices[start_idx:end_idx]


def save_distributed_checkpoint(
    model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_path: str,
    accelerator: Optional[Accelerator] = None,
):
    """
    Save checkpoint in distributed training.
    Only saves from main process unless using FSDP.
    """
    if accelerator is not None:
        # Use Accelerate's save method
        accelerator.save_state(checkpoint_path)
    elif is_main_process():
        # Manual save from main process only
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        th.save(state, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Wait for save to complete
    barrier()


def load_distributed_checkpoint(
    model: th.nn.Module,
    optimizer: Optional[th.optim.Optimizer],
    checkpoint_path: str,
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint in distributed training.
    
    Returns:
        Dictionary with epoch and step information
    """
    if accelerator is not None:
        # Use Accelerate's load method
        accelerator.load_state(checkpoint_path)
        # Try to load metadata
        import json
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"epoch": 0, "step": 0}
    else:
        # Manual load
        checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        return {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", 0),
        }


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast an object from source rank to all other ranks.
    
    Args:
        obj: Object to broadcast (only needed on source rank)
        src: Source rank
        
    Returns:
        The broadcasted object on all ranks
    """
    if not is_distributed():
        return obj
    
    # Use gather_object for complex objects
    if get_rank() == src:
        objects = [obj]
    else:
        objects = [None]
    
    dist.broadcast_object_list(objects, src=src)
    return objects[0]