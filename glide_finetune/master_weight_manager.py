#!/usr/bin/env python3
"""
Master Weight Manager for FP16 Training
Maintains FP32 master copies of weights for stable gradient accumulation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import OrderedDict
import gc


class MasterWeightManager:
    """
    Manages FP32 master weights for FP16 training.
    
    Key features:
    - Maintains FP32 copies of FP16 model parameters
    - Handles gradient accumulation in FP32
    - Synchronizes between FP16 and FP32 weights
    - Memory-efficient parameter grouping
    """
    
    def __init__(self, model: nn.Module, create_master_weights: bool = True):
        """
        Initialize master weight manager.
        
        Args:
            model: Model with potentially mixed precision parameters
            create_master_weights: Whether to create FP32 master weights
        """
        self.model = model
        self.create_master_weights = create_master_weights
        self.logger = logging.getLogger(__name__)
        
        # Master weight storage
        self.master_params: List[torch.Tensor] = []
        self.model_params: List[torch.nn.Parameter] = []
        self.param_groups: List[Dict[str, Any]] = []
        
        # Mapping between model and master params
        self.model_to_master: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self.master_to_model: Dict[torch.Tensor, torch.nn.Parameter] = {}
        
        # Statistics
        self.stats = {
            'fp16_params': 0,
            'fp32_params': 0,
            'master_params_created': 0,
            'memory_overhead_gb': 0.0,
        }
        
        # Initialize master weights
        if create_master_weights:
            self._create_master_weights()
    
    def _create_master_weights(self) -> None:
        """Create FP32 master copies of model parameters."""
        self.logger.info("Creating FP32 master weights...")
        
        # Clear existing
        self.master_params.clear()
        self.model_params.clear()
        self.model_to_master.clear()
        self.master_to_model.clear()
        
        memory_overhead = 0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            self.model_params.append(param)
            
            # Check if parameter needs master weight
            if param.dtype == torch.float16:
                # Create FP32 master copy
                master_param = param.detach().float()
                master_param.requires_grad = True
                
                self.master_params.append(master_param)
                self.model_to_master[param] = master_param
                self.master_to_model[master_param] = param
                
                self.stats['fp16_params'] += param.numel()
                self.stats['master_params_created'] += 1
                
                # Calculate memory overhead (FP32 copy of FP16 param)
                memory_overhead += param.numel() * 2  # 2 extra bytes per param
                
                self.logger.debug(f"Created master weight for {name} "
                                f"(shape={list(param.shape)}, dtype={param.dtype})")
            else:
                # Use original parameter as master
                self.master_params.append(param)
                self.model_to_master[param] = param
                self.master_to_model[param] = param
                
                self.stats['fp32_params'] += param.numel()
        
        self.stats['memory_overhead_gb'] = memory_overhead / 1e9
        
        self.logger.info(f"Created {self.stats['master_params_created']} master weights")
        self.logger.info(f"Memory overhead: {self.stats['memory_overhead_gb']:.2f} GB")
    
    def zero_grad(self) -> None:
        """Zero gradients on master parameters."""
        for param in self.master_params:
            if param.grad is not None:
                param.grad.zero_()
    
    def backward_sync(self, loss: torch.Tensor, loss_scale: float = 1.0) -> None:
        """
        Perform backward pass with gradient sync to master weights.
        
        Args:
            loss: Loss tensor (potentially scaled)
            loss_scale: Current loss scale for unscaling
        """
        # Backward pass on model
        loss.backward()
        
        # Sync gradients to master weights
        self.sync_gradients_to_master(loss_scale)
    
    def sync_gradients_to_master(self, loss_scale: float = 1.0) -> None:
        """
        Copy and accumulate gradients from model to master weights.
        
        Args:
            loss_scale: Scale factor to unscale gradients
        """
        inv_scale = 1.0 / loss_scale if loss_scale != 1.0 else 1.0
        
        for model_param, master_param in self.model_to_master.items():
            if model_param.grad is None:
                continue
            
            # Convert gradient to FP32 and unscale
            if model_param.dtype == torch.float16:
                if master_param.grad is None:
                    # First gradient - create and assign
                    master_param.grad = model_param.grad.float() * inv_scale
                else:
                    # Accumulate gradient
                    master_param.grad.add_(model_param.grad.float() * inv_scale)
                
                # Clear model gradient to save memory
                model_param.grad = None
            else:
                # FP32 parameter - just unscale
                if model_param.grad is not None:
                    model_param.grad.mul_(inv_scale)
    
    def sync_master_to_model(self) -> None:
        """Copy master weights back to model parameters."""
        with torch.no_grad():
            for model_param, master_param in self.model_to_master.items():
                if model_param.dtype == torch.float16 and master_param is not model_param:
                    # Convert FP32 master to FP16 model param
                    model_param.copy_(master_param.half())
    
    def clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        """
        Clip gradients by norm on master parameters.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (default: 2.0 for L2)
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(self.master_params, max_norm, norm_type)
    
    def get_param_groups(self, base_lr: float = 1e-4, 
                        weight_decay: float = 0.0) -> List[Dict[str, Any]]:
        """
        Create parameter groups for optimizer.
        
        Args:
            base_lr: Base learning rate
            weight_decay: Weight decay factor
            
        Returns:
            List of parameter groups for optimizer
        """
        if not self.param_groups:
            # Create default groups
            self.param_groups = [
                {
                    'params': self.master_params,
                    'lr': base_lr,
                    'weight_decay': weight_decay,
                }
            ]
        
        return self.param_groups
    
    def create_optimizer(self, optimizer_class: type,
                         **optimizer_kwargs) -> torch.optim.Optimizer:
        """
        Create optimizer with master weights.
        
        Args:
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            **optimizer_kwargs: Arguments for optimizer
            
        Returns:
            Initialized optimizer
        """
        param_groups = self.get_param_groups(
            base_lr=optimizer_kwargs.pop('lr', 1e-4),
            weight_decay=optimizer_kwargs.pop('weight_decay', 0.0)
        )
        
        return optimizer_class(param_groups, **optimizer_kwargs)
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step and sync weights.
        
        Args:
            optimizer: Optimizer with master parameters
        """
        # Step on master weights
        optimizer.step()
        
        # Sync master weights back to model
        self.sync_master_to_model()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'master_params': [p.cpu() for p in self.master_params],
            'stats': self.stats,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if 'master_params' in state_dict:
            for master, saved in zip(self.master_params, state_dict['master_params']):
                master.data.copy_(saved.to(master.device))
        
        if 'stats' in state_dict:
            self.stats.update(state_dict['stats'])
        
        # Sync to model after loading
        self.sync_master_to_model()
    
    def check_sync(self) -> Dict[str, float]:
        """
        Check synchronization between model and master weights.
        
        Returns:
            Dictionary with sync statistics
        """
        sync_stats = {
            'max_diff': 0.0,
            'mean_diff': 0.0,
            'mismatched_params': 0,
        }
        
        diffs = []
        
        with torch.no_grad():
            for model_param, master_param in self.model_to_master.items():
                if model_param.dtype == torch.float16 and master_param is not model_param:
                    # Compare FP16 model with FP32 master
                    diff = (model_param.float() - master_param).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    
                    diffs.append(mean_diff)
                    
                    if max_diff > 1e-3:  # Tolerance for FP16 conversion
                        sync_stats['mismatched_params'] += 1
                    
                    sync_stats['max_diff'] = max(sync_stats['max_diff'], max_diff)
        
        if diffs:
            sync_stats['mean_diff'] = sum(diffs) / len(diffs)
        
        return sync_stats
    
    def log_stats(self) -> None:
        """Log manager statistics."""
        total_params = self.stats['fp16_params'] + self.stats['fp32_params']
        fp16_pct = self.stats['fp16_params'] / total_params * 100 if total_params > 0 else 0
        
        self.logger.info(f"Master Weight Manager Stats:")
        self.logger.info(f"  FP16 params: {self.stats['fp16_params']:,} ({fp16_pct:.1f}%)")
        self.logger.info(f"  FP32 params: {self.stats['fp32_params']:,}")
        self.logger.info(f"  Master weights: {self.stats['master_params_created']}")
        self.logger.info(f"  Memory overhead: {self.stats['memory_overhead_gb']:.2f} GB")
        
        sync_stats = self.check_sync()
        self.logger.info(f"  Sync max diff: {sync_stats['max_diff']:.6f}")
        self.logger.info(f"  Sync mean diff: {sync_stats['mean_diff']:.6f}")


class GradientAccumulator:
    """
    Handles gradient accumulation for mixed precision training.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.logger = logging.getLogger(__name__)
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def accumulate(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient accumulation.
        
        Args:
            loss: Unscaled loss
            
        Returns:
            Scaled loss for accumulation
        """
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        
        self.current_step += 1
        return loss
    
    def reset(self) -> None:
        """Reset accumulation counter."""
        self.current_step = 0


def test_master_weights():
    """Test master weight manager."""
    print("Testing Master Weight Manager...")
    
    # Create mixed precision model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.LayerNorm(20),  # Keep in FP32
        nn.Linear(20, 10),
    )
    
    # Convert to mixed precision
    model[0].half()  # Linear to FP16
    model[2].half()  # Linear to FP16
    # LayerNorm stays FP32
    
    model = model.cuda()
    
    # Create manager
    manager = MasterWeightManager(model)
    manager.log_stats()
    
    # Create optimizer with master weights
    optimizer = manager.create_optimizer(torch.optim.Adam, lr=1e-3)
    
    # Test training step
    print("\nSimulating training step...")
    x = torch.randn(32, 10).cuda()
    
    # Forward pass
    y = model(x)
    loss = y.mean()
    
    # Backward with sync
    optimizer.zero_grad()
    manager.backward_sync(loss, loss_scale=256.0)
    
    # Check gradients
    print(f"Master param gradients created: "
          f"{sum(1 for p in manager.master_params if p.grad is not None)}")
    
    # Clip gradients
    grad_norm = manager.clip_grad_norm(max_norm=1.0)
    print(f"Gradient norm: {grad_norm:.4f}")
    
    # Optimizer step with sync
    manager.step(optimizer)
    
    # Check synchronization
    sync_stats = manager.check_sync()
    print(f"\nSynchronization check:")
    print(f"  Max difference: {sync_stats['max_diff']:.6f}")
    print(f"  Mean difference: {sync_stats['mean_diff']:.6f}")
    print(f"  Mismatched params: {sync_stats['mismatched_params']}")
    
    print("\n✅ Master Weight Manager test complete!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_master_weights()