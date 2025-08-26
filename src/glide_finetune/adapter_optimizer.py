"""
Adapter-only optimizer setup for CLIP adapter training.

This module provides specialized optimizer configuration for training only the
CLIP adapter parameters while keeping the base GLIDE model frozen. Follows
best practices from ControlNet/LoRA fine-tuning.

Key features:
- Separate parameter groups with different learning rates
- Gate parameter gets higher LR for faster adaptation
- Weight decay only on linear weights (not biases/norms)
- Automatic exclusion of frozen base model parameters
- AMP-aware gradient clipping
- DDP-compatible design
- Validation utilities to ensure correct setup
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterOptimizerConfig:
    """Configuration for adapter-only optimizer."""
    
    # Learning rates
    adapter_lr: float = 1e-4  # Main adapter layers
    gate_lr: float = 5e-4  # Gate gets higher LR to "wake up" faster
    
    # Weight decay
    weight_decay: float = 0.01  # Only for linear weights
    
    # AdamW parameters
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Performance options
    foreach: bool = False  # Use foreach implementation if available
    fused: bool = False  # Use fused kernels if available
    capturable: bool = False  # For CUDA graphs compatibility
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    error_if_nonfinite: bool = True  # Catch NaN/inf early
    
    # Scheduler
    scheduler_type: str = "linear"  # "linear", "cosine", "constant"
    warmup_steps: int = 2000
    
    # Validation
    strict_validation: bool = True  # Ensure only adapter params are trainable


def split_adapter_params_by_type(adapter_module: nn.Module) -> tuple[list, list, list]:
    """
    Split adapter parameters by type rather than name heuristics.
    
    More robust than string matching - handles different norm types,
    custom modules, and future extensions.
    
    Args:
        adapter_module: The adapter module to extract params from
    
    Returns:
        Tuple of (weight_params, bias_norm_params, gate_params)
    """
    weight_params = []
    bias_norm_params = []
    gate_params = []
    
    # Define norm layer types (extend as needed)
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
    
    # Gate is a direct parameter, not in a submodule
    if hasattr(adapter_module, 'gate'):
        gate_params.append(adapter_module.gate)
    
    # Iterate through submodules to categorize params
    for module_name, module in adapter_module.named_modules():
        if module_name == '':  # Skip root
            continue
            
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
                
            # Check if this is a norm layer or bias
            if isinstance(module, norm_types) or param_name == "bias":
                bias_norm_params.append(param)
            else:
                # Weight parameters (Linear.weight, Conv.weight, etc)
                weight_params.append(param)
    
    return weight_params, bias_norm_params, gate_params


def create_adapter_optimizer(
    model: nn.Module,
    config: AdapterOptimizerConfig | None = None,
    total_training_steps: int | None = None,
) -> tuple[Optimizer, Any | None]:
    """
    Create an optimizer that trains only the CLIP adapter parameters.
    
    This follows the ControlNet/LoRA pattern of freezing the base model
    and training only the added adapter components.
    
    Args:
        model: The model with integrated CLIP adapter
        config: Optimizer configuration (uses defaults if None)
        total_training_steps: Total number of training steps for scheduler
    
    Returns:
        Tuple of (optimizer, scheduler) where scheduler may be None
    
    Raises:
        ValueError: If no adapter found or validation fails
    """
    if config is None:
        config = AdapterOptimizerConfig()
    
    # Get the adapter module directly (more robust than string matching)
    adapter = getattr(model, "clip_adapter", None)
    if adapter is None:
        raise ValueError("No clip_adapter attribute found on model. Is the adapter integrated?")
    
    # Split parameters by type (not by name)
    weight_params, bias_norm_params, gate_params = split_adapter_params_by_type(adapter)
    
    # Count parameters
    counts = {
        "adapter_weights": sum(p.numel() for p in weight_params),
        "adapter_bias_norm": sum(p.numel() for p in bias_norm_params),
        "gate": sum(p.numel() for p in gate_params),
    }
    
    # Also count frozen params in entire model
    counts["frozen"] = sum(
        p.numel() for p in model.parameters() 
        if not p.requires_grad
    )
    
    # Validate we found parameters
    total_trainable = counts["adapter_weights"] + counts["adapter_bias_norm"] + counts["gate"]
    if total_trainable == 0:
        raise ValueError("No adapter parameters found! Check adapter integration and freezing.")
    
    # Validate in strict mode - ensure ONLY adapter params are trainable
    if config.strict_validation:
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        adapter_trainable = weight_params + bias_norm_params + gate_params
        
        # Convert to sets for comparison
        all_trainable_set = set(all_trainable)
        adapter_trainable_set = set(adapter_trainable)
        
        if all_trainable_set != adapter_trainable_set:
            extra_params = all_trainable_set - adapter_trainable_set
            raise ValueError(
                f"Found {len(extra_params)} non-adapter trainable params. "
                "In strict mode, only adapter should be trainable. "
                "Call freeze_base_model() first."
            )
    
    # Log parameter distribution
    logger.info("=" * 60)
    logger.info("Adapter-Only Optimizer Setup")
    logger.info("=" * 60)
    logger.info(f"Adapter weights: {counts['adapter_weights']:,} params (LR={config.adapter_lr}, WD={config.weight_decay})")
    logger.info(f"Adapter bias/norm: {counts['adapter_bias_norm']:,} params (LR={config.adapter_lr}, WD=0.0)")
    logger.info(f"Gate parameter: {counts['gate']:,} params (LR={config.gate_lr}, WD=0.0)")
    logger.info(f"Frozen (base model): {counts['frozen']:,} params")
    logger.info(f"Total trainable: {total_trainable:,} params ({total_trainable / 1e6:.2f}M)")
    logger.info("=" * 60)
    
    # Build optimizer with parameter groups
    param_groups = []
    
    if weight_params:
        param_groups.append({
            "params": weight_params,
            "lr": config.adapter_lr,
            "weight_decay": config.weight_decay,
            "name": "adapter_weights"
        })
    
    if bias_norm_params:
        param_groups.append({
            "params": bias_norm_params,
            "lr": config.adapter_lr,
            "weight_decay": 0.0,
            "name": "adapter_bias_norm"
        })
    
    if gate_params:
        param_groups.append({
            "params": gate_params,
            "lr": config.gate_lr,
            "weight_decay": 0.0,
            "name": "gate"
        })
    
    # Build optimizer kwargs
    optimizer_kwargs = {
        "betas": config.betas,
        "eps": config.eps,
    }
    
    # Add performance flags if available (PyTorch 2.0+)
    if config.foreach and hasattr(AdamW, "__init__"):
        # Check if foreach is supported
        import inspect
        sig = inspect.signature(AdamW.__init__)
        if "foreach" in sig.parameters:
            optimizer_kwargs["foreach"] = True
            logger.info("Using foreach implementation for AdamW")
    
    if config.fused:
        try:
            # Fused is only available with CUDA
            if torch.cuda.is_available():
                optimizer_kwargs["fused"] = True
                logger.info("Using fused AdamW kernels")
        except Exception:
            pass
    
    if config.capturable:
        optimizer_kwargs["capturable"] = config.capturable
    
    # Create AdamW optimizer
    optimizer = AdamW(param_groups, **optimizer_kwargs)
    
    # Create scheduler if requested
    scheduler = None
    if total_training_steps and config.scheduler_type != "constant":
        try:
            from transformers import get_scheduler
            
            scheduler = get_scheduler(
                config.scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=total_training_steps,
            )
            logger.info(f"Created {config.scheduler_type} scheduler with {config.warmup_steps} warmup steps")
        except ImportError:
            logger.warning("transformers not available, skipping scheduler creation")
    
    return optimizer, scheduler


def amp_safe_gradient_clip(
    model: nn.Module,
    config: AdapterOptimizerConfig,
    scaler: Any | None = None,
    optimizer: Optimizer | None = None,
    only_adapter: bool = True,
) -> float:
    """
    Apply gradient clipping with AMP awareness.
    
    CRITICAL: Must unscale gradients before clipping when using AMP!
    Call this after backward() and before optimizer.step().
    
    Args:
        model: The model with gradients
        config: Optimizer configuration with gradient_clip_norm
        scaler: GradScaler for AMP (if using mixed precision)
        optimizer: Optimizer (required if scaler is provided)
        only_adapter: If True, only clip adapter gradients
    
    Returns:
        The total gradient norm before clipping (0 if no params to clip)
    """
    if config.gradient_clip_norm <= 0:
        return 0.0
    
    # Unscale gradients if using AMP (critical!)
    if scaler is not None and optimizer is not None:
        scaler.unscale_(optimizer)
    
    # Collect parameters to clip
    if only_adapter:
        # Get adapter module directly
        adapter = getattr(model, "clip_adapter", None)
        if adapter is None:
            return 0.0
        
        params_to_clip = [
            p for p in adapter.parameters() 
            if p.requires_grad and p.grad is not None
        ]
    else:
        params_to_clip = [
            p for p in model.parameters() 
            if p.requires_grad and p.grad is not None
        ]
    
    # Guard against empty list
    if not params_to_clip:
        return 0.0
    
    # Clip with optional NaN/inf checking
    clip_kwargs = {"max_norm": config.gradient_clip_norm}
    
    # Check PyTorch version for error_if_nonfinite support
    import torch
    if config.error_if_nonfinite and hasattr(torch, "__version__"):
        # error_if_nonfinite added in PyTorch 1.9
        try:
            # Test if the parameter is accepted
            import inspect
            sig = inspect.signature(torch.nn.utils.clip_grad_norm_)
            if "error_if_nonfinite" in sig.parameters:
                clip_kwargs["error_if_nonfinite"] = True
        except Exception:
            pass
    
    total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, **clip_kwargs)
    
    return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


def freeze_base_model(model: nn.Module, skip_eval_mode: bool = True) -> dict[str, int]:
    """
    Freeze all non-adapter parameters in the model.
    
    This ensures only the CLIP adapter is trainable, following the
    ControlNet/LoRA pattern of adapter-only training.
    
    Args:
        model: The model to freeze
        skip_eval_mode: If True, don't change train/eval mode (let caller control)
    
    Returns:
        Dictionary with counts of frozen and trainable parameters
    """
    counts = {"frozen": 0, "trainable": 0, "adapter": 0}
    
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
        counts["frozen"] += param.numel()
    
    # Then unfreeze adapter if it exists
    adapter = getattr(model, "clip_adapter", None)
    if adapter is not None:
        for param in adapter.parameters():
            param.requires_grad = True
            counts["adapter"] += param.numel()
            counts["trainable"] += param.numel()
        
        # Adjust frozen count
        counts["frozen"] -= counts["adapter"]
    
    # Note: We DON'T set modules to eval mode by default
    # This avoids changing dropout/norm behavior which could
    # shift the loss distribution. Let the caller control train/eval.
    if not skip_eval_mode:
        logger.warning(
            "Setting non-adapter modules to eval mode. "
            "This may change model behavior (dropout, batchnorm). "
            "Consider skip_eval_mode=True."
        )
        for name, module in model.named_modules():
            if "clip_adapter" not in name:
                module.eval()
    
    logger.info(
        f"Froze base model: {counts['frozen']:,} params frozen, "
        f"{counts['adapter']:,} adapter params trainable"
    )
    
    return counts


def validate_adapter_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
) -> None:
    """
    Validate that the optimizer is correctly set up for adapter-only training.
    
    Checks:
    1. All optimizer params have requires_grad=True
    2. All optimizer params are from clip_adapter
    3. No base model params are in optimizer
    4. Gate parameter has higher learning rate (if exists)
    
    Args:
        optimizer: The optimizer to validate
        model: The model being optimized
    
    Raises:
        AssertionError: If validation fails
    """
    # Get adapter module
    adapter = getattr(model, "clip_adapter", None)
    assert adapter is not None, "Model has no clip_adapter attribute"
    
    # Collect all params in optimizer
    optimizer_params = set()
    gate_lr = None
    adapter_lr = None
    
    for group in optimizer.param_groups:
        for param in group["params"]:
            optimizer_params.add(param)
            
            # Track learning rates by group name
            if "name" in group:
                if group["name"] == "gate":
                    gate_lr = group["lr"]
                elif group["name"] in ["adapter_weights", "adapter_bias_norm"]:
                    adapter_lr = group["lr"]
    
    # Check 1: All optimizer params have requires_grad=True
    for param in optimizer_params:
        assert param.requires_grad, "Optimizer contains frozen parameter"
    
    # Check 2 & 3: All optimizer params are adapter params
    adapter_params = set(p for p in adapter.parameters() if p.requires_grad)
    
    assert optimizer_params == adapter_params, (
        f"Optimizer params mismatch. "
        f"Expected {len(adapter_params)} adapter params, "
        f"got {len(optimizer_params)} optimizer params"
    )
    
    # Check no base model params are trainable
    for name, param in model.named_parameters():
        if param.requires_grad and "clip_adapter" not in name:
            raise AssertionError(
                f"Base model param {name} is trainable. "
                "Only adapter should be trainable."
            )
    
    # Check 4: Gate has higher or equal LR (if both exist)
    if gate_lr is not None and adapter_lr is not None:
        assert gate_lr >= adapter_lr, (
            f"Gate LR ({gate_lr}) should be >= adapter LR ({adapter_lr})"
        )
    
    logger.info("✓ Adapter optimizer validation passed")


def get_adapter_param_norm(model: nn.Module) -> float:
    """
    Calculate the L2 norm of adapter parameters.
    
    Useful for monitoring adapter weight magnitudes during training.
    
    Args:
        model: Model with adapter
    
    Returns:
        L2 norm of all adapter parameters
    """
    adapter = getattr(model, "clip_adapter", None)
    if adapter is None:
        return 0.0
    
    total_norm = 0.0
    for param in adapter.parameters():
        if param.requires_grad:
            total_norm += param.data.norm(2).item() ** 2
    
    return total_norm ** 0.5


def log_adapter_stats(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
) -> dict[str, float]:
    """
    Log statistics about adapter training.
    
    Args:
        model: Model with adapter
        optimizer: Adapter optimizer
        step: Current training step
    
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Get adapter module
    adapter = getattr(model, "clip_adapter", None)
    if adapter is None:
        return stats
    
    # Get parameter norms
    stats["adapter_param_norm"] = get_adapter_param_norm(model)
    
    # Get gate value (safely handle scalar or vector gates)
    if hasattr(adapter, 'gate'):
        with torch.no_grad():
            gate_param = adapter.gate
            if gate_param.numel() == 1:
                # Scalar gate
                gate_value = torch.sigmoid(gate_param).item()
                stats["gate_value"] = gate_value
                stats["gate_logit"] = gate_param.item()
            else:
                # Vector gate (future extension)
                gate_values = torch.sigmoid(gate_param)
                stats["gate_value"] = gate_values.mean().item()
                stats["gate_value_std"] = gate_values.std().item()
                stats["gate_logit"] = gate_param.mean().item()
    
    # Get learning rates from optimizer
    for group in optimizer.param_groups:
        if "name" in group:
            stats[f"lr_{group['name']}"] = group["lr"]
        else:
            # Fallback if no name
            stats["lr"] = group["lr"]
            break
    
    # Log every 100 steps
    if step % 100 == 0:
        logger.info(
            f"[Step {step}] "
            f"Gate: {stats.get('gate_value', 0):.4f}, "
            f"Adapter norm: {stats['adapter_param_norm']:.3f}, "
            f"LR: {stats.get('lr_adapter_weights', stats.get('lr', 0)):.2e}"
        )
    
    return stats