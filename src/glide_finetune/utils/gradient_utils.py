"""Gradient clipping and processing utilities for GLIDE training.

Centralized gradient clipping logic with support for different strategies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from glide_finetune.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger("glide_finetune.gradient_utils")


@dataclass
class GradientStats:
    """Statistics about gradients."""

    total_norm: float
    max_norm: float
    min_norm: float
    num_parameters: int
    num_clipped: int
    clip_coef: float
    has_nan: bool
    has_inf: bool


class GradientClipper:
    """Manages gradient clipping with multiple strategies."""

    def __init__(
        self,
        clip_value: float = 1.0,
        clip_type: str = "norm",
        norm_type: float = 2.0,
        error_if_nonfinite: bool = True,
    ) -> None:
        """Initialize gradient clipper.
        
        Args:
            clip_value: Maximum allowed value for gradients
            clip_type: Type of clipping ("norm", "value", "adaptive", "agc")
            norm_type: Norm type for gradient clipping (e.g., 2 for L2)
            error_if_nonfinite: Raise error on NaN/Inf gradients
        """
        self.clip_value = clip_value
        self.clip_type = clip_type
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite

        # For adaptive clipping
        self.gradient_history: deque[float] = deque(maxlen=100)

        # For AGC (Adaptive Gradient Clipping)
        self.agc_eps = 1e-3

    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GradientStats:
        """Clip gradients of a model.
        
        Args:
            model: Model whose gradients to clip
            optimizer: Optional optimizer (for parameter groups)
            
        Returns:
            Gradient statistics
        """
        if self.clip_type == "none" or self.clip_value <= 0:
            return self._compute_stats(model)

        if self.clip_type == "norm":
            return self._clip_by_norm(model)
        if self.clip_type == "value":
            return self._clip_by_value(model)
        if self.clip_type == "adaptive":
            return self._clip_adaptive(model)
        if self.clip_type == "agc":
            return self._clip_agc(model, optimizer)
        msg = f"Unknown clip type: {self.clip_type}"
        raise ValueError(msg)

    def _clip_by_norm(self, model: nn.Module) -> GradientStats:
        """Clip gradients by global norm.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Gradient statistics
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return GradientStats(0, 0, 0, 0, 0, 1.0, False, False)

        # Compute total norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.clip_value,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
        )

        # Check for NaN/Inf
        has_nan = torch.isnan(total_norm).item()
        has_inf = torch.isinf(total_norm).item()

        # Compute clipping coefficient
        clip_coef = float(self.clip_value / (total_norm + 1e-6))
        clip_coef = min(clip_coef, 1.0)

        # Count clipped parameters
        num_clipped = 1 if clip_coef < 1.0 else 0

        return GradientStats(
            total_norm=float(total_norm),
            max_norm=float(total_norm),
            min_norm=float(total_norm),
            num_parameters=len(parameters),
            num_clipped=num_clipped,
            clip_coef=clip_coef,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def _clip_by_value(self, model: nn.Module) -> GradientStats:
        """Clip gradients by value.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Gradient statistics
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return GradientStats(0, 0, 0, 0, 0, 1.0, False, False)

        # Clip each gradient by value
        torch.nn.utils.clip_grad_value_(parameters, self.clip_value)

        # Compute statistics
        total_norm = 0.0
        max_norm = 0.0
        min_norm = float("inf")
        num_clipped = 0
        has_nan = False
        has_inf = False

        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type).item()
            total_norm += param_norm ** self.norm_type
            max_norm = max(max_norm, param_norm)
            min_norm = min(min_norm, param_norm)

            # Check if clipped
            if (p.grad.data.abs() > self.clip_value).any():
                num_clipped += 1

            # Check for NaN/Inf
            if torch.isnan(p.grad.data).any():
                has_nan = True
            if torch.isinf(p.grad.data).any():
                has_inf = True

        total_norm = total_norm ** (1.0 / self.norm_type)

        return GradientStats(
            total_norm=total_norm,
            max_norm=max_norm,
            min_norm=min_norm,
            num_parameters=len(parameters),
            num_clipped=num_clipped,
            clip_coef=1.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def _clip_adaptive(self, model: nn.Module) -> GradientStats:
        """Adaptive gradient clipping based on gradient history.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Gradient statistics
        """
        # First compute current gradient norm
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return GradientStats(0, 0, 0, 0, 0, 1.0, False, False)

        # Compute total norm
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type)
            total_norm += param_norm ** self.norm_type
        total_norm = float((total_norm ** (1.0 / self.norm_type)).item())

        # Update history
        self.gradient_history.append(total_norm)

        # Compute adaptive threshold
        if len(self.gradient_history) < 10:
            # Not enough history, use fixed threshold
            adaptive_threshold = self.clip_value
        else:
            # Use percentile of recent gradients
            import numpy as np
            percentile = 90  # Use 90th percentile
            adaptive_threshold = float(np.percentile(list(self.gradient_history), percentile))
            adaptive_threshold = max(adaptive_threshold, self.clip_value)

        # Clip using adaptive threshold
        actual_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            adaptive_threshold,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
        )

        # Compute statistics
        clip_coef = float(adaptive_threshold / (actual_norm + 1e-6))
        clip_coef = min(clip_coef, 1.0)

        has_nan = torch.isnan(actual_norm).item()
        has_inf = torch.isinf(actual_norm).item()

        return GradientStats(
            total_norm=float(actual_norm),
            max_norm=float(actual_norm),
            min_norm=float(actual_norm),
            num_parameters=len(parameters),
            num_clipped=1 if clip_coef < 1.0 else 0,
            clip_coef=clip_coef,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def _clip_agc(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> GradientStats:
        """Adaptive Gradient Clipping (AGC) from NFNets.
        
        Clips gradients based on the ratio of gradient norm to parameter norm.
        
        Args:
            model: Model whose gradients to clip
            optimizer: Optional optimizer for parameter groups
            
        Returns:
            Gradient statistics
        """
        if optimizer is None:
            param_groups = [{"params": model.parameters()}]
        else:
            param_groups = optimizer.param_groups

        total_norm = 0.0
        num_clipped = 0
        num_parameters = 0
        has_nan = False
        has_inf = False

        for group in param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                num_parameters += 1

                # Compute parameter and gradient norms
                param_norm = p.data.norm(self.norm_type).item()
                grad_norm = p.grad.data.norm(self.norm_type).item()
                total_norm += grad_norm ** self.norm_type

                # Check for NaN/Inf
                if torch.isnan(p.grad.data).any():
                    has_nan = True
                if torch.isinf(p.grad.data).any():
                    has_inf = True

                # AGC clipping
                if param_norm > self.agc_eps:
                    max_norm = param_norm * self.clip_value

                    if grad_norm > max_norm:
                        # Clip this parameter's gradient
                        p.grad.data.mul_(max_norm / grad_norm)
                        num_clipped += 1

        total_norm = total_norm ** (1.0 / self.norm_type)

        return GradientStats(
            total_norm=total_norm,
            max_norm=total_norm,
            min_norm=total_norm,
            num_parameters=num_parameters,
            num_clipped=num_clipped,
            clip_coef=1.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def _compute_stats(self, model: nn.Module) -> GradientStats:
        """Compute gradient statistics without clipping.
        
        Args:
            model: Model whose gradients to analyze
            
        Returns:
            Gradient statistics
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return GradientStats(0, 0, 0, 0, 0, 1.0, False, False)

        total_norm = 0.0
        max_norm = 0.0
        min_norm = float("inf")
        has_nan = False
        has_inf = False

        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type).item()
            total_norm += param_norm ** self.norm_type
            max_norm = max(max_norm, param_norm)
            min_norm = min(min_norm, param_norm)

            if torch.isnan(p.grad.data).any():
                has_nan = True
            if torch.isinf(p.grad.data).any():
                has_inf = True

        total_norm = total_norm ** (1.0 / self.norm_type)

        return GradientStats(
            total_norm=total_norm,
            max_norm=max_norm,
            min_norm=min_norm,
            num_parameters=len(parameters),
            num_clipped=0,
            clip_coef=1.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )


def clip_grad_norm(
    parameters: torch.Tensor | Iterator[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradient norm of parameters.
    
    Args:
        parameters: Model parameters or single tensor
        max_norm: Maximum allowed norm
        norm_type: Type of norm (e.g., 2 for L2)
        
    Returns:
        Total norm of the parameters (before clipping)
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


def clip_grad_value(
    parameters: torch.Tensor | Iterator[torch.Tensor],
    clip_value: float,
) -> None:
    """Clip gradient values of parameters.
    
    Args:
        parameters: Model parameters or single tensor
        clip_value: Maximum allowed value
    """
    torch.nn.utils.clip_grad_value_(parameters, clip_value)


def compute_grad_norm(
    parameters: torch.Tensor | Iterator[torch.Tensor],
    norm_type: float = 2.0,
) -> float:
    """Compute gradient norm of parameters.
    
    Args:
        parameters: Model parameters or single tensor
        norm_type: Type of norm (e.g., 2 for L2)
        
    Returns:
        Total norm of the gradients
    """
    parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)

    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type

    return float((total_norm ** (1.0 / norm_type)).item())


def add_gradient_noise(
    model: nn.Module,
    iteration: int,
    eta: float = 0.01,
    gamma: float = 0.55,
) -> None:
    """Add Gaussian noise to gradients for regularization.
    
    Based on "Adding Gradient Noise Improves Learning for Very Deep Networks"
    https://arxiv.org/abs/1511.06807
    
    Args:
        model: Model whose gradients to add noise to
        iteration: Current training iteration
        eta: Base noise level
        gamma: Noise decay factor
    """
    if iteration <= 0:
        return

    # Compute noise standard deviation
    std = (eta / (1 + iteration) ** gamma) ** 0.5

    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * std
            p.grad.data.add_(noise)


def zero_grad(
    model: nn.Module,
    set_to_none: bool = True,
) -> None:
    """Zero out gradients of a model.
    
    Args:
        model: Model whose gradients to zero
        set_to_none: Set gradients to None instead of zero tensor
    """
    if set_to_none:
        for p in model.parameters():
            p.grad = None
    else:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


def accumulate_gradients(
    loss: torch.Tensor,
    model: nn.Module,
    accumulation_steps: int,
    current_step: int,
) -> bool:
    """Handle gradient accumulation.
    
    Args:
        loss: Loss to backpropagate
        model: Model to accumulate gradients for
        accumulation_steps: Number of accumulation steps
        current_step: Current accumulation step (0-indexed)
        
    Returns:
        True if gradients should be applied (accumulation complete)
    """
    # Scale loss by accumulation steps
    scaled_loss = loss / accumulation_steps

    # Backward pass
    scaled_loss.backward()

    # Check if we should apply gradients
    return (current_step + 1) % accumulation_steps == 0


def check_gradients(model: nn.Module) -> dict[str, bool]:
    """Check gradients for common issues.
    
    Args:
        model: Model to check
        
    Returns:
        Dictionary with gradient health checks
    """
    checks = {
        "has_gradients": False,
        "has_nan": False,
        "has_inf": False,
        "all_zero": True,
        "exploding": False,
        "vanishing": False,
    }

    grad_norms: list[float] = []

    for p in model.parameters():
        if p.grad is not None:
            checks["has_gradients"] = True

            # Check for NaN/Inf
            if torch.isnan(p.grad).any():
                checks["has_nan"] = True
            if torch.isinf(p.grad).any():
                checks["has_inf"] = True

            # Check if all zero
            if (p.grad != 0).any():
                checks["all_zero"] = False

            # Collect gradient norms
            grad_norm = p.grad.data.norm(2).item()
            grad_norms.append(grad_norm)

    if grad_norms:
        max_norm = max(grad_norms)
        min(grad_norms)

        # Check for exploding gradients
        if max_norm > 100:
            checks["exploding"] = True

        # Check for vanishing gradients
        if max_norm < 1e-6:
            checks["vanishing"] = True

    return checks


def log_gradient_info(
    model: nn.Module,
    step: int,
    logger_fn: callable | None = None,
) -> None:
    """Log gradient information for debugging.
    
    Args:
        model: Model to log gradients for
        step: Current training step
        logger_fn: Optional logging function
    """
    if logger_fn is None:
        logger_fn = logger.info

    # Compute gradient statistics
    total_norm = compute_grad_norm(model.parameters())
    health = check_gradients(model)

    # Log information
    logger_fn(f"Step {step} - Gradient norm: {total_norm:.4f}")

    if health["has_nan"]:
        logger_fn(f"Step {step} - WARNING: Gradients contain NaN")
    if health["has_inf"]:
        logger_fn(f"Step {step} - WARNING: Gradients contain Inf")
    if health["all_zero"]:
        logger_fn(f"Step {step} - WARNING: All gradients are zero")
    if health["exploding"]:
        logger_fn(f"Step {step} - WARNING: Gradients may be exploding")
    if health["vanishing"]:
        logger_fn(f"Step {step} - WARNING: Gradients may be vanishing")
