#!/usr/bin/env python3
"""
Dynamic Loss Scaler for Manual FP16 Training
Implements dynamic loss scaling with NaN detection and recovery.
Based on best practices from APEX and DeepSpeed implementations.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("glide_finetune.dynamic_loss_scaler")


@dataclass
class LossScalerConfig:
    """Configuration for dynamic loss scaler."""

    init_scale: float = 2**8  # Start conservative (256) based on analysis
    scale_factor: float = 2.0  # Growth/decay factor
    scale_window: int = 200  # Steps before scale increase
    min_scale: float = 1.0  # Minimum scale value
    max_scale: float = 2**24  # Maximum scale value (16M)
    tolerance: float = 0.0  # Gradient overflow tolerance
    enabled: bool = True  # Enable/disable scaling


class DynamicLossScaler:
    """
    Dynamic loss scaler for manual FP16 training.

    Features:
    - Dynamic scale adjustment based on gradient overflow/underflow
    - NaN/Inf detection and recovery
    - Gradient statistics tracking
    - Automatic scale warmup
    """

    def __init__(self, config: LossScalerConfig | None = None):
        """Initialize the dynamic loss scaler."""
        self.config = config or LossScalerConfig()

        # Current scale
        self._scale = self.config.init_scale

        # Tracking
        self._unskipped = 0  # Successful steps since last overflow
        self._has_overflow = False
        self._has_nan = False

        # Statistics
        self.stats = {
            "total_steps": 0,
            "skipped_steps": 0,
            "scale_updates": 0,
            "max_scale_reached": 0,
            "min_scale_reached": 0,
            "nan_detected": 0,
            "inf_detected": 0,
            "gradient_norms": [],
        }

        # Logging
        self.logger = logging.getLogger(__name__)

    @property
    def scale(self) -> float:
        """Get current loss scale."""
        return self._scale if self.config.enabled else 1.0

    @property
    def inv_scale(self) -> float:
        """Get inverse scale for unscaling."""
        return 1.0 / self.scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale the loss for backward pass.

        Args:
            loss: Unscaled loss tensor

        Returns:
            Scaled loss tensor
        """
        if not self.config.enabled:
            return loss

        return loss * self.scale

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Unscale gradients in optimizer param groups.

        Args:
            optimizer: Optimizer with scaled gradients

        Returns:
            True if gradients are valid, False if overflow/NaN detected
        """
        if not self.config.enabled:
            return True

        self._has_overflow = False
        self._has_nan = False
        inv_scale = self.inv_scale

        # Unscale gradients for all parameter groups
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.mul_(inv_scale)

                    # Check for overflow/NaN
                    if not self._has_overflow:
                        self._check_grad_overflow(param.grad)

        return not (self._has_overflow or self._has_nan)

    def _check_grad_overflow(self, grad: torch.Tensor) -> None:
        """Check gradient for overflow or NaN."""
        # Check for NaN
        if torch.isnan(grad).any():
            self._has_nan = True
            self.stats["nan_detected"] += 1
            self.logger.warning("NaN detected in gradients")
            return

        # Check for Inf
        if torch.isinf(grad).any():
            self._has_overflow = True
            self.stats["inf_detected"] += 1
            self.logger.debug("Inf detected in gradients")
            return

        # Check for numerical overflow (very large values)
        grad_norm = grad.abs().max().item()
        if grad_norm > 65504:  # FP16 max
            self._has_overflow = True
            self.logger.debug(f"Gradient overflow: max value {grad_norm:.2e}")

    def check_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Check if gradients contain overflow/NaN.

        Args:
            optimizer: Optimizer to check

        Returns:
            True if overflow/NaN detected
        """
        self._has_overflow = False
        self._has_nan = False

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self._check_grad_overflow(param.grad)
                    if self._has_overflow or self._has_nan:
                        return True

        return False

    def update_scale(self, overflow: bool) -> None:
        """
        Update loss scale based on overflow status.

        Args:
            overflow: Whether overflow occurred in this step
        """
        if not self.config.enabled:
            return

        self.stats["total_steps"] += 1

        if overflow:
            # Decrease scale on overflow
            self.stats["skipped_steps"] += 1
            self._scale = max(self._scale / self.config.scale_factor, self.config.min_scale)
            self._unskipped = 0
            self.stats["scale_updates"] += 1

            if self._scale == self.config.min_scale:
                self.stats["min_scale_reached"] += 1
                self.logger.warning(f"Loss scale reached minimum: {self._scale}")
            else:
                self.logger.debug(f"Loss scale decreased to {self._scale}")
        else:
            # Increase scale after successful steps
            self._unskipped += 1

            if self._unskipped >= self.config.scale_window:
                old_scale = self._scale
                self._scale = min(self._scale * self.config.scale_factor, self.config.max_scale)
                self._unskipped = 0

                if self._scale != old_scale:
                    self.stats["scale_updates"] += 1
                    if self._scale == self.config.max_scale:
                        self.stats["max_scale_reached"] += 1
                    self.logger.debug(f"Loss scale increased to {self._scale}")

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Perform a complete scaling step.

        This method:
        1. Unscales gradients
        2. Checks for overflow/NaN
        3. Updates scale
        4. Performs optimizer step if gradients are valid

        Args:
            optimizer: Optimizer to step

        Returns:
            True if optimizer step was performed, False if skipped
        """
        # Unscale and check gradients
        valid_grads = self.unscale_gradients(optimizer)

        # Update scale based on overflow
        self.update_scale(not valid_grads)

        # Only step if gradients are valid
        if valid_grads:
            optimizer.step()
            return True
        # Skip optimizer step on overflow
        self.logger.debug("Skipping optimizer step due to gradient overflow")
        return False

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """
        Scale loss and perform backward pass.

        Args:
            loss: Unscaled loss tensor
            **kwargs: Additional arguments for backward()
        """
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward(**kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "scale": self._scale,
            "unskipped": self._unskipped,
            "stats": self.stats,
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._scale = state_dict.get("scale", self.config.init_scale)
        self._unskipped = state_dict.get("unskipped", 0)
        self.stats = state_dict.get("stats", self.stats)

        # Update config if provided
        if "config" in state_dict:
            for key, value in state_dict["config"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

    def get_grad_norm(self, optimizer: torch.optim.Optimizer, norm_type: float = 2.0) -> float:
        """
        Compute gradient norm for monitoring.

        Args:
            optimizer: Optimizer with gradients
            norm_type: Type of norm (default: 2.0 for L2 norm)

        Returns:
            Gradient norm value
        """
        total_norm = 0.0

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type

        total_norm = total_norm ** (1.0 / norm_type)
        self.stats["gradient_norms"].append(total_norm)

        # Keep only recent history
        if len(self.stats["gradient_norms"]) > 100:
            self.stats["gradient_norms"] = self.stats["gradient_norms"][-100:]

        return total_norm

    def log_stats(self) -> None:
        """Log current statistics."""
        if self.stats["total_steps"] == 0:
            return

        skip_rate = self.stats["skipped_steps"] / self.stats["total_steps"] * 100
        avg_grad_norm = (
            sum(self.stats["gradient_norms"]) / len(self.stats["gradient_norms"])
            if self.stats["gradient_norms"]
            else 0
        )

        self.logger.info(
            f"Loss Scaler Stats: scale={self._scale:.1f}, "
            f"skip_rate={skip_rate:.1f}%, "
            f"avg_grad_norm={avg_grad_norm:.2e}"
        )


class NaNRecoverySystem:
    """
    System for recovering from NaN during training.

    Strategies:
    1. Restore from last good checkpoint
    2. Reset optimizer state
    3. Reduce learning rate
    4. Increase loss scale
    """

    def __init__(self, max_recovery_attempts: int = 3):
        """Initialize NaN recovery system."""
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_attempts = 0
        self.last_good_state = None
        self.logger = logging.getLogger(__name__)

    def save_good_state(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_scaler: DynamicLossScaler
    ) -> None:
        """Save current state as last known good."""
        self.last_good_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_scaler": loss_scaler.state_dict(),
        }
        self.recovery_attempts = 0  # Reset on successful save

    def attempt_recovery(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_scaler: DynamicLossScaler
    ) -> bool:
        """
        Attempt to recover from NaN.

        Returns:
            True if recovery successful, False if max attempts exceeded
        """
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.logger.error("Max NaN recovery attempts exceeded")
            return False

        self.recovery_attempts += 1
        self.logger.warning(f"Attempting NaN recovery (attempt {self.recovery_attempts})")

        if self.recovery_attempts == 1:
            # First attempt: Increase loss scale
            loss_scaler._scale *= 4
            self.logger.info(f"Increased loss scale to {loss_scaler._scale}")

        elif self.recovery_attempts == 2:
            # Second attempt: Reset optimizer state
            optimizer.state = {}
            self.logger.info("Reset optimizer state")

        elif self.recovery_attempts == 3:
            # Third attempt: Restore from last good state
            if self.last_good_state is not None:
                model.load_state_dict(self.last_good_state["model"])
                optimizer.load_state_dict(self.last_good_state["optimizer"])
                loss_scaler.load_state_dict(self.last_good_state["loss_scaler"])

                # Reduce learning rate
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
                    self.logger.info(f"Reduced learning rate to {group['lr']}")

                self.logger.info("Restored from last good state")
            else:
                self.logger.error("No good state to restore from")
                return False

        return True


def test_loss_scaler():
    """Test the dynamic loss scaler."""
    logger.info("Testing Dynamic Loss Scaler...")

    # Create dummy model and optimizer
    model = nn.Linear(10, 10).half().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create loss scaler
    config = LossScalerConfig(init_scale=256, scale_window=10)
    scaler = DynamicLossScaler(config)

    logger.info(f"Initial scale: {scaler.scale}")

    # Simulate training steps
    for step in range(20):
        # Forward pass
        x = torch.randn(32, 10).half().cuda()
        y = model(x)
        loss = y.mean()

        # Scale and backward
        optimizer.zero_grad()
        scaler.backward(loss)

        # Simulate overflow every 5 steps
        if step % 5 == 0 and step > 0:
            # Inject large gradient
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(1e10)

        # Step with scaler
        stepped = scaler.step(optimizer)

        if stepped:
            logger.info(
                f"Step {step}: scale={scaler.scale:.1f}, grad_norm={scaler.get_grad_norm(optimizer):.2e}"
            )
        else:
            logger.info(f"Step {step}: SKIPPED due to overflow, scale={scaler.scale:.1f}")

    # Print final stats
    logger.info("\nFinal Statistics:")
    logger.info(f"  Total steps: {scaler.stats['total_steps']}")
    logger.info(f"  Skipped steps: {scaler.stats['skipped_steps']}")
    logger.info(f"  Scale updates: {scaler.stats['scale_updates']}")
    logger.info(f"  Final scale: {scaler.scale}")

    logger.info("\n✅ Dynamic Loss Scaler test complete!")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    test_loss_scaler()
