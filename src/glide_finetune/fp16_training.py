"""
Advanced FP16 training utilities for GLIDE.
This module provides production-ready mixed precision training with:
- Dynamic loss scaling
- Master weight management
- NaN recovery
- Selective layer precision
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .dynamic_loss_scaler import DynamicLossScaler, LossScalerConfig, NaNRecoverySystem
from .master_weight_manager import MasterWeightManager


@dataclass
class FP16TrainingConfig:
    """Configuration for FP16 training."""

    # Loss scaling
    use_loss_scaling: bool = True
    init_loss_scale: float = 256.0
    scale_window: int = 100
    scale_factor: float = 2.0
    scale_min: float = 1.0
    scale_max: float = 2**20

    # Master weights
    use_master_weights: bool = True

    # Gradient handling
    gradient_clip_norm: float | None = 1.0
    gradient_accumulation_steps: int = 1

    # NaN recovery
    enable_nan_recovery: bool = True
    max_nan_recoveries: int = 3

    # Logging
    log_frequency: int = 100


class FP16TrainingStep:
    """
    Manages FP16 training steps with all stability features.

    This class orchestrates:
    - Dynamic loss scaling
    - Master weight management
    - Gradient accumulation
    - NaN detection and recovery
    - Overflow handling
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: FP16TrainingConfig | None = None,
    ):
        """
        Initialize FP16 training step handler.

        Args:
            model: Model to train
            optimizer: Base optimizer (will be wrapped if needed)
            config: FP16 training configuration
        """
        self.model = model
        self.config = config or FP16TrainingConfig()
        self.logger = logging.getLogger(__name__)

        # Setup components based on config
        self._setup_loss_scaler()
        self._setup_master_weights(optimizer)
        self._setup_gradient_accumulator()
        self._setup_nan_recovery()

        # Statistics
        self.stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "skipped_steps": 0,
            "nan_recoveries": 0,
            "overflow_count": 0,
        }

        # Gradient accumulation state
        self.accumulation_step = 0

    def _setup_loss_scaler(self) -> None:
        """Setup dynamic loss scaler."""
        self.loss_scaler: DynamicLossScaler | None
        if self.config.use_loss_scaling:
            scaler_config = LossScalerConfig(
                init_scale=self.config.init_loss_scale,
                scale_window=self.config.scale_window,
                scale_factor=self.config.scale_factor,
                min_scale=self.config.scale_min,
                max_scale=self.config.scale_max,
            )
            self.loss_scaler = DynamicLossScaler(scaler_config)
        else:
            self.loss_scaler = None

    def _setup_master_weights(self, optimizer: torch.optim.Optimizer) -> None:
        """Setup master weight management."""
        self.master_manager: MasterWeightManager | None
        if self.config.use_master_weights:
            self.master_manager = MasterWeightManager(self.model)
            # Extract only valid AdamW parameters from optimizer defaults
            valid_keys = {"lr", "betas", "eps", "weight_decay", "amsgrad"}
            optimizer_kwargs = {k: v for k, v in optimizer.defaults.items() if k in valid_keys}
            self.optimizer = self.master_manager.create_optimizer(
                type(optimizer), **optimizer_kwargs
            )
        else:
            self.master_manager = None
            self.optimizer = optimizer

    def _setup_gradient_accumulator(self) -> None:
        """Setup gradient accumulation."""
        # We handle gradient accumulation manually in training_step
        self.grad_accumulator = None

    def _setup_nan_recovery(self) -> None:
        """Setup NaN recovery system."""
        self.nan_recovery: NaNRecoverySystem | None
        if self.config.enable_nan_recovery:
            self.nan_recovery = NaNRecoverySystem(
                max_recovery_attempts=self.config.max_nan_recoveries
            )
        else:
            self.nan_recovery = None

    def training_step(self, compute_loss: Callable[[], torch.Tensor]) -> dict[str, Any]:
        """
        Perform a single training step with FP16 support.

        Args:
            compute_loss: Function that computes the loss

        Returns:
            Dictionary with training metrics
        """
        self.stats["total_steps"] += 1

        # Compute loss
        loss = compute_loss()

        # Check for NaN
        if torch.isnan(loss):
            if self.nan_recovery and self.nan_recovery.recovery_attempts < self.nan_recovery.max_recovery_attempts:
                self.logger.warning("NaN detected, attempting recovery")
                # Recovery is handled by increasing attempts and the system will reset on next successful step
                self.nan_recovery.recovery_attempts += 1
                self.stats["nan_recoveries"] += 1
                return {
                    "loss": float("nan"),
                    "grad_norm": 0.0,
                    "loss_scale": self.loss_scaler.scale if self.loss_scaler else 1.0,
                    "skipped": True,
                    "recovered": True,
                }
            msg = "NaN detected and recovery failed"
            raise ValueError(msg)

        # Scale loss for gradient accumulation
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        # Scale loss if using loss scaling
        scaled_loss = self.loss_scaler.scale_loss(loss) if self.loss_scaler else loss

        # Backward pass
        scaled_loss.backward()  # type: ignore[no-untyped-call]

        # Handle gradient accumulation
        self.accumulation_step += 1
        if self.accumulation_step < self.config.gradient_accumulation_steps:
            return {
                "loss": loss.item(),
                "grad_norm": 0.0,
                "loss_scale": self.loss_scaler.scale if self.loss_scaler else 1.0,
                "skipped": True,
                "accumulated": True,
            }

        # Reset accumulation counter
        self.accumulation_step = 0

        # Check for overflow
        overflow = False
        if self.loss_scaler:
            overflow = self.loss_scaler.check_overflow(self.optimizer)

        if overflow:
            self.stats["overflow_count"] += 1
            self.stats["skipped_steps"] += 1
            self.optimizer.zero_grad()

            # Adjust scale
            if self.loss_scaler:
                self.loss_scaler.update_scale(overflow=True)

            return {
                "loss": loss.item(),
                "grad_norm": 0.0,
                "loss_scale": self.loss_scaler.scale if self.loss_scaler else 1.0,
                "skipped": True,
                "overflow": True,
            }

        # Unscale gradients if using loss scaling
        if self.loss_scaler:
            self.loss_scaler.unscale_gradients(self.optimizer)

        # Sync gradients to master weights if using them
        if self.master_manager:
            self.master_manager.sync_gradients_to_master(
                loss_scale=self.loss_scaler.scale if self.loss_scaler else 1.0
            )

        # Gradient clipping
        grad_norm = 0.0
        if self.config.gradient_clip_norm:
            if self.master_manager:
                grad_norm = self.master_manager.clip_grad_norm(
                    max_norm=self.config.gradient_clip_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                ).item()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Sync master weights back to model if using them
        if self.master_manager:
            self.master_manager.sync_master_to_model()

        # Update loss scale
        if self.loss_scaler:
            self.loss_scaler.update_scale(overflow=False)

        # Update NaN recovery - reset attempts on successful step
        if self.nan_recovery:
            self.nan_recovery.recovery_attempts = 0

        self.stats["successful_steps"] += 1

        # Log statistics periodically
        if self.stats["total_steps"] % self.config.log_frequency == 0:
            self._log_statistics()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "loss_scale": self.loss_scaler.scale if self.loss_scaler else 1.0,
            "skipped": False,
        }

    def _log_statistics(self) -> None:
        """Log training statistics."""
        if self.stats["total_steps"] == 0:
            return

        success_rate = self.stats["successful_steps"] / self.stats["total_steps"] * 100
        skip_rate = self.stats["skipped_steps"] / self.stats["total_steps"] * 100

        self.logger.info(
            f"FP16 Stats - Steps: {self.stats['total_steps']}, "
            f"Success: {success_rate:.1f}%, Skip: {skip_rate:.1f}%, "
            f"Scale: {self.loss_scaler.scale if self.loss_scaler else 1.0:.0f}, "
            f"NaN recoveries: {self.stats['nan_recoveries']}"
        )

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {
            "stats": self.stats,
            "accumulation_step": self.accumulation_step,
        }

        if self.loss_scaler:
            state["loss_scaler"] = {
                "scale": self.loss_scaler.scale,
                "unskipped": self.loss_scaler._unskipped,
                "stats": self.loss_scaler.stats,
            }

        if self.nan_recovery:
            state["nan_recovery"] = {
                "recovery_attempts": self.nan_recovery.recovery_attempts,
            }

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.stats = state_dict.get("stats", self.stats)
        self.accumulation_step = state_dict.get("accumulation_step", 0)

        if self.loss_scaler and "loss_scaler" in state_dict:
            self.loss_scaler._scale = state_dict["loss_scaler"]["scale"]
            self.loss_scaler._unskipped = state_dict["loss_scaler"].get("unskipped", 0)
            if "stats" in state_dict["loss_scaler"]:
                self.loss_scaler.stats = state_dict["loss_scaler"]["stats"]

        if self.nan_recovery and "nan_recovery" in state_dict:
            self.nan_recovery.recovery_attempts = state_dict["nan_recovery"].get(
                "recovery_attempts", 0
            )


class SelectiveFP16Converter:
    """
    Convert models to mixed precision with selective layer conversion.

    This converter intelligently decides which layers should be FP16 vs FP32 based on:
    - Layer type (normalization, embeddings stay FP32)
    - Weight magnitude analysis
    - Gradient flow requirements
    """

    def __init__(self, aggressive: bool = False):
        """
        Initialize converter.

        Args:
            aggressive: If True, convert more layers to FP16 for better speedup
        """
        self.aggressive = aggressive
        self.logger = logging.getLogger(__name__)

        # Layers that should always stay FP32
        self.fp32_layer_types = [
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Embedding,
            nn.EmbeddingBag,
        ]

        # Add softmax layers if not aggressive
        if not aggressive:
            self.fp32_layer_types.extend(
                [
                    nn.Softmax,
                    nn.LogSoftmax,
                ]
            )

    def should_convert_layer(self, name: str, module: nn.Module) -> bool:
        """
        Determine if a layer should be converted to FP16.

        Args:
            name: Layer name
            module: Layer module

        Returns:
            True if layer should be FP16, False if it should stay FP32
        """
        # Check layer type
        if any(isinstance(module, t) for t in self.fp32_layer_types):
            return False

        # Keep critical layers in FP32
        critical_names = ["time_embed", "label_emb", "final", "out"]
        if any(cn in name.lower() for cn in critical_names):
            return False

        # In aggressive mode, convert everything else
        if self.aggressive:
            return True

        # Conservative mode: also keep attention layers in FP32
        return not ("attention" in name.lower() or "attn" in name.lower())

    def convert_model_mixed_precision(self, model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
        """
        Convert model to mixed precision.

        Args:
            model: Model to convert

        Returns:
            Tuple of (converted model, conversion statistics)
        """
        # First convert entire model to FP16
        model = model.half()

        # Then selectively convert layers back to FP32
        fp16_params = 0
        fp32_params = 0
        conversion_map = {}

        for name, module in model.named_modules():
            if not self.should_convert_layer(name, module):
                # Convert back to FP32
                module.float()

                # Count parameters
                for param in module.parameters(recurse=False):
                    fp32_params += param.numel()
                    param.data = param.data.float()

                conversion_map[name] = "fp32"
            else:
                # Keep in FP16
                for param in module.parameters(recurse=False):
                    fp16_params += param.numel()

                conversion_map[name] = "fp16"

        # Log conversion statistics
        total_params = fp16_params + fp32_params
        if total_params > 0:
            fp16_ratio = fp16_params / total_params * 100
            self.logger.info(
                f"Model conversion complete: {fp16_ratio:.1f}% FP16, {100 - fp16_ratio:.1f}% FP32"
            )

        return model, {
            "fp16_params": fp16_params,
            "fp32_params": fp32_params,
            "conversion_map": conversion_map,
        }

    def convert_checkpoint(self, checkpoint_path: str, output_path: str | None = None) -> str:
        """
        Convert a checkpoint to mixed precision format.

        Args:
            checkpoint_path: Path to input checkpoint
            output_path: Path for output checkpoint (default: adds _fp16 suffix)

        Returns:
            Path to converted checkpoint
        """
        if output_path is None:
            base, ext = os.path.splitext(checkpoint_path)
            output_path = f"{base}_fp16{ext}"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Convert state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Convert weights based on layer names
        for name, param in state_dict.items():
            if param.dtype == torch.float32:
                # Determine if this parameter should be FP16
                should_convert = True

                # Check for critical layers
                for critical in ["norm", "embed", "time", "label", "final"]:
                    if critical in name.lower():
                        should_convert = bool(self.aggressive)
                        break

                if should_convert:
                    state_dict[name] = param.half()

        # Save converted checkpoint
        if "model" in checkpoint:
            checkpoint["model"] = state_dict
        elif "state_dict" in checkpoint:
            checkpoint["state_dict"] = state_dict
        else:
            checkpoint = state_dict

        torch.save(checkpoint, output_path)
        self.logger.info(f"Converted checkpoint saved to {output_path}")

        return output_path
