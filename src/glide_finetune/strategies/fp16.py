"""FP16 mixed precision training strategy implementation."""

from typing import Any

import torch as th
from torch import nn
from torch.utils.data import DataLoader

from glide_finetune.checkpoint_manager import CheckpointManager as CheckpointManagerImpl
from glide_finetune.fp16_training import (
    FP16TrainingConfig,
    FP16TrainingStep,
    SelectiveFP16Converter,
)
from glide_finetune.training_types import CheckpointManager, DiffusionProcess, TrainConfig
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.model_utils import (
    apply_model_modifications,
    create_dataloader,
    create_optimizer,
    load_glide_model,
)

logger = get_logger(__name__)


class FP16Strategy:
    """FP16 mixed precision training strategy."""

    def __init__(self, device: th.device) -> None:
        """Initialize FP16 strategy.
        
        Args:
            device: Device to run training on.
        """
        self.device = device
        self.trainer: FP16TrainingStep | None = None
        self.checkpoint_manager: CheckpointManagerImpl | None = None

    def setup_model(self, config: TrainConfig) -> tuple[nn.Module, DiffusionProcess, dict[str, Any]]:
        """Setup model for FP16 training.
        
        Args:
            config: Training configuration.
            
        Returns:
            Tuple of (model, diffusion, options).
        """
        model, diffusion, options = load_glide_model(config.model, device=self.device)

        # Apply freeze/randomization policies BEFORE FP16 conversion
        apply_model_modifications(model, config.model)

        # Apply FP16 conversion
        if config.fp16.use_fp16:
            logger.info(f"Converting model to FP16 (mode: {config.fp16.fp16_mode})")
            converter = SelectiveFP16Converter(aggressive=(config.fp16.fp16_mode == "aggressive"))
            model, conv_stats = converter.convert_model_mixed_precision(model)
            logger.info(f"  FP16 params: {conv_stats['fp16_params']:,}")
            logger.info(f"  FP32 params: {conv_stats['fp32_params']:,}")
            logger.info(
                f"  FP16 ratio: {conv_stats['fp16_params'] / (conv_stats['fp16_params'] + conv_stats['fp32_params']) * 100:.1f}%"
            )

        model.train()
        return model, diffusion, options

    def setup_optimizer(self, model: nn.Module, config: TrainConfig) -> th.optim.Optimizer:
        """Setup optimizer for FP16 training.
        
        Args:
            model: Model to optimize.
            config: Training configuration.
            
        Returns:
            Configured optimizer.
        """
        base_optimizer = create_optimizer(model, config.training, config.training.use_8bit_adam)

        # Create FP16 trainer
        fp16_config = FP16TrainingConfig(
            use_loss_scaling=config.fp16.use_fp16,
            init_loss_scale=config.fp16.fp16_loss_scale,
            use_master_weights=config.fp16.use_fp16,  # Always use master weights with FP16
            gradient_clip_norm=config.training.grad_clip,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_frequency=config.logging.log_frequency,
            enable_nan_recovery=config.fp16.use_fp16,
        )

        self.trainer = FP16TrainingStep(model, base_optimizer, fp16_config)
        return base_optimizer

    def setup_dataloader(self, config: TrainConfig, model: nn.Module) -> DataLoader[Any]:
        """Setup data loader for FP16 training.
        
        Args:
            config: Training configuration.
            model: Model (for tokenizer access).
            
        Returns:
            Configured DataLoader.
        """
        return create_dataloader(config, model, distributed=False)

    def setup_checkpoint_manager(self, config: TrainConfig) -> CheckpointManager:
        """Setup checkpoint manager.
        
        Args:
            config: Training configuration.
            
        Returns:
            Configured checkpoint manager.
        """
        save_dir = config.checkpoint.save_directory
        self.checkpoint_manager = CheckpointManagerImpl(
            checkpoints_dir=save_dir,
            save_frequency=config.logging.save_frequency
        )
        return self.checkpoint_manager

    def training_step(
        self,
        model: nn.Module,
        diffusion: DiffusionProcess,
        batch: Any,
        optimizer: th.optim.Optimizer,
        config: TrainConfig,
    ) -> dict[str, float]:
        """Perform FP16 training step.
        
        Args:
            model: Model to train.
            diffusion: Diffusion process.
            batch: Training batch.
            optimizer: Optimizer.
            config: Training configuration.
            
        Returns:
            Dictionary of training metrics.
        """
        if self.trainer is None:
            msg = "FP16 trainer not initialized"
            raise RuntimeError(msg)

        # Define loss computation function
        def compute_loss() -> th.Tensor:
            # Unpack batch
            if isinstance(batch, list | tuple):
                if len(batch) == 3:
                    tokens, masks, images = batch
                elif len(batch) == 4:
                    tokens, masks, images, _ = batch
                else:
                    msg = f"Unexpected batch format with {len(batch)} elements"
                    raise ValueError(msg)
            else:
                images = batch["images"]
                tokens = batch.get("tokens")
                masks = batch.get("masks")

            # Move to device
            images = images.to(self.device).float()
            if tokens is not None:
                tokens = tokens.to(self.device)
                masks = masks.to(self.device) if masks is not None else None

                # Apply unconditional training
                if config.data.uncond_p > 0:
                    mask = th.rand(images.shape[0], device=self.device) < config.data.uncond_p
                    tokens = tokens.clone()
                    tokens[mask] = 0

            # Sample timesteps
            timesteps = th.randint(
                0, len(diffusion.betas) - 1, (images.shape[0],), device=self.device
            )

            # Add noise
            noise = th.randn_like(images, device=self.device)
            x_t = diffusion.q_sample(images, timesteps, noise=noise)

            # Forward pass
            model_output = model(
                x_t,
                timesteps,
                tokens=tokens if tokens is not None else None,
                mask=masks if masks is not None else None,
            )

            # Compute loss
            _, C = x_t.shape[:2]
            epsilon, _ = th.split(model_output, C, dim=1)
            return th.nn.functional.mse_loss(epsilon, noise.detach())

        # Use FP16 trainer for the step
        return self.trainer.training_step(compute_loss)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: th.optim.Optimizer,
        epoch: int,
        step: int,
        is_interrupted: bool = False,
    ) -> str | None:
        """Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            epoch: Current epoch.
            step: Current step.
            is_interrupted: Whether this is an interrupt save.
            
        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        if self.checkpoint_manager:
            # For FP16, we need to save the master weights if they exist
            if self.trainer and self.trainer.master_weight_manager:
                # Temporarily swap in master weights for saving
                self.trainer.master_weight_manager.load_master_weights()
                path = self.checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, step, is_interrupted
                )
                # Swap back to FP16 weights
                self.trainer.master_weight_manager.convert_to_fp16()
                return path
            return self.checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, step, is_interrupted
            )
        return None

    def should_save(self, step: int) -> bool:
        """Check if should save at this step.
        
        Args:
            step: Current training step.
            
        Returns:
            Whether to save checkpoint.
        """
        if self.checkpoint_manager:
            return self.checkpoint_manager.should_save(step)
        return False

    def cleanup_interrupted_files(self) -> None:
        """Clean up interrupted checkpoint files."""
        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup_interrupted_files()
