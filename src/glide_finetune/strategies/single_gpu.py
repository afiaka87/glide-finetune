"""Single GPU training strategy implementation."""

from typing import TYPE_CHECKING, Any

import torch as th
from torch import nn
from torch.utils.data import DataLoader

from glide_finetune.checkpoint_manager import CheckpointManager as CheckpointManagerImpl
from glide_finetune.training_types import CheckpointManager, DiffusionProcess, TrainConfig
from glide_finetune.utils.model_utils import (
    apply_model_modifications,
    create_dataloader,
    create_optimizer,
    load_glide_model,
)

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler


class SingleGPUStrategy:
    """Standard single GPU training strategy."""

    def __init__(self, device: th.device) -> None:
        """Initialize single GPU strategy.
        
        Args:
            device: Device to run training on.
        """
        self.device = device
        self.checkpoint_manager: CheckpointManagerImpl | None = None
        self.scaler: GradScaler | None = None

    def setup_model(self, config: TrainConfig) -> tuple[nn.Module, DiffusionProcess, dict[str, Any]]:
        """Setup model for single GPU training.
        
        Args:
            config: Training configuration.
            
        Returns:
            Tuple of (model, diffusion, options).
        """
        model, diffusion, options = load_glide_model(config.model, device=self.device)
        apply_model_modifications(model, config.model)
        model.train()
        return model, diffusion, options

    def setup_optimizer(self, model: nn.Module, config: TrainConfig) -> th.optim.Optimizer:
        """Setup optimizer for single GPU training.
        
        Args:
            model: Model to optimize.
            config: Training configuration.
            
        Returns:
            Configured optimizer.
        """
        return create_optimizer(model, config.training, config.training.use_8bit_adam)

    def setup_dataloader(self, config: TrainConfig, model: nn.Module) -> DataLoader[Any]:
        """Setup data loader for single GPU training.
        
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
        """Perform single GPU training step.
        
        Args:
            model: Model to train.
            diffusion: Diffusion process.
            batch: Training batch.
            optimizer: Optimizer.
            config: Training configuration.
            
        Returns:
            Dictionary of training metrics.
        """
        # Unpack batch
        if isinstance(batch, list | tuple):
            if len(batch) == 3:
                tokens, masks, images = batch
            elif len(batch) == 4:
                tokens, masks, images, _ = batch  # Ignore upsampled for base model
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

        # Forward pass
        timesteps = th.randint(
            0, len(diffusion.betas) - 1, (images.shape[0],), device=self.device
        )
        noise = th.randn_like(images, device=self.device)
        x_t = diffusion.q_sample(images, timesteps, noise=noise)

        model_output = model(
            x_t,
            timesteps,
            tokens=tokens if tokens is not None else None,
            mask=masks if masks is not None else None,
        )

        # Compute loss
        _, channels = x_t.shape[:2]
        epsilon, _ = th.split(model_output, channels, dim=1)
        loss = th.nn.functional.mse_loss(epsilon, noise.detach())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = th.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.grad_clip
        )

        optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

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
