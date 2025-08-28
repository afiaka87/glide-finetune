"""Multi-GPU distributed training strategy implementation using Accelerate."""

import contextlib
import io
from typing import Any

import torch as th
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch import nn
from torch.utils.data import DataLoader

from glide_finetune.checkpoint_manager import CheckpointManager as CheckpointManagerImpl
from glide_finetune.training_types import CheckpointManager, DiffusionProcess, TrainConfig
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.model_utils import (
    apply_model_modifications,
    create_dataloader,
    create_optimizer,
    load_glide_model,
)

logger = get_logger(__name__)


class MultiGPUStrategy:
    """Multi-GPU distributed training strategy using Accelerate."""

    def __init__(self) -> None:
        """Initialize multi-GPU strategy."""
        self.accelerator: Accelerator | None = None
        self.checkpoint_manager: CheckpointManagerImpl | None = None

    def setup_accelerator(self, config: TrainConfig) -> Accelerator:
        """Setup Accelerator for distributed training.
        
        Args:
            config: Training configuration.
            
        Returns:
            Configured Accelerator instance.
        """
        # Configure project for logging and checkpointing
        project_config = ProjectConfiguration(
            project_dir=config.checkpoint.save_directory,
            automatic_checkpoint_naming=False,
            total_limit=5,
        )

        # Determine mixed precision mode
        mixed_precision = None
        if config.fp16.use_fp16:
            mixed_precision = "fp16"
        elif config.fp16.bf16:
            mixed_precision = "bf16"

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb" if config.logging.use_wandb else None,
            project_config=project_config,
            step_scheduler_with_optimizer=False,
        )

        # Initialize wandb tracking if enabled
        if self.accelerator.is_main_process and config.logging.use_wandb:
            self.accelerator.init_trackers(
                project_name=config.logging.wandb_project,
                config=config.dict(),
                init_kwargs={"wandb": {"dir": config.checkpoint.save_directory}},
            )

        return self.accelerator

    def setup_model(self, config: TrainConfig) -> tuple[nn.Module, DiffusionProcess, dict[str, Any]]:
        """Setup model for multi-GPU training.
        
        Args:
            config: Training configuration.
            
        Returns:
            Tuple of (model, diffusion, options).
        """
        if self.accelerator is None:
            self.accelerator = self.setup_accelerator(config)

        model, diffusion, options = load_glide_model(config.model)

        # Apply freeze/randomization policies (only print on main process)
        if (
            config.model.freeze_transformer
            or config.model.freeze_diffusion
            or config.model.randomize_transformer
            or config.model.randomize_diffusion
        ):
            if self.accelerator.is_main_process:
                apply_model_modifications(model, config.model)
            else:
                # Apply quietly on other processes
                with contextlib.redirect_stdout(io.StringIO()):
                    apply_model_modifications(model, config.model)

        return model, diffusion, options

    def setup_optimizer(self, model: nn.Module, config: TrainConfig) -> th.optim.Optimizer:
        """Setup optimizer for multi-GPU training.
        
        Args:
            model: Model to optimize.
            config: Training configuration.
            
        Returns:
            Configured optimizer.
        """
        return create_optimizer(model, config.training, config.training.use_8bit_adam)

    def setup_dataloader(self, config: TrainConfig, model: nn.Module) -> DataLoader[Any]:
        """Setup data loader for multi-GPU training.
        
        Args:
            config: Training configuration.
            model: Model (for tokenizer access).
            
        Returns:
            Configured DataLoader.
        """
        return create_dataloader(config, model, distributed=True)

    def setup_checkpoint_manager(self, config: TrainConfig) -> CheckpointManager | None:
        """Setup checkpoint manager (only on main process).
        
        Args:
            config: Training configuration.
            
        Returns:
            Configured checkpoint manager or None for non-main processes.
        """
        if self.accelerator and self.accelerator.is_main_process:
            save_dir = config.checkpoint.save_directory
            self.checkpoint_manager = CheckpointManagerImpl(
                checkpoints_dir=save_dir,
                save_frequency=config.logging.save_frequency,
            )
            return self.checkpoint_manager
        return None

    def training_step(
        self,
        model: nn.Module,
        diffusion: DiffusionProcess,
        batch: Any,
        optimizer: th.optim.Optimizer,
        config: TrainConfig,
    ) -> dict[str, float]:
        """Perform multi-GPU training step.
        
        Args:
            model: Model to train.
            diffusion: Diffusion process.
            batch: Training batch.
            optimizer: Optimizer.
            config: Training configuration.
            
        Returns:
            Dictionary of training metrics.
        """
        if self.accelerator is None:
            msg = "Accelerator not initialized"
            raise RuntimeError(msg)

        with self.accelerator.accumulate(model):
            # Forward pass
            if config.model.train_upsample:
                tokens, masks, low_res, high_res = batch
                loss = diffusion.training_losses(
                    model,
                    high_res,
                    t=None,
                    model_kwargs={"tokens": tokens, "mask": masks, "low_res": low_res},
                )["loss"].mean()
            else:
                tokens, masks, images = batch
                loss = diffusion.training_losses(
                    model,
                    images,
                    t=None,
                    model_kwargs={"tokens": tokens, "mask": masks},
                )["loss"].mean()

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient clipping
            if config.training.grad_clip > 0:
                self.accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        # Properly average loss across all processes
        avg_loss = self.accelerator.gather(loss.detach()).mean().item()

        # Get gradient norm for metrics
        grad_norm = 0.0
        if config.training.grad_clip > 0:
            grad_norm = th.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip
            ).item()

        return {
            "loss": avg_loss,
            "grad_norm": grad_norm,
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
        """Save a checkpoint using accelerator.save_state for proper distributed saving.
        
        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            epoch: Current epoch.
            step: Current step.
            is_interrupted: Whether this is an interrupt save.
            
        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        if self.checkpoint_manager and self.accelerator:
            # Build checkpoint path
            checkpoint_dir = self.checkpoint_manager.checkpoints_dir
            
            if is_interrupted:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = checkpoint_dir / f"interrupted_{timestamp}"
            else:
                save_path = checkpoint_dir / f"checkpoint_{step:08d}"
            
            # Use accelerator.save_state which handles distributed coordination
            # This automatically saves model, optimizer, RNG states, etc.
            self.accelerator.save_state(str(save_path))
            
            # Save metadata alongside (only on main process)
            if self.accelerator.is_main_process:
                import json
                metadata = {
                    "epoch": epoch,
                    "global_step": step,
                    "interrupted": is_interrupted,
                }
                metadata_path = save_path / "metadata.json"
                with metadata_path.open("w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"💾 Saved checkpoint to {save_path}")
                return str(save_path)
        
        return None

    def should_save(self, step: int) -> bool:
        """Check if should save at this step (only on main process).
        
        Args:
            step: Current training step.
            
        Returns:
            Whether to save checkpoint.
        """
        if self.checkpoint_manager and self.accelerator and self.accelerator.is_main_process:
            return self.checkpoint_manager.should_save(step)
        return False

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module | None = None,
        optimizer: th.optim.Optimizer | None = None,
    ) -> tuple[int, int]:
        """Load a checkpoint using accelerator.load_state.
        
        Args:
            checkpoint_path: Path to checkpoint directory.
            model: Model (not used, kept for interface compatibility).
            optimizer: Optimizer (not used, kept for interface compatibility).
            
        Returns:
            Tuple of (epoch, global_step) from checkpoint metadata.
        """
        from pathlib import Path
        import json
        
        checkpoint_dir = Path(checkpoint_path)
        epoch = 0
        global_step = 0
        
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            # Load state using accelerator (handles distributed loading)
            self.accelerator.load_state(str(checkpoint_dir))
            
            # Load metadata
            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open() as f:
                    metadata = json.load(f)
                    epoch = metadata.get("epoch", 0)
                    global_step = metadata.get("global_step", 0)
                    
                logger.info(f"✓ Loaded checkpoint from {checkpoint_dir}")
                logger.info(f"  Resuming from epoch {epoch}, step {global_step}")
        else:
            logger.warning(f"Checkpoint directory {checkpoint_dir} not found")
            
        return epoch, global_step
    
    def cleanup_interrupted_files(self) -> None:
        """Clean up interrupted checkpoint files (only on main process)."""
        if self.checkpoint_manager and self.accelerator and self.accelerator.is_main_process:
            # Clean up old-style interrupted files
            self.checkpoint_manager.cleanup_interrupted_files()
            
            # Also clean up new accelerator-style interrupted checkpoints
            from pathlib import Path
            checkpoint_dir = self.checkpoint_manager.checkpoints_dir
            for interrupted_dir in checkpoint_dir.glob("interrupted_*"):
                if interrupted_dir.is_dir():
                    import shutil
                    shutil.rmtree(interrupted_dir)
                    logger.info(f"🧹 Cleaned up {interrupted_dir.name}")

    def prepare_components(
        self,
        model: nn.Module,
        optimizer: th.optim.Optimizer,
        dataloader: DataLoader[Any],
    ) -> tuple[nn.Module, th.optim.Optimizer, DataLoader[Any]]:
        """Prepare model, optimizer, and dataloader for distributed training.
        
        Args:
            model: Model to prepare.
            optimizer: Optimizer to prepare.
            dataloader: DataLoader to prepare.
            
        Returns:
            Tuple of prepared (model, optimizer, dataloader).
        """
        if self.accelerator is None:
            msg = "Accelerator not initialized"
            raise RuntimeError(msg)

        return self.accelerator.prepare(model, optimizer, dataloader)
