"""Core training pipeline logic extracted from train.py.

This module contains the pure training logic separated from CLI and setup code.
It provides reusable training functions that can be called from different interfaces.
"""

import math
import signal
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from glide_finetune.glide_finetune import create_image_grid
from glide_finetune.training_types import (
    CheckpointManager,
    DiffusionProcess,
    TrainConfig,
    TrainingMetrics,
    TrainingStrategy,
)
from glide_finetune.utils.glide_util import sample
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.train_util import create_warmup_scheduler, pred_to_pil

logger = get_logger(__name__)


class InterruptHandler:
    """Handle interruption signals for graceful shutdown."""

    def __init__(self) -> None:
        """Initialize interrupt handler."""
        self.interrupted: bool = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal."""
        logger.info("\n\nReceived interrupt signal. Saving checkpoint and exiting gracefully...")
        self.interrupted = True


def setup_training(
    config: TrainConfig,
    strategy: TrainingStrategy
) -> tuple[nn.Module, DiffusionProcess, dict[str, Any], th.optim.Optimizer, DataLoader[Any],
           CheckpointManager | None, LambdaLR | None]:
    """Setup all training components.
    
    Args:
        config: Training configuration
        strategy: Training strategy to use
        
    Returns:
        Tuple of (model, diffusion, options, optimizer, dataloader, 
                  checkpoint_manager, scheduler)
    """
    logger.info("Setting up model...")
    model, diffusion, options = strategy.setup_model(config)

    logger.info("Setting up optimizer...")
    optimizer = strategy.setup_optimizer(model, config)

    logger.info("Setting up data loader...")
    dataloader = strategy.setup_dataloader(config, model)

    # Setup checkpoint manager
    checkpoint_manager = strategy.setup_checkpoint_manager(config)

    # Setup warmup scheduler if needed
    scheduler = None
    if hasattr(config.training, "warmup_steps") and config.training.warmup_steps > 0:
        scheduler = create_warmup_scheduler(
            optimizer,
            config.training.warmup_steps,
            getattr(config.training, "warmup_start_lr", 0.0),
            config.training.learning_rate,
        )
        logger.info(f"Setup warmup scheduler: {config.training.warmup_steps} steps")

    return model, diffusion, options, optimizer, dataloader, checkpoint_manager, scheduler


def load_training_state(
    config: TrainConfig,
    checkpoint_manager: CheckpointManager | None,
    model: nn.Module,
    optimizer: th.optim.Optimizer
) -> tuple[int, int]:
    """Load training state from checkpoint if needed.
    
    Args:
        config: Training configuration
        checkpoint_manager: Checkpoint manager instance
        model: Model to load state into
        optimizer: Optimizer to load state into
        
    Returns:
        Tuple of (start_epoch, global_step)
    """
    start_epoch = 0
    global_step = 0

    if hasattr(config.model, "resume_ckpt") and config.model.resume_ckpt and checkpoint_manager:
        try:
            start_epoch, global_step = checkpoint_manager.load_checkpoint(
                config.model.resume_ckpt, model, optimizer
            )
            logger.info(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")

            if global_step > 0:
                checkpoint_manager.cleanup_interrupted_files()

        except Exception as e:
            logger.warning(f"Warning: Failed to resume from checkpoint: {e}")
            logger.info("Starting from scratch...")

    return start_epoch, global_step


def train_epoch(
    epoch: int,
    model: nn.Module,
    diffusion: DiffusionProcess,
    dataloader: DataLoader[Any],
    optimizer: th.optim.Optimizer,
    strategy: TrainingStrategy,
    config: TrainConfig,
    global_step: int,
    scheduler: LambdaLR | None,
    checkpoint_manager: CheckpointManager | None,
    interrupt_handler: InterruptHandler,
    eval_prompts: list[str] | None,
    grid_size: int,
    output_dir: Path,
    wandb_run: Any | None
) -> tuple[list[float], int, bool]:
    """Train for one epoch.
    
    Args:
        epoch: Current epoch number
        model: Model to train
        diffusion: Diffusion instance
        dataloader: Data loader
        optimizer: Optimizer
        strategy: Training strategy
        config: Training configuration
        global_step: Current global step
        scheduler: Optional learning rate scheduler
        checkpoint_manager: Optional checkpoint manager
        interrupt_handler: Interrupt handler
        eval_prompts: Optional evaluation prompts
        grid_size: Size of sample grid
        output_dir: Output directory for samples
        wandb_run: Optional wandb run
        
    Returns:
        Tuple of (epoch_losses, updated_global_step, interrupted)
    """
    epoch_losses = []

    # Set up progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        if interrupt_handler.interrupted:
            break

        # Training step
        result = strategy.training_step(model, diffusion, batch, optimizer, config)

        global_step += 1

        # Update scheduler if present
        if scheduler is not None:
            scheduler.step()

        # Track loss
        if not np.isnan(result["loss"]):
            epoch_losses.append(result["loss"])

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{result['loss']:.4f}",
            "lr": f"{result.get('learning_rate', 0):.2e}",
            "step": global_step,
        })

        # Log metrics
        log_training_metrics(result, epoch, batch_idx, len(dataloader), global_step, wandb_run)

        # Generate samples if needed
        if should_generate_samples(global_step, config):
            generate_training_samples(
                model, diffusion, strategy, config, eval_prompts,
                grid_size, global_step, output_dir, wandb_run
            )

        # Save checkpoint if needed
        if checkpoint_manager and checkpoint_manager.should_save(global_step):
            logger.info(f"\nSaving checkpoint at step {global_step}...")
            checkpoint_manager.save_checkpoint(model, optimizer, epoch, global_step)

    progress_bar.close()

    return epoch_losses, global_step, interrupt_handler.interrupted


def log_training_metrics(
    result: TrainingMetrics,
    epoch: int,
    batch_idx: int,
    num_batches: int,
    global_step: int,
    wandb_run: Any | None
) -> None:
    """Log training metrics to wandb.
    
    Args:
        result: Training step result
        epoch: Current epoch
        batch_idx: Current batch index
        num_batches: Total number of batches
        global_step: Global step count
        wandb_run: Optional wandb run
    """
    if not wandb_run:
        return

    # Ensure all values are Python scalars
    loss_val = result["loss"]
    if hasattr(loss_val, "item"):
        loss_val = loss_val.item()
    loss_val = float(loss_val)

    # Skip logging if loss is NaN or inf
    if math.isnan(loss_val) or math.isinf(loss_val):
        logger.warning(f"Warning: Skipping W&B log at step {global_step} due to NaN/inf loss")
        return

    log_data = {
        "train/loss": loss_val,
        "train/learning_rate": float(result.get("learning_rate", 0)),
        "train/epoch": float(epoch + (batch_idx / num_batches)),
        "train/global_step": int(global_step),
    }

    # Add optional metrics
    if "grad_norm" in result:
        grad_val = result["grad_norm"]
        if hasattr(grad_val, "item"):
            grad_val = grad_val.item()
        log_data["train/grad_norm"] = float(grad_val)

    if "loss_scale" in result:
        scale_val = result["loss_scale"]
        if hasattr(scale_val, "item"):
            scale_val = scale_val.item()
        log_data["train/loss_scale"] = float(scale_val)

    wandb_run.log(log_data, step=global_step)


def should_generate_samples(global_step: int, config: TrainConfig) -> bool:
    """Check if samples should be generated at current step.
    
    Args:
        global_step: Current global step
        config: Training configuration
        
    Returns:
        True if samples should be generated
    """
    if global_step == 0:
        return False

    sample_freq = getattr(config.logging, "sample_frequency", 1000)
    return global_step % sample_freq == 0


def generate_training_samples(
    model: nn.Module,
    diffusion: DiffusionProcess,
    strategy: TrainingStrategy,
    config: TrainConfig,
    eval_prompts: list[str] | None,
    grid_size: int,
    global_step: int,
    output_dir: Path,
    wandb_run: Any | None
) -> None:
    """Generate and log training samples.
    
    Args:
        model: Model to generate samples with
        diffusion: Diffusion instance
        strategy: Training strategy
        config: Training configuration
        eval_prompts: Evaluation prompts
        grid_size: Size of sample grid
        global_step: Current global step
        output_dir: Output directory
        wandb_run: Optional wandb run
    """
    if not eval_prompts:
        # Use test prompt from config
        eval_prompts = [config.sampling.test_prompt]

    logger.info(f"\nGenerating samples at step {global_step}...")

    # Get device from model
    device = next(model.parameters()).device

    # Get model options from diffusion (standard GLIDE options)
    options = {
        "diffusion_steps": 1000,  # Standard GLIDE steps
        "noise_schedule": "linear",
    }

    # Generate samples
    model.eval()
    sample_images = []

    with th.no_grad():
        for prompt_idx, prompt in enumerate(eval_prompts[:grid_size]):
            # Generate sample using the GLIDE sample function
            samples = sample(
                glide_model=model,
                glide_options=options,
                side_x=config.data.side_x,
                side_y=config.data.side_y,
                prompt=prompt,
                batch_size=1,
                guidance_scale=config.sampling.test_guidance_scale,
                device=str(device),
                prediction_respacing=str(config.sampling.timestep_respacing)
                if config.sampling.num_steps is None
                else str(config.sampling.num_steps),
                sampler=config.sampling.sampler,
                num_steps=config.sampling.num_steps,
                eta=config.sampling.eta,
                use_swinir=config.sampling.use_swinir,
                swinir_model_type=config.sampling.swinir_model_type,
            )

            # Convert to PIL image
            sample_img = pred_to_pil(samples)
            sample_images.append(sample_img)

            # Save individual sample
            sample_path = output_dir / f"step{global_step:06d}_prompt{prompt_idx:03d}.png"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_img.save(sample_path)
            logger.debug(f"Saved sample to {sample_path}")

    # Create and save grid if multiple samples
    if len(sample_images) > 1:
        # Calculate grid dimensions
        grid_configs = {
            1: (1, 1),
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (4, 8),
            64: (8, 8),
        }
        grid_rows, grid_cols = grid_configs.get(grid_size, (2, 2))

        grid_img = create_image_grid(sample_images, rows=grid_rows, cols=grid_cols)
        grid_path = output_dir / f"step{global_step:06d}_grid.png"
        grid_img.save(grid_path)
        logger.info(f"Saved sample grid to {grid_path}")

        # Log to wandb if available
        if wandb_run:
            import wandb  # noqa: PLC0415 - Optional dependency, imported on demand
            wandb_run.log({
                "samples/grid": wandb.Image(grid_img, caption=f"Step {global_step}"),
            }, step=global_step)
    elif sample_images and wandb_run:
        # Log single image to wandb
        import wandb  # noqa: PLC0415 - Optional dependency, imported on demand
        wandb_run.log({
            "samples/image": wandb.Image(sample_images[0], caption=f"Step {global_step}"),
        }, step=global_step)

    model.train()
    logger.info(f"Generated {len(sample_images)} samples at step {global_step}")


def log_epoch_summary(
    epoch: int,
    epoch_losses: list[float],
    global_step: int,
    wandb_run: Any | None
) -> None:
    """Log epoch summary statistics.
    
    Args:
        epoch: Current epoch (0-indexed)
        epoch_losses: List of losses from the epoch
        global_step: Current global step
        wandb_run: Optional wandb run
    """
    if not epoch_losses:
        return

    avg_loss = np.mean(epoch_losses)
    std_loss = np.std(epoch_losses)

    logger.info(f"\nEpoch {epoch + 1} complete:")
    logger.info(f"  Average loss: {avg_loss:.6f} (±{std_loss:.6f})")
    logger.info(f"  Total steps: {global_step:,}")

    if wandb_run:
        wandb_run.log({
            "epoch/avg_loss": avg_loss,
            "epoch/std_loss": std_loss,
            "epoch/num": epoch + 1,
        }, step=global_step)


def save_final_checkpoint(
    checkpoint_manager: CheckpointManager | None,
    model: nn.Module,
    optimizer: th.optim.Optimizer,
    num_epochs: int,
    global_step: int,
    interrupted: bool
) -> None:
    """Save final checkpoint if not interrupted.
    
    Args:
        checkpoint_manager: Checkpoint manager
        model: Model to save
        optimizer: Optimizer to save
        num_epochs: Total number of epochs
        global_step: Final global step
        interrupted: Whether training was interrupted
    """
    if checkpoint_manager and not interrupted:
        logger.info("\nSaving final checkpoint...")
        checkpoint_manager.save_checkpoint(
            model, optimizer, num_epochs, global_step
        )


def run_training_pipeline(
    config: TrainConfig,
    strategy: TrainingStrategy,
    wandb_run: Any | None = None,
    eval_prompts: list[str] | None = None,
    grid_size: int = 4
) -> dict[str, int | float | bool | None]:
    """Main training pipeline execution.
    
    This is the core training logic extracted from train.py, providing
    a clean interface for running training with different strategies.
    
    Args:
        config: Complete training configuration
        strategy: Training strategy (SingleGPU, FP16, or MultiGPU)
        wandb_run: Optional wandb run for logging
        eval_prompts: Optional evaluation prompts for sampling
        grid_size: Size of sample grid (default: 4)
        
    Returns:
        Dictionary with training results and statistics
    """
    # Setup interrupt handler
    interrupt_handler = InterruptHandler()

    # Initialize variables for exception handling
    checkpoint_manager = None
    model = None
    optimizer = None

    # Training statistics
    training_stats: dict[str, int | float | bool | None] = {
        "total_steps": 0,
        "final_loss": None,
        "interrupted": False,
        "completed_epochs": 0,
    }

    try:
        # Setup all training components
        model, diffusion, options, optimizer, dataloader, checkpoint_manager, scheduler = \
            setup_training(config, strategy)

        # Load checkpoint if resuming
        start_epoch, global_step = load_training_state(
            config, checkpoint_manager, model, optimizer
        )
        training_stats["total_steps"] = global_step

        # Create output directory
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)

        # Training loop
        logger.info("\nStarting training...")
        logger.info(f"  Epochs: {config.training.num_epochs}")
        logger.info(f"  Batch size: {config.data.batch_size}")
        logger.info(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: "
                   f"{config.data.batch_size * config.training.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {config.training.learning_rate}")

        for epoch in range(start_epoch, config.training.num_epochs):
            if interrupt_handler.interrupted:
                logger.info("Training interrupted, saving checkpoint...")
                if checkpoint_manager:
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, global_step, is_interrupted=True
                    )
                training_stats["interrupted"] = True
                break

            logger.info(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")

            # Train for one epoch
            epoch_losses, global_step, interrupted = train_epoch(
                epoch, model, diffusion, dataloader, optimizer, strategy,
                config, global_step, scheduler, checkpoint_manager,
                interrupt_handler, eval_prompts, grid_size, output_dir, wandb_run
            )

            training_stats["total_steps"] = global_step
            training_stats["completed_epochs"] = epoch + 1

            if epoch_losses:
                training_stats["final_loss"] = float(np.mean(epoch_losses[-100:]))

            # Log epoch summary
            log_epoch_summary(epoch, epoch_losses, global_step, wandb_run)

            if interrupted:
                training_stats["interrupted"] = True
                break

        # Save final checkpoint
        save_final_checkpoint(
            checkpoint_manager, model, optimizer,
            config.training.num_epochs, global_step,
            interrupt_handler.interrupted
        )

        # Log final summary
        logger.info("\n" + "=" * 80)
        logger.info("Training complete!")
        logger.info(f"  Total steps: {training_stats['total_steps']:,}")
        if training_stats["final_loss"] is not None:
            logger.info(f"  Final loss: {training_stats['final_loss']:.4f}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nError during training: {e}")
        traceback.print_exc()

        # Save emergency checkpoint
        if checkpoint_manager and model and optimizer:
            logger.info("Saving emergency checkpoint...")
            try:
                # Get current epoch and step from stats
                completed_epochs = training_stats.get("completed_epochs", 0)
                total_steps = training_stats.get("total_steps", 0)
                epoch = int(completed_epochs) if isinstance(completed_epochs, int | float) else 0
                global_step = int(total_steps) if isinstance(total_steps, int | float) else 0

                checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, global_step, is_interrupted=True
                )
                logger.info("Emergency checkpoint saved.")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")

        raise

    finally:
        # Cleanup is handled by caller (wandb finish, etc.)
        pass

    return training_stats
