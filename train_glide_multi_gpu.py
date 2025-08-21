#!/usr/bin/env python3
"""
Multi-GPU training script for GLIDE fine-tuning using Hugging Face Accelerate.

Supports:
- Data Parallel (DP) and Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP) for large model training
- DeepSpeed integration for advanced optimization
- Mixed precision training (FP16/BF16)
- Gradient accumulation and checkpointing
"""

import argparse
import glob
import os
import random
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch as th
import wandb
import PIL.Image
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# TF32 handling
if os.environ.get("GLIDE_ENABLE_TF32"):
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
else:
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False

from glide_finetune.glide_finetune import run_glide_finetune_epoch, base_train_step, upsample_train_step, create_image_grid
from glide_finetune import glide_util
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset
from glide_finetune import train_util
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader
from glide_finetune.wds_loader_optimized import glide_wds_loader_optimized
from glide_finetune.checkpoint_manager import CheckpointManager
from glide_finetune.freeze_utils import apply_freeze_policy, build_optimizer_params
from glide_finetune.metrics_tracker import MetricsTracker
from glide_finetune.fp16_training import (
    FP16TrainingConfig,
    FP16TrainingStep,
    SelectiveFP16Converter,
)


def get_warmup_scheduler(
    optimizer,
    warmup_steps: int,
    warmup_start_lr: float = 7e-7,
    target_lr: float = 1e-5,
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup
        target_lr: Target learning rate after warmup
        
    Returns:
        LambdaLR scheduler with warmup
    """
    def lr_lambda(current_step: int):
        if warmup_steps == 0:
            return 1.0
            
        if current_step < warmup_steps:
            # Linear warmup
            progress = float(current_step) / float(max(1, warmup_steps))
            actual_lr = warmup_start_lr + progress * (target_lr - warmup_start_lr)
            return actual_lr / target_lr
        else:
            # After warmup, use full learning rate
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def setup_accelerator(args) -> Accelerator:
    """Initialize and configure the Accelerator for distributed training."""
    
    # Configure project for logging and checkpointing
    project_config = ProjectConfiguration(
        project_dir=args.checkpoints_dir,
        automatic_checkpoint_naming=False,  # We handle our own checkpoint naming
        total_limit=5,  # Keep only 5 most recent checkpoints
    )
    
    # Determine mixed precision mode
    mixed_precision = None
    if args.use_fp16:
        mixed_precision = "fp16"
    elif args.use_bf16:
        mixed_precision = "bf16"
    
    # Initialize accelerator with configuration
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if not args.no_wandb else None,
        project_config=project_config,
        step_scheduler_with_optimizer=False,  # We handle scheduler manually
    )
    
    # Initialize wandb tracking if enabled
    if accelerator.is_main_process and not args.no_wandb:
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={"wandb": {"dir": args.checkpoints_dir}}
        )
    
    return accelerator


def create_distributed_dataloader(
    accelerator: Accelerator,
    args,
    glide_model=None,
) -> DataLoader:
    """Create a DataLoader that works with distributed training."""
    
    if args.use_webdataset:
        # WebDataset for large-scale training
        if args.use_optimized_loader:
            from glide_finetune.wds_loader_optimized import create_optimized_dataloader
            
            # Need bloom filter path for optimized loader
            bloom_filter_path = args.bloom_filter_path if hasattr(args, 'bloom_filter_path') else None
            if not bloom_filter_path:
                # Fall back to standard loader if no bloom filter
                print("Warning: No bloom filter path provided, falling back to distributed WebDataset loader")
                args.use_optimized_loader = False
            else:
                loader = create_optimized_dataloader(
                    urls=args.data_dir,
                    bloom_filter_path=bloom_filter_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    tokenizer=glide_model.tokenizer if glide_model else None,
                    # Parameters for glide_wds_loader_optimized
                    base_x=args.side_x,
                    base_y=args.side_y,
                    uncond_p=args.uncond_p,
                    image_key=args.image_key,
                    caption_key=args.caption_key,
                    dataset_name=args.wds_dataset_name,
                    enable_upsample=args.train_upsample,
                    enable_text=True,
                )
        
        # If not using optimized loader or fallback from optimized loader
        if not args.use_optimized_loader:
            # For distributed training, use the distributed WebDataset loader
            from glide_finetune.wds_loader_distributed import distributed_wds_loader
            
            loader = distributed_wds_loader(
                urls=args.data_dir,
                batch_size=args.batch_size,
                side_x=args.side_x,
                side_y=args.side_y,
                resize_ratio=getattr(args, 'resize_ratio', 1.0),
                uncond_p=args.uncond_p,
                image_key=args.image_key,
                caption_key=args.caption_key,
                enable_metadata=True,
                wds_dataset_name=args.wds_dataset_name,
                enable_upsample=args.train_upsample,
                upscale_factor=args.upscale_factor if args.train_upsample else 4,
                world_size=accelerator.num_processes,
                rank=accelerator.process_index,
                num_workers=args.num_workers,
                seed=getattr(args, 'seed', 0),
                epoch_samples=getattr(args, 'epoch_samples', None),
                tokenizer=glide_model.tokenizer if glide_model else None,
                trim_white_padding=getattr(args, 'trim_white_padding', False),
                white_thresh=getattr(args, 'white_thresh', 245),
                use_augmentations=getattr(args, 'use_augmentations', True),
            )
    else:
        # Local dataset
        dataset = TextImageDataset(
            folder=args.data_dir,
            side_x=args.side_x,
            side_y=args.side_y,
            resize_ratio=getattr(args, 'resize_ratio', 1.0),
            uncond_p=args.uncond_p,
            flip_p=0.5,
            use_captions=args.use_captions,
            upscale_factor=args.upscale_factor if args.train_upsample else 4,
            enable_glide_upsample=args.train_upsample,
            tokenizer=glide_model.tokenizer if glide_model else None,
            text_ctx_len=128,  # Default text context length
            use_augmentations=getattr(args, 'use_augmentations', True),
            dataset_type="local"  # Local dataset type
        )
        
        # Create DataLoader with DistributedSampler handled by Accelerate
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,  # Important for distributed training
        )
    
    return loader


def generate_samples_distributed(
    accelerator,
    glide_model,
    glide_diffusion,
    glide_options,
    test_prompts,
    args,
    step
):
    """Generate samples for logging during distributed training."""
    sample_images = []
    wandb_gallery_images = []
    
    # Only generate samples on main process
    if accelerator.is_main_process:
        glide_model.eval()
        
        for i, prompt in enumerate(test_prompts):
            if i >= 32:  # Limit to prevent memory issues
                break
                
            with th.no_grad():
                samples = glide_util.sample(
                    glide_model=glide_model,
                    glide_diffusion=glide_diffusion,
                    glide_options=glide_options,
                    prompt=prompt,
                    batch_size=1,  # Small batch size for sampling
                    guidance_scale=3.0,
                    device=accelerator.device,
                    prediction_respacing="30",
                    sampler="euler",
                    num_steps=30,
                )
                sample_img = train_util.pred_to_pil(samples)
                sample_images.append(sample_img)
                wandb_gallery_images.append(wandb.Image(sample_img, caption=prompt))
    
    glide_model.train()
    
    # Create grid - automatically adjust rows/cols based on number of images
    if len(sample_images) > 0:
        num_images = len(sample_images)
        
        # Calculate optimal grid dimensions
        if num_images <= 4:
            grid_rows, grid_cols = 1, num_images
        elif num_images <= 8:
            grid_rows, grid_cols = 2, 4
        elif num_images <= 16:
            grid_rows, grid_cols = 4, 4
        elif num_images <= 32:
            grid_rows, grid_cols = 8, 4  # 8x4 grid for 32 images
        else:
            # For more than 32, create an 8x8 grid (max 64)
            grid_rows, grid_cols = 8, 8
        
        # Ensure we don't exceed available images
        images_to_use = min(grid_rows * grid_cols, num_images)
        grid_img = create_image_grid(sample_images[:images_to_use], rows=grid_rows, cols=grid_cols)
        
        return {
            "samples_grid": wandb.Image(grid_img, caption=f"Step {step} - {images_to_use} Samples"),
            "samples_gallery": wandb_gallery_images
        }
    return None


class InterruptHandler:
    """Handle interrupts and emergency checkpoint saving."""
    
    def __init__(self):
        self.interrupted = False
        self.original_sigint = signal.signal(signal.SIGINT, self.handle_sigint)
        
    def handle_sigint(self, signum, frame):
        """Handle CTRL-C interrupt."""
        if self.interrupted:
            # Second interrupt - force exit
            print("\n\n⚠️  Force exit requested. Exiting immediately...")
            signal.signal(signal.SIGINT, self.original_sigint)
            sys.exit(1)
        
        self.interrupted = True
        print("\n\n⚠️  Training interrupted!")
        print("Press CTRL-C again to force exit, or wait for checkpoint prompt...")
    
    def reset(self):
        """Reset interrupt flag after handling."""
        self.interrupted = False
    
    def __del__(self):
        """Restore original signal handler."""
        signal.signal(signal.SIGINT, self.original_sigint)


def save_emergency_checkpoint(
    accelerator: Accelerator,
    checkpoint_manager: Optional[CheckpointManager],
    model,
    optimizer,
    epoch: int,
    step: int,
    args,
    reason: str = "interrupt"
):
    """Save an emergency checkpoint when training is interrupted."""
    
    if accelerator.is_main_process:
        print(f"\n💾 Saving emergency checkpoint (reason: {reason})...")
        
        try:
            # Create emergency checkpoint directory
            emergency_dir = Path(args.checkpoints_dir) / f"emergency_{reason}_{step}"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            
            # Save accelerate state
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=str(emergency_dir))
            
            # Save checkpoint manager state
            if checkpoint_manager:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    unwrapped_model,
                    optimizer,
                    epoch,
                    step
                )
                print(f"✓ Emergency checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to save emergency checkpoint: {e}")
            traceback.print_exc()


def run_distributed_training(args):
    """Main distributed training loop."""
    
    # Initialize accelerator
    accelerator = setup_accelerator(args)
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Only main process should print
    if accelerator.is_main_process:
        print(f"🚀 Starting distributed training on {accelerator.num_processes} GPUs")
        print(f"   Mixed precision: {accelerator.mixed_precision}")
        print(f"   Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    
    # Load model and prepare for distributed training
    if accelerator.is_main_process:
        print("Loading model...")
    
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=args.model_path if args.model_path else None,
        use_fp16=False,  # Accelerate handles mixed precision
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        activation_checkpointing=args.activation_checkpointing,
        model_type="base" if not args.train_upsample else "upsample",
    )
    
    # Apply freeze policy if needed
    if args.freeze_transformer or args.freeze_diffusion:
        freeze_summary = apply_freeze_policy(
            glide_model,
            freeze_transformer=args.freeze_transformer,
            freeze_diffusion=args.freeze_diffusion
        )
        if accelerator.is_main_process:
            print(f"\n{freeze_summary}\n")
    
    # Create optimizer with proper parameter groups
    optimizer_params = build_optimizer_params(
        glide_model,
        weight_decay=args.adam_weight_decay
    )
    
    optimizer = th.optim.AdamW(
        optimizer_params,
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
        eps=1e-8
    )
    
    # Create data loader
    dataloader = create_distributed_dataloader(accelerator, args)
    
    # Prepare for distributed training
    # This wraps the model with DDP/FSDP and handles device placement
    glide_model, optimizer, dataloader = accelerator.prepare(
        glide_model, optimizer, dataloader
    )
    
    # Initialize checkpoint manager (only on main process)
    checkpoint_manager = None
    if accelerator.is_main_process:
        checkpoint_manager = CheckpointManager(
            checkpoints_dir=args.checkpoints_dir,
            save_frequency=args.save_frequency
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_ckpt:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_ckpt}")
        
        # Load checkpoint using Accelerate's method for distributed models
        accelerator.load_state(args.resume_ckpt)
        
        # Get training state from checkpoint
        checkpoint_path = Path(args.resume_ckpt)
        if checkpoint_path.is_dir():
            state_files = list(checkpoint_path.glob("*_state*.json"))
            if state_files:
                import json
                with open(state_files[0], 'r') as f:
                    state = json.load(f)
                    start_epoch = state.get('epoch', 0)
                    global_step = state.get('global_step', 0)
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        if accelerator.is_main_process:
            print(f"\n📊 Epoch {epoch}/{args.num_epochs}")
        
        # Set epoch for proper shuffling in distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        # Progress bar only on main process
        progress_bar = None
        if accelerator.is_main_process:
            progress_bar = tqdm(
                total=len(dataloader),
                desc=f"Epoch {epoch}",
                unit="batch"
            )
        
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(glide_model):
                # Forward pass
                if args.train_upsample:
                    # Upsampling expects (tokens, masks, low_res, high_res)
                    tokens, masks, low_res, high_res = batch
                    loss = glide_diffusion.training_losses(
                        glide_model,
                        high_res,
                        t=None,
                        model_kwargs={
                            "tokens": tokens,
                            "mask": masks,
                            "low_res": low_res
                        }
                    )["loss"].mean()
                else:
                    # Base model expects (tokens, masks, images)
                    tokens, masks, images = batch
                    loss = glide_diffusion.training_losses(
                        glide_model,
                        images,
                        t=None,
                        model_kwargs={
                            "tokens": tokens,
                            "mask": masks
                        }
                    )["loss"].mean()
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping if specified
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(glide_model.parameters(), args.grad_clip)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Logging
            if global_step % args.log_frequency == 0:
                # Properly average loss across all processes with correct weighting
                # Use reduce instead of gather for proper averaging
                avg_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                
                if accelerator.is_main_process:
                    print(f"Step {global_step}: loss = {avg_loss:.4f}")
                    
                    # Log to wandb
                    if not args.no_wandb:
                        accelerator.log({
                            "train/loss": avg_loss,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                        }, step=global_step)
            
            # Checkpointing (only on main process)
            if checkpoint_manager and checkpoint_manager.should_save(global_step):
                if accelerator.is_main_process:
                    print(f"💾 Saving checkpoint at step {global_step}")
                
                # Wait for all processes to reach this point
                accelerator.wait_for_everyone()
                
                # Save using Accelerate's method
                accelerator.save_state(
                    output_dir=str(checkpoint_manager.checkpoints_dir / f"checkpoint_{global_step}")
                )
                
                # Also save in custom format for compatibility
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(glide_model)
                    checkpoint_manager.save_checkpoint(
                        unwrapped_model,
                        optimizer,
                        epoch,
                        global_step
                    )
        
        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()
    
    # Final checkpoint
    if accelerator.is_main_process:
        print("💾 Saving final checkpoint...")
        
    accelerator.wait_for_everyone()
    accelerator.save_state(
        output_dir=str(Path(args.checkpoints_dir) / "final_checkpoint")
    )
    
    if accelerator.is_main_process and checkpoint_manager:
        unwrapped_model = accelerator.unwrap_model(glide_model)
        checkpoint_manager.save_checkpoint(
            unwrapped_model,
            optimizer,
            args.num_epochs,
            global_step
        )
    
    # End training
    if accelerator.is_main_process:
        print("✅ Training complete!")
    
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU GLIDE fine-tuning with Accelerate")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--use_webdataset", action="store_true", help="Use WebDataset format")
    parser.add_argument("--use_optimized_loader", action="store_true", help="Use optimized WebDataset loader")
    parser.add_argument("--wds_dataset_name", type=str, default=None, help="WebDataset name (laion, alamy, synthetic)")
    parser.add_argument("--image_key", type=str, default="jpg", help="WebDataset image key")
    parser.add_argument("--caption_key", type=str, default="txt", help="WebDataset caption key")
    parser.add_argument("--person_filter", type=str, default=None, help="Path to person filter bloom file")
    parser.add_argument("--epoch_samples", type=int, default=None, help="Samples per epoch for WebDataset")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--train_upsample", action="store_true", help="Train upsampler instead of base model")
    parser.add_argument("--freeze_transformer", action="store_true", help="Freeze transformer/text encoder")
    parser.add_argument("--freeze_diffusion", action="store_true", help="Freeze diffusion/UNet")
    parser.add_argument("--activation_checkpointing", action="store_true", help="Enable activation checkpointing")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--warmup_start_lr", type=float, default=7e-7, help="Starting learning rate for warmup")
    parser.add_argument("--skip_optimizer_resume", action="store_true", help="Skip optimizer state when resuming")
    
    # Mixed precision arguments
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--use_bf16", action="store_true", help="Use BF16 mixed precision")
    
    # Image arguments
    parser.add_argument("--side_x", type=int, default=64, help="Image width (before upscaling)")
    parser.add_argument("--side_y", type=int, default=64, help="Image height (before upscaling)")
    parser.add_argument("--resize_ratio", type=float, default=1.0, help="Random crop ratio")
    parser.add_argument("--uncond_p", type=float, default=0.2, help="Unconditional probability")
    parser.add_argument("--use_captions", action="store_true", default=True, help="Use text captions")
    parser.add_argument("--upscale_factor", type=int, default=4, help="Upscaling factor for upsampler")
    parser.add_argument("--class_cond_dropout_prob", type=float, default=0.1, help="Class conditioning dropout")
    parser.add_argument("--trim_white_padding", action="store_true", help="Remove white padding from images")
    parser.add_argument("--white_thresh", type=int, default=245, help="White detection threshold (0-255)")
    parser.add_argument("--use_augmentations", action="store_true", default=True, help="Use data augmentations")
    parser.add_argument("--no_augmentations", action="store_true", help="Disable augmentations")
    
    # Checkpointing arguments
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_frequency", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume from checkpoint")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="glide_multi_gpu", help="W&B project name")
    parser.add_argument("--log_frequency", type=int, default=100, help="Log every N steps")
    parser.add_argument("--sample_frequency", type=int, default=500, help="Generate samples every N steps")
    parser.add_argument("--eval_prompt_file", type=str, default=None, help="File with evaluation prompts")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Handle augmentation flags
    if args.no_augmentations:
        args.use_augmentations = False
    
    # Validate arguments
    if args.freeze_transformer and args.freeze_diffusion:
        parser.error("Cannot freeze both transformer and diffusion")
    
    if args.use_fp16 and args.use_bf16:
        parser.error("Cannot use both FP16 and BF16")
    
    # Run distributed training
    run_distributed_training(args)


if __name__ == "__main__":
    main()