#!/usr/bin/env python3
"""
GLIDE FP16 Training Script
Production-ready mixed precision training with comprehensive stability features.
"""

import argparse
import os
import sys
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'glide-text2im'))

from glide_finetune.glide_util import load_model
from glide_finetune.glide_util import sample
from glide_finetune.loader import TextImageDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader
from glide_finetune.checkpoint_manager import CheckpointManager
from glide_finetune.fp16_training import (
    FP16TrainingConfig,
    FP16TrainingStep,
    SelectiveFP16Converter,
)
from glide_finetune.dynamic_loss_scaler import DynamicLossScaler, LossScalerConfig
from glide_finetune.master_weight_manager import MasterWeightManager


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def setup_seed(seed: int = None) -> int:
    """Setup seeds for reproducible training."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"No seed specified, using random seed: {seed}")
    else:
        print(f"Using seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if seed != 0:
        print("⚠️  Enabling deterministic mode for reproducibility")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using seed 0 - performance mode enabled")
        torch.backends.cudnn.benchmark = True
    
    return seed


def train_step_fp16(
    model: nn.Module,
    diffusion,
    batch,
    trainer: FP16TrainingStep,
    device: torch.device,
    uncond_p: float = 0.0,
) -> dict:
    """
    Perform a single FP16 training step.
    """
    # Unpack batch - WebDataset returns tuples
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            tokens, masks, images = batch
        elif len(batch) == 4:
            tokens, masks, images, _ = batch  # Ignore upsampled for base model
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
    else:
        # Handle dict format for regular dataset
        images = batch['images']
        tokens = batch.get('tokens')
        masks = batch.get('masks')
    
    # Move to device - keep everything in FP32
    images = images.to(device).float()
    
    if tokens is not None:
        tokens = tokens.to(device)
        masks = masks.to(device) if masks is not None else None
        # Apply unconditional training
        if uncond_p > 0:
            mask = torch.rand(images.shape[0], device=device) < uncond_p
            tokens = tokens.clone()
            tokens[mask] = 0  # Zero out tokens for unconditional training
    
    # Define loss computation function
    def compute_loss():
        # Sample timesteps
        timesteps = torch.randint(
            0, len(diffusion.betas) - 1, (images.shape[0],), device=device
        )
        
        # Add noise to images (in FP32)
        noise = torch.randn_like(images, device=device)
        x_t = diffusion.q_sample(images, timesteps, noise=noise).to(device)
        
        # Get model prediction
        _, C = x_t.shape[:2]
        model_output = model(
            x_t,
            timesteps,
            tokens=tokens if tokens is not None else None,
            mask=masks if masks is not None else None,
        )
        
        # Split output and compute loss
        epsilon, _ = torch.split(model_output, C, dim=1)
        return torch.nn.functional.mse_loss(epsilon, noise.detach())
    
    # Use FP16 trainer for the step
    result = trainer.training_step(compute_loss)
    
    return result


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GLIDE FP16 Training")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Model arguments
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "upsample"])
    parser.add_argument("--uncond_p", type=float, default=0.2)
    parser.add_argument("--freeze_transformer", action="store_true")
    parser.add_argument("--freeze_diffusion", action="store_true")
    
    # FP16 arguments
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 training")
    parser.add_argument("--fp16_mode", type=str, default="auto", 
                       choices=["auto", "conservative", "aggressive"],
                       help="FP16 conversion mode")
    parser.add_argument("--fp16_loss_scale", type=float, default=256.0)
    parser.add_argument("--fp16_grad_clip", type=float, default=1.0)
    parser.add_argument("--use_master_weights", action="store_true", default=True)
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps. Effective batch size = batch_size * gradient_accumulation_steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Logging arguments
    parser.add_argument("--log_frequency", type=int, default=100)
    parser.add_argument("--sample_frequency", type=int, default=500)
    parser.add_argument("--save_frequency", type=int, default=1000)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_fp16")
    parser.add_argument("--project_name", type=str, default="glide_fp16")
    
    # WebDataset arguments
    parser.add_argument("--use_webdataset", action="store_true")
    parser.add_argument("--wds_dataset_name", type=str, default="laion")
    parser.add_argument("--wds_image_key", type=str, default="jpg")
    parser.add_argument("--wds_caption_key", type=str, default="txt")
    
    # Other arguments
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--test_prompt", type=str, 
                       default="a beautiful sunset over mountains")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("="*80)
    logger.info("GLIDE FP16 TRAINING")
    logger.info("="*80)
    
    # Setup seed
    seed = setup_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=args.resume_ckpt if args.resume_ckpt else None,
        use_fp16=False,  # We'll convert manually
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        activation_checkpointing=args.activation_checkpointing,
        model_type=args.model_type,
    )
    
    # Move model to device
    glide_model = glide_model.to(device)
    
    # Don't convert to FP16 here - keep model in FP32
    # The FP16TrainingStep will handle mixed precision training
    logger.info("Model will remain in FP32 for mixed precision training")
    
    glide_model.train()
    
    # Print model info
    total_params = sum(p.numel() for p in glide_model.parameters())
    trainable_params = sum(p.numel() for p in glide_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    base_optimizer = torch.optim.AdamW(
        glide_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Create FP16 trainer with gradient accumulation
    fp16_config = FP16TrainingConfig(
        use_loss_scaling=args.use_fp16,
        init_loss_scale=args.fp16_loss_scale,
        use_master_weights=args.use_fp16 and args.use_master_weights,
        gradient_clip_norm=args.fp16_grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_frequency=args.log_frequency,
        enable_nan_recovery=args.use_fp16,
    )
    
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    trainer = FP16TrainingStep(glide_model, base_optimizer, fp16_config)
    
    # Setup data loader
    logger.info("Setting up data loader...")
    if args.use_webdataset:
        dataset = glide_wds_loader(
            urls=args.data_dir,
            dataset_name=args.wds_dataset_name,
            image_key=args.wds_image_key,
            caption_key=args.wds_caption_key,
            base_x=64,
            base_y=64,
            tokenizer=glide_model.tokenizer,
            enable_text=True,
            enable_image=True,
            uncond_p=args.uncond_p,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        # Count number of tar files to estimate dataset size
        import glob
        tar_files = glob.glob(args.data_dir)
        num_tars = len(tar_files)
        data_len = num_tars * 10000  # 10k images per tar estimate
        logger.info(f"Found {num_tars} tar files, estimated {data_len:,} total images")
    else:
        dataset = TextImageDataset(
            args.data_dir,
            side_x=64,
            side_y=64,
            resize_ratio=1.0,
            use_captions=True,
            uncond_p=args.uncond_p,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        data_len = len(dataset)
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoints_dir=args.checkpoints_dir,
        save_frequency=args.save_frequency,
    )
    
    # Setup wandb (optional)
    try:
        wandb_run = wandb_setup(
            batch_size=args.batch_size,
            side_x=64,
            side_y=64,
            learning_rate=args.learning_rate,
            use_fp16=args.use_fp16,
            device=device,
            data_dir=args.data_dir,
            base_dir=args.checkpoints_dir,
            project_name=args.project_name,
        )
    except Exception as e:
        logger.warning(f"WandB setup failed: {e}. Continuing without wandb logging.")
        wandb_run = None
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    accumulation_step = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_losses = []
        
        progress_bar = trange(
            data_len // args.batch_size,  # Full epoch
            desc=f"Epoch {epoch + 1}"
        )
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step (handles gradient accumulation internally)
            result = train_step_fp16(
                model=glide_model,
                diffusion=glide_diffusion,
                batch=batch,
                trainer=trainer,
                device=device,
                uncond_p=args.uncond_p,
            )
            
            accumulation_step += 1
            
            if not np.isnan(result['loss']):
                epoch_losses.append(result['loss'])
            
            # Update progress bar - show both actual and effective batch
            progress_bar.set_postfix({
                'loss': result['loss'],
                'grad_norm': result.get('grad_norm', 0),
                'loss_scale': result.get('loss_scale', 1.0),
                'acc_step': f"{accumulation_step % args.gradient_accumulation_steps}/{args.gradient_accumulation_steps}",
            })
            progress_bar.update(1)
            
            # Logging
            if wandb_run and global_step % args.log_frequency == 0:
                wandb_run.log({
                    'loss': result['loss'],
                    'grad_norm': result['grad_norm'],
                    'loss_scale': result['loss_scale'],
                    'learning_rate': args.learning_rate,
                    'epoch': epoch,
                    'step': global_step,
                })
            
            # Save checkpoint
            if global_step % args.save_frequency == 0 and global_step > 0:
                logger.info(f"Saving checkpoint at step {global_step}...")
                checkpoint_data = {
                    'model': glide_model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'trainer_state': trainer.state_dict(),
                    'fp16_config': fp16_config.__dict__,
                }
                
                checkpoint_manager.save_checkpoint(
                    checkpoint_data,
                    step=global_step,
                    metrics={'loss': np.mean(epoch_losses[-100:]) if epoch_losses else 0},
                )
            
            global_step += 1
            
            # Break for webdataset after reasonable number of steps
            if args.use_webdataset and batch_idx >= 1000:
                break
        
        progress_bar.close()
        
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Log trainer statistics
        trainer._log_statistics()
    
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Final loss: {avg_epoch_loss:.4f}")
    
    if args.use_fp16:
        success_rate = trainer.stats['successful_steps'] / trainer.stats['total_steps'] * 100
        logger.info(f"  FP16 success rate: {success_rate:.1f}%")
        logger.info(f"  NaN recoveries: {trainer.stats['nan_recoveries']}")
    
    logger.info("="*80)
    
    # Clean up
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()