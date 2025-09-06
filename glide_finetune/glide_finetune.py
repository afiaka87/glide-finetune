import os
import random
import time
from typing import Tuple

import numpy as np
import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from wandb import wandb

from glide_finetune import glide_util, train_util

def base_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, reals = [x.to(device) for x in batch]
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())

def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where 
                - tokens is a tensor of shape (batch_size, seq_len), 
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image, high_res_image = [ x.to(device) for x in batch ]
    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(high_res_image, device=device) # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device))
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    sample_bs: int,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = '30', # respacing for inference - using 30 steps with Euler
    sample_sampler: str = 'euler',  # sampler for inference - Euler is fast and deterministic
    eval_base_sampler: str = 'euler',  # sampler for base model evaluation
    eval_sr_sampler: str = 'euler',  # sampler for super-resolution evaluation
    eval_base_sampler_steps: int = 30,  # number of steps for base model evaluation
    eval_sr_sampler_steps: int = 17,  # number of steps for super-resolution evaluation
    prompt: str = "",  # prompt for inference, not training
    sample_captions_file: str = "eval_captions.txt",  # file with captions to sample from
    num_captions_sample: int = 1,  # number of captions to sample and generate
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    sample_interval: int = 500,
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    upsampler_model=None,
    upsampler_options=None,
    use_sr_eval: bool = False,
    use_lora: bool = False,
    lora_save_steps: int = 1000,
    save_checkpoint_interval: int = 5000,
):
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step

    # Load eval captions if available
    eval_captions = []
    if os.path.exists(sample_captions_file):
        with open(sample_captions_file, 'r') as f:
            eval_captions = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(eval_captions)} eval captions from {sample_captions_file}")
    else:
        print(f"No {sample_captions_file} found, using fixed prompt: {prompt}")

    glide_model.to(device)
    glide_model.train()
    log = {}
    
    # Initialize timing for samples/sec calculation
    start_time = time.time()
    last_log_time = start_time
    samples_processed = 0
    
    for train_idx, batch in enumerate(dataloader):
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        accumulated_loss.backward()
        optimizer.step()
        glide_model.zero_grad()
        
        # Calculate samples per second
        batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
        samples_processed += batch_size
        
        # Calculate instantaneous and average samples/sec
        current_time = time.time()
        total_elapsed = current_time - start_time
        avg_samples_per_sec = samples_processed / total_elapsed if total_elapsed > 0 else 0
        
        log = {
            **log, 
            "iter": train_idx, 
            "loss": accumulated_loss.item() / gradient_accumualation_steps,
            "samples_per_sec": avg_samples_per_sec,
            "total_samples": samples_processed,
        }
        
        # Log metrics to wandb every step
        wandb_run.log(log)
        
        # Print to console at log_frequency
        if train_idx > 0 and train_idx % log_frequency == 0:
            # Calculate interval samples/sec since last log
            interval_time = current_time - last_log_time
            interval_samples = batch_size * log_frequency
            interval_samples_per_sec = interval_samples / interval_time if interval_time > 0 else 0
            
            # Add interval samples/sec to wandb log
            wandb_run.log({"interval_samples_per_sec": interval_samples_per_sec})
            
            print(f"Step {train_idx}: loss: {accumulated_loss.item():.4f}, "
                  f"samples/sec: {interval_samples_per_sec:.1f} (interval), "
                  f"{avg_samples_per_sec:.1f} (avg), "
                  f"total samples: {samples_processed}")
            
            last_log_time = current_time
        
        # Sample from the model at sample_interval
        if train_idx > 0 and train_idx % sample_interval == 0:
            # Select captions for sampling
            if eval_captions:
                # Sample multiple captions randomly
                n_captions = min(num_captions_sample, len(eval_captions))
                sample_prompts = random.sample(eval_captions, n_captions)
            else:
                # Use the fixed prompt multiple times
                sample_prompts = [prompt] * num_captions_sample
            
            print(f"Sampling {len(sample_prompts)} images from model at iteration {train_idx}")
            
            # Generate images for all prompts
            all_images = []
            wandb_images = []
            
            for idx, sample_prompt in enumerate(sample_prompts):
                print(f"  [{idx+1}/{len(sample_prompts)}] {sample_prompt[:80]}..." if len(sample_prompt) > 80 else f"  [{idx+1}/{len(sample_prompts)}] {sample_prompt}")
                
                # Use full pipeline if requested and we're training base model
                if use_sr_eval and not train_upsample and upsampler_model is not None:
                    # Map 'standard' to appropriate samplers
                    base_sampler = 'plms' if eval_base_sampler == 'standard' else eval_base_sampler
                    sr_sampler = 'plms' if eval_sr_sampler == 'standard' else eval_sr_sampler
                    
                    samples = glide_util.sample_with_superres(
                        base_model=glide_model,
                        base_options=glide_options,
                        upsampler_model=upsampler_model,
                        upsampler_options=upsampler_options,
                        prompt=sample_prompt,
                        batch_size=sample_bs,
                        guidance_scale=sample_gs,
                        device=device,
                        base_respacing=str(eval_base_sampler_steps),
                        upsampler_respacing=str(eval_sr_sampler_steps),
                        upsample_temp=0.997,
                        base_sampler=base_sampler,  # Use specific sampler for base model
                        upsampler_sampler=sr_sampler,  # Use specific sampler for upsampler
                    )
                else:
                    # Use regular sampling
                    # Map 'standard' to appropriate sampler
                    sampler_to_use = 'plms' if eval_base_sampler == 'standard' else eval_base_sampler
                    
                    samples = glide_util.sample(
                        glide_model=glide_model,
                        glide_options=glide_options,
                        side_x=side_x,
                        side_y=side_y,
                        prompt=sample_prompt,
                        batch_size=sample_bs,
                        guidance_scale=sample_gs,
                        device=device,
                        prediction_respacing=str(eval_base_sampler_steps),
                        sampler=sampler_to_use,
                        image_to_upsample=image_to_upsample,
                    )
                
                # Convert to PIL image
                pil_image = train_util.pred_to_pil(samples)
                all_images.append(pil_image)
                
                # Add to wandb gallery
                wandb_images.append(wandb.Image(pil_image, caption=sample_prompt))
            
            # Create and save grid image
            if len(all_images) > 1:
                grid_size = int(np.ceil(np.sqrt(len(all_images))))
                grid_image = train_util.make_grid(all_images, grid_size=grid_size)
                
                # Save grid
                suffix = "_256px" if (use_sr_eval and not train_upsample) else ""
                grid_save_path = os.path.join(outputs_dir, f"{train_idx}_grid{suffix}.png")
                grid_image.save(grid_save_path)
                print(f"Saved grid with {len(all_images)} images to {grid_save_path}")
                
                # Log to wandb with both grid and gallery
                wandb_run.log({
                    "iter": train_idx,
                    f"sample_grid{suffix}": wandb.Image(grid_save_path, caption=f"Grid of {len(all_images)} samples"),
                    f"sample_gallery{suffix}": wandb_images,
                })
            else:
                # Single image case
                suffix = "_256px" if (use_sr_eval and not train_upsample) else ""
                sample_save_path = os.path.join(outputs_dir, f"{train_idx}{suffix}.png")
                all_images[0].save(sample_save_path)
                print(f"Saved sample {sample_save_path}")
                
                wandb_run.log({
                    "iter": train_idx,
                    f"samples{suffix}": wandb_images[0],
                })
        # Save LoRA adapter if enabled and at save interval
        if use_lora and lora_save_steps > 0 and train_idx % lora_save_steps == 0 and train_idx > 0:
            from glide_finetune.lora_wrapper import save_lora_checkpoint
            lora_save_path = os.path.join(checkpoints_dir, f"lora_adapter_{epoch}_{train_idx}")
            save_lora_checkpoint(
                glide_model,
                lora_save_path,
                metadata={
                    "epoch": epoch,
                    "step": train_idx,
                    "loss": accumulated_loss.item(),
                }
            )
            print(f"Saved LoRA adapter to {lora_save_path}")
        
        if save_checkpoint_interval > 0 and train_idx % save_checkpoint_interval == 0 and train_idx > 0:
            if use_lora:
                # For LoRA, save adapter separately
                from glide_finetune.lora_wrapper import save_lora_checkpoint
                lora_save_path = os.path.join(checkpoints_dir, f"lora_checkpoint_{epoch}_{train_idx}")
                save_lora_checkpoint(
                    glide_model,
                    lora_save_path,
                    metadata={
                        "epoch": epoch,
                        "step": train_idx,
                        "loss": accumulated_loss.item(),
                    }
                )
                print(f"Saved LoRA checkpoint to {lora_save_path}")
            else:
                # Save full model if not using LoRA
                train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
                print(
                    f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
                )
        wandb_run.log(log)
    
    print(f"Finished training, saving final checkpoint")
    if use_lora:
        from glide_finetune.lora_wrapper import save_lora_checkpoint
        final_lora_path = os.path.join(checkpoints_dir, f"lora_final_{epoch}")
        save_lora_checkpoint(
            glide_model,
            final_lora_path,
            metadata={
                "epoch": epoch,
                "step": train_idx,
                "final": True,
            }
        )
        print(f"Saved final LoRA adapter to {final_lora_path}")
    else:
        train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
