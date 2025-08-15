import os
import sys
from typing import Tuple, List

# Add glide-text2im to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glide-text2im'))

import torch as th
import PIL.Image
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from wandb import wandb

from glide_finetune import glide_util, train_util
from glide_finetune.metrics_tracker import MetricsTracker, print_model_info

# Midjourney-style test prompts for evaluation
DEFAULT_TEST_PROMPTS = [
    "ethereal bioluminescent forest, volumetric fog, mystical atmosphere, fantasy art, octane render, 8k",
    "cyberpunk street market, neon lights, rain reflections, detailed architecture, blade runner aesthetic",
    "ancient ruins overgrown with vines, golden hour lighting, atmospheric perspective, concept art",
    "abstract geometric patterns, vibrant gradients, minimalist design, contemporary digital art",
    "steampunk airship above clouds, brass and copper details, dramatic sunset, highly detailed",
    "underwater coral reef city, rays of sunlight, vibrant colors, fantastical architecture",
    "crystalline ice cave, prismatic light refraction, frozen waterfalls, magical realism",
    "desert oasis at twilight, palm trees silhouette, starry sky, cinematic lighting"
]

def create_image_grid(images: List[PIL.Image.Image], rows: int = 2, cols: int = 4) -> PIL.Image.Image:
    """Create a grid of images for wandb logging."""
    if not images:
        return None
    
    # Get dimensions from first image
    width, height = images[0].size
    
    # Create new image for grid
    grid_width = width * cols
    grid_height = height * rows
    grid_img = PIL.Image.new('RGB', (grid_width, grid_height))
    
    # Paste images into grid
    for idx, img in enumerate(images[:rows*cols]):
        row = idx // cols
        col = idx % cols
        grid_img.paste(img, (col * width, row * height))
    
    return grid_img

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
    sample_respacing: str = '100', # respacing for inference
    prompt: str = "",  # prompt for inference, not training (deprecated, use test_prompts)
    test_prompts: List[str] = None,  # list of prompts for evaluation
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,  # console logging frequency (loss, metrics)
    sample_frequency: int = 500,  # image generation frequency
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    global_step: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    sampler: str = "plms",
    num_steps: int = None,
    eta: float = 0.0,
    checkpoint_manager=None,
):
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step
    
    # Use default prompts if none provided
    if test_prompts is None:
        test_prompts = DEFAULT_TEST_PROMPTS
    elif isinstance(test_prompts, str):
        # Handle backward compatibility with single prompt
        test_prompts = [test_prompts] * 8
    
    # Ensure we have exactly 8 prompts for 2x4 grid
    while len(test_prompts) < 8:
        test_prompts.append(test_prompts[-1] if test_prompts else DEFAULT_TEST_PROMPTS[0])
    test_prompts = test_prompts[:8]

    glide_model.to(device)
    glide_model.train()
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(window_size=log_frequency)
    
    # Print model info once
    print_model_info(glide_model, "GLIDE Model")
    
    log = {}
    for train_idx, batch in enumerate(dataloader):
        # Check for interrupt signal
        if checkpoint_manager and checkpoint_manager.interrupted:
            print("\n⚠️  Interrupt detected in training loop, stopping...")
            break
        
        current_step = global_step + train_idx
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        accumulated_loss.backward()
        
        # Update gradient norm before optimizer step
        metrics_tracker.update_grad_norm(glide_model)
        
        optimizer.step()
        glide_model.zero_grad()
        
        # Update metrics
        metrics_tracker.update_loss(accumulated_loss.item())
        metrics_tracker.update_timing()
        metrics_tracker.update_batch_size(len(batch[0]) if isinstance(batch, tuple) else 1)
        
        # Get current learning rate (if using a scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        metrics_tracker.update_lr(current_lr)
        
        log = {**log, "iter": train_idx, "loss": accumulated_loss.item() / gradient_accumualation_steps, "global_step": current_step}
        
        # Console logging (loss, metrics, etc.)
        if train_idx > 0 and train_idx % log_frequency == 0:
            console_summary = metrics_tracker.get_console_summary()
            print(f"[Epoch {epoch} | Iter {train_idx}] {console_summary}")
            
            # Get all metrics for wandb
            all_metrics = metrics_tracker.get_metrics()
            all_metrics.update({"iter": train_idx, "epoch": epoch})
            
            # Log metrics to wandb (no images)
            wandb_run.log(all_metrics)
        
        # Sample from the model (image generation)
        if train_idx > 0 and train_idx % sample_frequency == 0:
            sample_metrics = metrics_tracker.get_metrics()
            print(f"\n🎨 [Iter {train_idx}] Generating sample images using {sampler} sampler...")
            print(f"   📈 Current metrics: Loss={sample_metrics.get('loss', 0):.4f}, Steps/s={sample_metrics.get('steps_per_sec', 0):.2f}")
            sampling_start_time = th.cuda.Event(enable_timing=True) if device == 'cuda' else None
            sampling_end_time = th.cuda.Event(enable_timing=True) if device == 'cuda' else None
            
            if sampling_start_time:
                sampling_start_time.record()
            
            # Generate samples for all test prompts
            sample_images = []
            wandb_gallery_images = []  # For W&B gallery with captions
            
            for prompt_idx, test_prompt in enumerate(test_prompts):
                samples = glide_util.sample(
                    glide_model=glide_model,
                    glide_options=glide_options,
                    side_x=side_x,
                    side_y=side_y,
                    prompt=test_prompt,
                    batch_size=sample_bs,
                    guidance_scale=sample_gs,
                    device=device,
                    prediction_respacing=sample_respacing,
                    image_to_upsample=image_to_upsample,
                    sampler=sampler,
                    num_steps=num_steps,
                    eta=eta,
                )
                sample_img = train_util.pred_to_pil(samples)
                sample_images.append(sample_img)
                
                # Save individual samples
                individual_save_path = os.path.join(outputs_dir, f"iter{train_idx:06d}_prompt{prompt_idx}.png")
                sample_img.save(individual_save_path)
                
                # Add to gallery with caption (W&B feature)
                wandb_gallery_images.append(wandb.Image(sample_img, caption=test_prompt))
            
            # Create and save grid (no visible captions)
            grid_img = create_image_grid(sample_images, rows=2, cols=4)
            grid_save_path = os.path.join(outputs_dir, f"iter{train_idx:06d}_grid.png")
            grid_img.save(grid_save_path)
            
            # Log to wandb with both grid and gallery
            wandb_log_dict = {
                **log,
                "iter": train_idx,
                "samples_grid": wandb.Image(grid_img, caption="2x4 Grid of Test Samples"),
                "samples_gallery": wandb_gallery_images,  # This creates a gallery with individual captions
            }
            
            wandb_run.log(wandb_log_dict)
            if sampling_end_time:
                sampling_end_time.record()
                th.cuda.synchronize()
                sampling_time = sampling_start_time.elapsed_time(sampling_end_time) / 1000.0  # Convert to seconds
                print(f"✓ Generated {len(sample_images)} samples in {sampling_time:.2f}s, saved grid to {grid_save_path}")
                # Log sampling time to wandb
                wandb_log_dict["sampling_time_sec"] = sampling_time
                wandb_log_dict["samples_per_sec_generation"] = len(sample_images) / sampling_time
            else:
                print(f"✓ Generated {len(sample_images)} samples, saved grid to {grid_save_path}")
            print()
        
        # Save checkpoint at regular intervals
        if checkpoint_manager and checkpoint_manager.should_save(current_step):
            checkpoint_manager.save_checkpoint(
                glide_model,
                optimizer,
                epoch,
                current_step,
                is_interrupted=False
            )
        
        wandb_run.log(log)
    
    # Update and return global step
    final_step = global_step + train_idx + 1
    
    # Save end-of-epoch checkpoint
    if checkpoint_manager and not checkpoint_manager.interrupted:
        print(f"Finished epoch {epoch}, saving checkpoint")
        checkpoint_manager.save_checkpoint(
            glide_model,
            optimizer,
            epoch,
            final_step,
            is_interrupted=False
        )
    
    return final_step
