import os
import signal
import sys
from typing import Any, Dict, Tuple, List

import torch as th
import wandb
import PIL.Image
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet

from glide_finetune import glide_util, train_util
from glide_finetune.checkpoint_utils import CheckpointManager


def prompt_with_timeout(prompt: str, timeout: int = 20, default: bool = True) -> bool:
    """Prompt user with a yes/no question with timeout.
    
    Args:
        prompt: The question to ask
        timeout: Seconds to wait for response
        default: Default response if timeout occurs
        
    Returns:
        True for yes, False for no
    """
    import select
    
    print(f"\n{prompt}")
    print(f"{'[Y/n]' if default else '[y/N]'} (timeout in {timeout}s): ", end='', flush=True)
    
    # Use select to implement timeout on stdin
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    
    if ready:
        try:
            response = sys.stdin.readline().strip().lower()
            if response in ['n', 'no']:
                return False
            elif response in ['y', 'yes', '']:
                return True
            else:
                # Invalid input, use default
                print(f"Invalid input, using default: {'Yes' if default else 'No'}")
                return default
        except:
            # Error reading input, use default
            return default
    else:
        # Timeout occurred
        print(f"\nTimeout - using default: {'Yes' if default else 'No'}")
        return default


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
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape
                (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len)
                and reals is a tensor of shape (batch_size, 3, side_x, side_y)
                normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            A tuple of (loss, metrics_dict) where metrics_dict contains detailed metrics.
    """
    tokens, masks, reals = batch
    tokens = tokens.to(device)
    masks = masks.to(device)
    reals = reals.to(device)

    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, model_output.shape[1] // 2, dim=1)
    loss = th.nn.functional.mse_loss(epsilon, noise.to(device).detach())
    
    # Calculate quartile losses for monitoring
    quartile_bounds = [0, 250, 500, 750, 1000]
    quartile_losses = {}
    
    for i in range(4):
        mask = (timesteps >= quartile_bounds[i]) & (timesteps < quartile_bounds[i+1])
        if mask.any():
            quartile_loss = th.nn.functional.mse_loss(
                epsilon[mask], 
                noise[mask].to(device).detach()
            )
            quartile_losses[f"loss_q{i}"] = quartile_loss.item()
        else:
            quartile_losses[f"loss_q{i}"] = 0.0
    
    return loss, quartile_losses


def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step for the upsampling model.

    Args:
        glide_model: The model to train.
        glide_diffusion: The diffusion to use.
        batch: A tuple of (tokens, masks, low_res, high_res) where tokens is
            a tensor of shape (batch_size, seq_len), masks is a tensor of shape
            (batch_size, seq_len), low_res is a tensor of shape (batch_size, 3,
            side_x, side_y), and high_res is a tensor of shape (batch_size, 3,
            side_x*4, side_y*4).
        device: The device to use for getting model outputs and computing loss.
    Returns:
        A tuple of (loss, metrics_dict) where metrics_dict contains detailed metrics.
    """
    tokens, masks, low_res_image, high_res_image = batch
    tokens = tokens.to(device)
    masks = masks.to(device)
    low_res_image = low_res_image.to(device)
    high_res_image = high_res_image.to(device)

    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device
    )
    noise = th.randn_like(high_res_image, device=device)
    noised_high_res_image = glide_diffusion.q_sample(
        high_res_image, timesteps, noise=noise
    ).to(device)
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, model_output.shape[1] // 2, dim=1)
    loss = th.nn.functional.mse_loss(epsilon, noise.to(device).detach())
    
    # Calculate quartile losses
    quartile_bounds = [0, 250, 500, 750, 1000]
    quartile_losses = {}
    
    for i in range(4):
        mask = (timesteps >= quartile_bounds[i]) & (timesteps < quartile_bounds[i+1])
        if mask.any():
            quartile_loss = th.nn.functional.mse_loss(
                epsilon[mask], 
                noise[mask].to(device).detach()
            )
            quartile_losses[f"loss_q{i}"] = quartile_loss.item()
        else:
            quartile_losses[f"loss_q{i}"] = 0.0
    
    return loss, quartile_losses


def get_warmup_lr(step: int, base_lr: float, warmup_steps: int, warmup_type: str) -> float:
    """Calculate learning rate during warmup period."""
    if warmup_steps == 0 or step >= warmup_steps:
        return base_lr
    
    if warmup_type == "linear":
        return base_lr * (step / warmup_steps)
    elif warmup_type == "cosine":
        import math
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * (1.0 - step / warmup_steps)))
    else:
        return base_lr


def update_metrics(
    log: Dict[str, float],
    accumulated_loss: th.Tensor,
    step_metrics: Dict[str, float],
    global_step: int,
    train_idx: int,
    current_lr: float,
    batch_size: int,
    gradient_accumualation_steps: int,
    glide_model: th.nn.Module,
) -> Dict[str, float]:
    """Update and return training metrics."""
    # Combine all metrics
    log = {
        **log,
        "step": global_step,
        "iter": train_idx,
        "loss": accumulated_loss.item() / gradient_accumualation_steps,
        "lr": current_lr,
        **step_metrics,  # Add all quartile metrics
    }
    
    # Calculate total samples processed
    samples_processed = (global_step + 1) * batch_size
    log["samples_seen"] = samples_processed
    
    # Calculate parameter norm (cheap operation)
    param_norm = 0.0
    for p in glide_model.parameters():
        if p.requires_grad:
            param_norm += p.data.norm(2).item() ** 2
    param_norm = param_norm ** 0.5
    log["param_norm"] = param_norm
    
    return log


def save_checkpoint_with_manager(
    checkpoint_manager: CheckpointManager,
    glide_model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    epoch: int,
    train_idx: int,
    global_step: int,
    warmup_steps: int,
    warmup_type: str,
    base_lr: float,
    checkpoint_type: str = "regular",
) -> None:
    """Save checkpoint using the checkpoint manager."""
    checkpoint_manager.save_checkpoint(
        model=glide_model,
        optimizer=optimizer,
        epoch=epoch,
        step=train_idx,
        global_step=global_step,
        warmup_steps=warmup_steps,
        warmup_type=warmup_type,
        base_lr=base_lr,
        checkpoint_type=checkpoint_type,
    )


def training_loop(
    dataloader: th.utils.data.DataLoader,
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    optimizer: th.optim.Optimizer,
    checkpoint_manager: CheckpointManager,
    train_step_fn: Any,
    config: Dict[str, Any],
) -> int:
    """Main training loop with error handling."""
    train_idx = -1
    current_state = None
    
    def get_current_state():
        return current_state
    
    # Setup SIGINT handler
    checkpoint_manager.setup_sigint_handler(get_current_state)
    
    # Generate initial samples before training starts
    print("\nGenerating initial samples before training...")
    with th.no_grad():
        if config.get("eval_prompts"):
            generate_eval_grid(
                glide_model,
                glide_options,
                config,
                config["eval_prompts"],
                0,  # train_idx = 0 for initial
                config["epoch_offset"],  # global_step
            )
        else:
            generate_sample(
                glide_model,
                glide_options,
                config,
                0,  # train_idx = 0 for initial
                config["epoch_offset"],  # global_step
            )
    
    try:
        for train_idx, batch in enumerate(dataloader):
            # Early stopping check
            if config["early_stop"] > 0 and train_idx >= config["early_stop"]:
                print(f"Early stopping at step {train_idx} (early_stop={config['early_stop']})")
                break
            
            # Calculate global step for warmup
            global_step = config["epoch_offset"] + train_idx
            
            # Apply learning rate warmup
            current_lr = get_warmup_lr(
                global_step, 
                config["base_lr"], 
                config["warmup_steps"], 
                config["warmup_type"]
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Update current state for SIGINT handler
            current_state = {
                "model": glide_model,
                "optimizer": optimizer,
                "epoch": config["epoch"],
                "step": train_idx,
                "global_step": global_step,
                "warmup_steps": config["warmup_steps"],
                "warmup_type": config["warmup_type"],
                "base_lr": config["base_lr"],
            }
            
            # Training step
            accumulated_loss, step_metrics = train_step_fn(
                glide_model=glide_model,
                glide_diffusion=glide_diffusion,
                batch=batch,
                device=config["device"],
            )
            accumulated_loss.backward()
            optimizer.step()
            glide_model.zero_grad()
            
            # Update metrics
            config["log"] = update_metrics(
                config["log"],
                accumulated_loss,
                step_metrics,
                global_step,
                train_idx,
                current_lr,
                config["batch_size"],
                config["gradient_accumualation_steps"],
                glide_model,
            )
            
            # Log metrics to wandb
            config["wandb_run"].log(config["log"])
            
            # Console output at log_frequency intervals
            if train_idx > 0 and train_idx % config["log_frequency"] == 0:
                print_metrics(
                    config["log"], 
                    step_metrics, 
                    global_step, 
                    accumulated_loss, 
                    current_lr,
                    config["warmup_steps"],
                    config["first_log"],
                )
                config["first_log"] = False
            
            # Sample generation
            if global_step > 0 and global_step % config["sample_interval"] == 0:
                if config.get("eval_prompts"):
                    generate_eval_grid(
                        glide_model,
                        glide_options,
                        config,
                        config["eval_prompts"],
                        train_idx,
                        global_step,
                    )
                else:
                    generate_sample(
                        glide_model,
                        glide_options,
                        config,
                        train_idx,
                        global_step,
                    )
            
            # Checkpoint saving
            if train_idx % 5000 == 0 and train_idx > 0:
                save_checkpoint_with_manager(
                    checkpoint_manager,
                    glide_model,
                    optimizer,
                    config["epoch"],
                    train_idx,
                    global_step,
                    config["warmup_steps"],
                    config["warmup_type"],
                    config["base_lr"],
                )
        
        # Save final checkpoint
        print("Finished training, saving final checkpoint")
        save_checkpoint_with_manager(
            checkpoint_manager,
            glide_model,
            optimizer,
            config["epoch"],
            train_idx,
            global_step,
            config["warmup_steps"],
            config["warmup_type"],
            config["base_lr"],
        )
        
    except Exception as e:
        handle_training_error(
            e,
            checkpoint_manager,
            glide_model,
            optimizer,
            config["epoch"],
            train_idx,
            config["epoch_offset"],
            config["warmup_steps"],
            config["warmup_type"],
            config["base_lr"],
        )
        raise
    
    return train_idx + 1


def handle_training_error(
    error: Exception,
    checkpoint_manager: CheckpointManager,
    glide_model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    epoch: int,
    train_idx: int,
    epoch_offset: int,
    warmup_steps: int,
    warmup_type: str,
    base_lr: float,
) -> None:
    """Handle errors during training by optionally saving emergency checkpoint."""
    print(f"\n\n🚨 ERROR during training: {type(error).__name__}: {error}")
    
    # Ask user if they want to save checkpoint
    if prompt_with_timeout("Do you want to save an emergency checkpoint?", timeout=20, default=True):
        print("💾 Saving emergency checkpoint...")
        
        try:
            if train_idx >= 0:
                checkpoint_manager.save_checkpoint(
                    model=glide_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=train_idx,
                    global_step=epoch_offset + train_idx if train_idx >= 0 else epoch_offset,
                    warmup_steps=warmup_steps,
                    warmup_type=warmup_type,
                    base_lr=base_lr,
                    checkpoint_type="emergency",
                    additional_state={"error": str(error), "error_type": type(error).__name__},
                )
                print("✅ Emergency checkpoint saved successfully!")
            else:
                print("❌ Cannot save checkpoint - training loop never started")
        except Exception as save_error:
            print(f"❌ Failed to save emergency checkpoint: {save_error}")
    else:
        print("⏭️  Skipping checkpoint save as requested")


def print_metrics(
    log: Dict[str, float],
    step_metrics: Dict[str, float],
    global_step: int,
    accumulated_loss: th.Tensor,
    current_lr: float,
    warmup_steps: int,
    first_log: bool,
) -> None:
    """Print training metrics to console."""
    if first_log:
        print("\n=== Metrics Legend ===")
        print("Quartiles (q0-q3) represent loss at different denoising stages:")
        print("  q0: Early denoising (t=0-250) - removing large-scale noise")
        print("  q1: Mid-early (t=250-500) - refining basic structure")
        print("  q2: Mid-late (t=500-750) - adding details")
        print("  q3: Late denoising (t=750-1000) - final refinements")
        print("Lower values = better performance at that stage\n")
    
    # Create metrics display
    metrics_str = f"Step {global_step}: loss: {accumulated_loss.item():.4f}"
    metrics_str += f", lr: {current_lr:.2e}"
    
    # Add quartile losses
    q_losses = [f"q{i}: {step_metrics.get(f'loss_q{i}', 0.0):.4f}" for i in range(4)]
    metrics_str += f" | Quartiles: {' '.join(q_losses)}"
    
    print(metrics_str)
    
    if warmup_steps > 0 and global_step < warmup_steps:
        print(f"  Warmup progress: {global_step}/{warmup_steps} ({global_step/warmup_steps*100:.1f}%)")


def create_image_grid(images: List[PIL.Image.Image], grid_size: int) -> PIL.Image.Image:
    """Create a square grid from a list of images.
    
    Args:
        images: List of PIL images
        grid_size: Number of images per row/column (must be sqrt of len(images))
        
    Returns:
        PIL Image containing the grid
    """
    if len(images) != grid_size * grid_size:
        raise ValueError(f"Expected {grid_size * grid_size} images, got {len(images)}")
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create new image for grid
    grid_width = img_width * grid_size
    grid_height = img_height * grid_size
    grid_img = PIL.Image.new('RGB', (grid_width, grid_height))
    
    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        x = col * img_width
        y = row * img_height
        grid_img.paste(img, (x, y))
    
    return grid_img


def generate_eval_grid(
    glide_model: th.nn.Module,
    glide_options: dict,
    config: Dict[str, Any],
    prompts: List[str],
    train_idx: int,
    global_step: int,
) -> None:
    """Generate and save a grid of images from multiple evaluation prompts."""
    print(f"Generating evaluation grid with {len(prompts)} prompts at step {global_step}...")
    
    # Calculate grid size
    grid_size = int(len(prompts) ** 0.5)
    
    # Generate images for all prompts
    all_images = []
    for i, prompt in enumerate(prompts):
        print(f"  Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
        samples = glide_util.sample(
            glide_model=glide_model,
            glide_options=glide_options,
            side_x=config["side_x"],
            side_y=config["side_y"],
            prompt=prompt,
            batch_size=1,  # Generate one at a time for memory efficiency
            guidance_scale=config["sample_gs"],
            device=config["device"],
            prediction_respacing=str(config["test_steps"]),
            image_to_upsample=config["image_to_upsample"],
            sampler_name=config["sampler_name"],
        )
        # Convert to PIL and add to list
        img = train_util.pred_to_pil(samples)
        all_images.append(img)
    
    # Create grid
    grid_img = create_image_grid(all_images, grid_size)
    
    # Save grid
    grid_save_path = os.path.join(config["outputs_dir"], f"eval_grid_{train_idx}.png")
    grid_img.save(grid_save_path)
    print(f"Saved evaluation grid to {grid_save_path}")
    
    # Log to wandb
    if hasattr(config["wandb_run"], "__class__") and config["wandb_run"].__class__.__name__ == "MockWandbRun":
        # Skip wandb.Image for mocked runs
        pass
    else:
        # Log the grid
        config["wandb_run"].log({
            "eval_grid": wandb.Image(grid_save_path, caption=f"Evaluation grid at step {global_step}"),
        })
        
        # Also log individual images as a gallery with captions
        wandb_images = []
        for img, prompt in zip(all_images, prompts):
            wandb_images.append(wandb.Image(img, caption=prompt))
        
        config["wandb_run"].log({
            "eval_gallery": wandb_images
        })


def generate_sample(
    glide_model: th.nn.Module,
    glide_options: dict,
    config: Dict[str, Any],
    train_idx: int,
    global_step: int,
) -> None:
    """Generate and save sample images during training."""
    print(f"Generating sample at step {global_step}...")
    samples = glide_util.sample(
        glide_model=glide_model,
        glide_options=glide_options,
        side_x=config["side_x"],
        side_y=config["side_y"],
        prompt=config["prompt"],
        batch_size=config["sample_bs"],
        guidance_scale=config["sample_gs"],
        device=config["device"],
        prediction_respacing=str(config["test_steps"]),
        image_to_upsample=config["image_to_upsample"],
        sampler_name=config["sampler_name"],
    )
    sample_save_path = os.path.join(config["outputs_dir"], f"{train_idx}.png")
    train_util.pred_to_pil(samples).save(sample_save_path)
    
    # Log sample image to wandb (may be mocked for early_stop runs)
    if hasattr(config["wandb_run"], "__class__") and config["wandb_run"].__class__.__name__ == "MockWandbRun":
        # Skip wandb.Image for mocked runs
        pass
    else:
        config["wandb_run"].log({
            "samples": wandb.Image(sample_save_path, caption=config["prompt"]),
        })
    print(f"Saved sample {sample_save_path}")


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    sample_bs: int,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = "100",  # respacing for inference
    prompt: str = "",  # prompt for inference, not training
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    sample_interval: int = 1000,
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample="low_res_face.png",
    early_stop: int = 0,
    sampler_name: str = "plms",
    test_steps: int = 100,
    warmup_steps: int = 0,
    warmup_type: str = "linear",
    base_lr: float = 1e-5,
    epoch_offset: int = 0,
    batch_size: int = 1,
    checkpoint_manager: CheckpointManager = None,
    eval_prompts: list = None,
):
    """Run a single epoch of GLIDE fine-tuning with error handling and checkpointing."""
    # Select training step function
    train_step_fn = upsample_train_step if train_upsample else base_train_step
    
    # Prepare model
    glide_model.to(device)
    glide_model.train()
    
    # Create checkpoint manager if not provided
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(checkpoints_dir)
    
    # Pack configuration
    config = {
        "epoch": epoch,
        "epoch_offset": epoch_offset,
        "early_stop": early_stop,
        "base_lr": base_lr,
        "warmup_steps": warmup_steps,
        "warmup_type": warmup_type,
        "device": device,
        "batch_size": batch_size,
        "gradient_accumualation_steps": gradient_accumualation_steps,
        "log_frequency": log_frequency,
        "sample_interval": sample_interval,
        "sample_bs": sample_bs,
        "sample_gs": sample_gs,
        "test_steps": test_steps,
        "side_x": side_x,
        "side_y": side_y,
        "prompt": prompt,
        "outputs_dir": outputs_dir,
        "image_to_upsample": image_to_upsample,
        "sampler_name": sampler_name,
        "wandb_run": wandb_run,
        "log": {},
        "first_log": True,
        "eval_prompts": eval_prompts,
    }
    
    # Run training loop
    steps_taken = training_loop(
        dataloader,
        glide_model,
        glide_diffusion,
        glide_options,
        optimizer,
        checkpoint_manager,
        train_step_fn,
        config,
    )
    
    return steps_taken