import os
from typing import Any, Tuple

import torch as th
import wandb
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet

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
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape
                (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len)
                and reals is a tensor of shape (batch_size, 3, side_x, side_y)
                normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            A tuple of (loss, metrics_dict) where metrics_dict contains detailed metrics.
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
    
    # Compute per-sample MSE for quartile analysis
    per_sample_mse = th.nn.functional.mse_loss(
        epsilon, noise.to(device).detach(), reduction='none'
    ).mean(dim=[1, 2, 3])  # Average over C, H, W dimensions
    
    # Overall loss
    loss = per_sample_mse.mean()
    
    # Compute quartile losses based on timesteps
    num_timesteps = len(glide_diffusion.betas)
    quartile_size = num_timesteps // 4
    
    metrics = {"mse": loss.item()}
    
    # Calculate losses for each quartile
    for q in range(4):
        q_start = q * quartile_size
        q_end = (q + 1) * quartile_size if q < 3 else num_timesteps
        q_mask = (timesteps >= q_start) & (timesteps < q_end)
        
        if q_mask.any():
            q_loss = per_sample_mse[q_mask].mean().item()
            metrics[f"loss_q{q}"] = q_loss
            metrics[f"mse_q{q}"] = q_loss  # MSE and loss are the same in this case
        else:
            metrics[f"loss_q{q}"] = 0.0
            metrics[f"mse_q{q}"] = 0.0
    
    return loss, metrics


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
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y),
                  normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4),
                  normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            A tuple of (loss, metrics_dict) where metrics_dict contains detailed metrics.
    """
    tokens, masks, low_res_image, high_res_image = [x.to(device) for x in batch]
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device
    )
    noise = th.randn_like(
        high_res_image, device=device
    )  # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(
        high_res_image, timesteps, noise=noise
    ).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1)
    
    # Compute per-sample MSE for quartile analysis
    per_sample_mse = th.nn.functional.mse_loss(
        epsilon, noise.to(device).detach(), reduction='none'
    ).mean(dim=[1, 2, 3])  # Average over C, H, W dimensions
    
    # Overall loss
    loss = per_sample_mse.mean()
    
    # Compute quartile losses based on timesteps
    num_timesteps = len(glide_diffusion.betas)
    quartile_size = num_timesteps // 4
    
    metrics = {"mse": loss.item()}
    
    # Calculate losses for each quartile
    for q in range(4):
        q_start = q * quartile_size
        q_end = (q + 1) * quartile_size if q < 3 else num_timesteps
        q_mask = (timesteps >= q_start) & (timesteps < q_end)
        
        if q_mask.any():
            q_loss = per_sample_mse[q_mask].mean().item()
            metrics[f"loss_q{q}"] = q_loss
            metrics[f"mse_q{q}"] = q_loss  # MSE and loss are the same in this case
        else:
            metrics[f"loss_q{q}"] = 0.0
            metrics[f"mse_q{q}"] = 0.0
    
    return loss, metrics


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
):
    train_step: Any
    if train_upsample:
        train_step = upsample_train_step
    else:
        train_step = base_train_step

    glide_model.to(device)
    glide_model.train()
    log: dict[str, float] = {}
    first_log = True
    
    # Warmup scheduler helper
    def get_warmup_lr(step, base_lr, warmup_steps, warmup_type):
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
    
    for train_idx, batch in enumerate(dataloader):
        # Early stopping check
        if early_stop > 0 and train_idx >= early_stop:
            print(f"Early stopping at step {train_idx} (early_stop={early_stop})")
            break
        
        # Calculate global step for warmup
        global_step = epoch_offset + train_idx
        
        # Apply learning rate warmup
        current_lr = get_warmup_lr(global_step, base_lr, warmup_steps, warmup_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        accumulated_loss, step_metrics = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        accumulated_loss.backward()
        optimizer.step()
        glide_model.zero_grad()
        
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
        log["samples"] = samples_processed
        
        # Calculate parameter norm (cheap operation)
        param_norm = 0.0
        for p in glide_model.parameters():
            if p.requires_grad:
                param_norm += p.data.norm(2).item() ** 2
        param_norm = param_norm ** 0.5
        log["param_norm"] = param_norm
        
        # Log metrics to wandb every iteration
        wandb_run.log(log)
        
        # Console output at log_frequency intervals
        if train_idx > 0 and train_idx % log_frequency == 0:
            if first_log:
                print("\n=== Metrics Legend ===")
                print("Quartiles (q0-q3) represent loss at different denoising stages:")
                print("  q0: Early denoising (t=0-250) - removing large-scale noise")
                print("  q1: Mid-early (t=250-500) - refining basic structure")
                print("  q2: Mid-late (t=500-750) - adding details")
                print("  q3: Late denoising (t=750-1000) - final refinements")
                print("Lower values = better performance at that stage\n")
                first_log = False
            
            # Create metrics display
            metrics_str = f"Step {global_step}: loss: {accumulated_loss.item():.4f}"
            metrics_str += f", lr: {current_lr:.2e}"
            
            # Add quartile losses
            q_losses = [f"q{i}: {step_metrics.get(f'loss_q{i}', 0.0):.4f}" for i in range(4)]
            metrics_str += f" | Quartiles: {' '.join(q_losses)}"
            
            print(metrics_str)
            
            if warmup_steps > 0 and global_step < warmup_steps:
                print(f"  Warmup progress: {global_step}/{warmup_steps} ({global_step/warmup_steps*100:.1f}%)")
        
        # Sample generation at sample_interval intervals
        if train_idx > 0 and train_idx % sample_interval == 0:
            print(f"Generating sample at step {global_step}...")
            samples = glide_util.sample(
                glide_model=glide_model,
                glide_options=glide_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt,
                batch_size=sample_bs,
                guidance_scale=sample_gs,
                device=device,
                prediction_respacing=str(test_steps),
                image_to_upsample=image_to_upsample,
                sampler_name=sampler_name,
            )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            # Log sample image to wandb (may be mocked for early_stop runs)
            if hasattr(wandb_run, "__class__") and wandb_run.__class__.__name__ == "MockWandbRun":
                # Skip wandb.Image for mocked runs
                pass
            else:
                wandb_run.log({
                    "samples": wandb.Image(sample_save_path, caption=prompt),
                })
            print(f"Saved sample {sample_save_path}")
        
        # Checkpoint saving
        if train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            print(
                f"Saved checkpoint {train_idx} to "
                f"{checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
    print("Finished training, saving final checkpoint")
    train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
    
    # Return the number of steps taken in this epoch
    return train_idx + 1
