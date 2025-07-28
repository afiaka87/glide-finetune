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
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y),
                  normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4),
                  normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
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
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


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
):
    train_step: Any
    if train_upsample:
        train_step = upsample_train_step
    else:
        train_step = base_train_step

    glide_model.to(device)
    glide_model.train()
    log: dict[str, float] = {}
    
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
            
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        accumulated_loss.backward()
        optimizer.step()
        glide_model.zero_grad()
        log = {
            **log,
            "iter": train_idx,
            "loss": accumulated_loss.item() / gradient_accumualation_steps,
            "lr": current_lr,
        }
        # Sample from the model
        if train_idx > 0 and train_idx % log_frequency == 0:
            print(f"Step {global_step}: loss: {accumulated_loss.item():.4f}, lr: {current_lr:.2e}")
            if warmup_steps > 0 and global_step < warmup_steps:
                print(f"  Warmup progress: {global_step}/{warmup_steps} ({global_step/warmup_steps*100:.1f}%)")
            print(f"Sampling from model at iteration {train_idx}")
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
            # Handle wandb logging (may be mocked for early_stop runs)
            if hasattr(wandb_run, "__class__") and wandb_run.__class__.__name__ == "MockWandbRun":
                # Skip wandb.Image for mocked runs
                wandb_run.log({**log, "iter": train_idx})
            else:
                wandb_run.log(
                    {
                        **log,
                        "iter": train_idx,
                        "samples": wandb.Image(sample_save_path, caption=prompt),
                    }
                )
            print(f"Saved sample {sample_save_path}")
        if train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            print(
                f"Saved checkpoint {train_idx} to "
                f"{checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
        wandb_run.log(log)
    print("Finished training, saving final checkpoint")
    train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
