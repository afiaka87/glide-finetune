from glob import glob
from torchvision import transforms as T
from cgitb import enable
from re import A
import numpy as np
import bitsandbytes as bnb
import argparse
import os
from typing import Tuple
import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm, trange
from wandb import wandb
import util
from loader import TextImageDataset, create_webdataset


def train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    with th.no_grad():
        tokens, masks, x_start = [ x.to('cpu') for x in batch ]
        timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (x_start.shape[0],), device='cpu')
        noise = th.randn_like(x_start, device='cpu')
        x_t = glide_diffusion.q_sample(x_start, timesteps, noise=noise).to('cpu')
        _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
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
    sample_bs: int,  # batch size for inference, not training
    sample_gs: float = 4.0,  # guidance scale for inference, not training
    prompt: str = "",  # prompt for inference, not training
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    wandb_run=None,
    ga_steps = 1,
    epoch: int = 0,
):
    os.makedirs(checkpoints_dir, exist_ok=True)
    glide_model.to(device)
    glide_model.train()
    log = {}
    for train_idx, batch in tqdm(enumerate(dataloader)):
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            batch=batch,
            device=device,)
        accumulated_loss.backward()
        optimizer.step()
        glide_model.zero_grad()
        log = {**log, "iter": train_idx, "loss": accumulated_loss.item() / ga_steps}
        tqdm.write(f"loss: {accumulated_loss.item():.4f}")
        # Sample from the model
        if train_idx > 0 and train_idx % log_frequency == 0:
            tqdm.write(f"Sampling from model at iteration {train_idx}")
            samples = util.sample(
                glide_model=glide_model, glide_options=glide_options,
                side_x=side_x, side_y=side_y,
                prompt=prompt, batch_size=sample_bs,
                guidance_scale=sample_gs, device=device,
                prediction_respacing='27' # TODO use args
            )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            util.pred_to_pil(samples).save(sample_save_path)
            wandb_run.log({
                "iter": train_idx,
                "samples": wandb.Image(sample_save_path, caption=prompt),
            })
            tqdm.write(f"Saved sample {sample_save_path}")
        if train_idx % 5000 == 0 and train_idx > 0:
            save_model(glide_model, checkpoints_dir, train_idx, epoch)
            tqdm.write(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
        wandb_run.log(log)
    tqdm.write(f"Finished training, saving final checkpoint")
    save_model(glide_model, checkpoints_dir, train_idx, epoch)