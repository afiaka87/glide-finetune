from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
import bitsandbytes as bnb
import argparse
import os
from typing import Tuple
import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm
from wandb import wandb
import fp16_util

import util
from loader import TextImageDataset

def train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
    respace: bool = True,
):
    with th.no_grad():
        tokens, masks, x_start = batch
        tokens, masks, x_start = tokens.to('cpu'), masks.to('cpu'), x_start.to('cpu')
        x_start = ( x_start.permute(0, 3, 1, 2).float() / 127.5 - 1).cpu()  # normalize to [-1, 1], shape: [batch_size, channels, side_x, side_y]
        timesteps = th.randint(1, len(glide_diffusion.betas) - 1, (x_start.shape[0],), device='cpu')  # random timesteps
        noise = th.randn_like(x_start, device='cpu')  # random noise
        if respace:
            timesteps = timesteps // ( glide_options["diffusion_steps"] // len(glide_diffusion.betas))
        
        x_t = glide_diffusion.q_sample(x_start, timesteps, noise=noise)  # sample from diffusion
        B, C = x_t.shape[:2]

    tokens = tokens.to(device)
    masks = masks.to(device)
    timesteps = timesteps.to(device)
    x_t = x_t.to(device)
    noise = noise.to(device)
    model_output = glide_model(x_t, timesteps, tokens=tokens, mask=masks,) 
    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
    epsilon, _ = th.split(model_output, C, dim=1)
    return util.mean_flat((noise.detach() - epsilon) ** 2)  # compute loss


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    batch_size: int,
    prompt: str = "",
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    wandb_run = None,
):
    os.makedirs(checkpoints_dir, exist_ok=True)
    trainer = fp16_util.MixedPrecisionTrainer(model=glide_model, use_fp16=glide_options["use_fp16"], fp16_scale_growth=0.001)
    trainer.zero_grad()
    for train_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        current_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            batch=batch,
            device=device,
        )
        current_loss = current_loss.mean()
        trainer.backward(current_loss)
        trainer.optimize(opt=optimizer)
        trainer.zero_grad()  # clear the gradients

        log = {} # 
        if train_idx % 10:
            tqdm.write(f"loss: {current_loss.item():.4f}")
            log = { **log, 'iter': train_idx, 'loss': current_loss.item() }

        # Sample from the model
        if train_idx % log_frequency == 0:
            with th.no_grad():
                print(f"Sampling from model at iteration {train_idx}")
                samples = util.sample(
                    model=glide_model, eval_diffusion=glide_diffusion,
                    options=glide_options, side_x=side_x,
                    side_y=side_y, prompt=prompt,
                    batch_size=batch_size, guidance_scale=1,
                    device=device,
                )
                sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
                util.pred_to_pil(samples).save(sample_save_path)
                log = { **log, 'iter': train_idx, 'samples': wandb.Image(sample_save_path, caption=prompt) }
                print(f"Saved sample {sample_save_path}")
        if train_idx % 1000 == 0 and train_idx > 0:
            save_model(glide_model, checkpoints_dir, train_idx)
        wandb_run.log(log)
    print(f"Finished training, saving final checkpoint")
    save_model(glide_model, checkpoints_dir, train_idx)

def save_model(glide_model: Text2ImUNet, checkpoints_dir: str, train_idx: int):
    th.save(glide_model.state_dict(), os.path.join(checkpoints_dir, f"glide-ft-{train_idx}.pt"))
    print(f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt")

def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    dropout=0.1,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    timestep_respacing="1000",
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,  # Tends to cause issues,not sure why as the paper states fp16 is stable.
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    project_name="glide_finetune",
    activation_checkpointing=True,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    test_prompt="a group of skiers are preparing to ski down a mountain.",
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Start wandb logging
    wandb_run = util.wandb_setup(
        batch_size=batch_size,
        side_x=side_x,
        side_y=side_y,
        learning_rate=learning_rate,
        use_fp16=use_fp16,
        device=device,
        data_dir=data_dir,
        base_dir=checkpoints_dir,
        project_name=project_name,
    )
    print("Wandb setup.")

    # Model setup
    glide_model, glide_diffusion, glide_options = util.load_base_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        dropout=dropout,
        timestep_respacing=timestep_respacing,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
    )
    glide_model.zero_grad()
    glide_model.train()
    glide_model.to(device)
    glide_model.requires_grad_(True)

    # Data setup
    dataset = TextImageDataset(
        folder=data_dir,
        side_x=side_x,
        side_y=side_y,
        resize_ratio=resize_ratio,
        uncond_p=uncond_p,
        shuffle=True,
        tokenizer=glide_model.tokenizer,
        text_ctx_len=glide_options["text_ctx"],
        use_captions=use_captions,
    )
    assert len(dataset) > 0, "Dataset is empty"
    print(f"Dataset contains {len(dataset)} images")

    # Data loader setup
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer setup
    optimizer = th.optim.SGD(
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
    )


    # Training setup
    outputs_dir = "./outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")
        run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=test_prompt,
            checkpoints_dir=checkpoints_dir,
            outputs_dir=outputs_dir,
            batch_size=batch_size,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--dropout", "-drop", type=float, default=0.1)
    parser.add_argument("--timestep_respacing", "-respace", type=str, default="1000")
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument("--resize_ratio", "-crop", type=float, default=0.8)
    parser.add_argument("--uncond_p", "-p", type=float, default=0.0)
    parser.add_argument("--resume_ckpt", "-resume", type=str, default="")
    parser.add_argument("--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/")
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument("--test_prompt", "-prompt", type=str, default="a group of skiers are preparing to ski down a mountain.")
    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    run_glide_finetune(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        timestep_respacing=args.timestep_respacing,
        dropout=args.dropout,
        side_x=args.side_x,
        side_y=args.side_y,
        resize_ratio=args.resize_ratio,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        use_fp16=args.use_fp16,
        device=device,
        log_frequency=args.log_frequency,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,
        use_captions=args.use_captions,
        num_epochs=args.epochs,
        test_prompt=args.test_prompt,
    )