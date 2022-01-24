from torch.cuda.amp import autocast
import argparse
import os
from typing import Tuple

import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm

import util
from loader import TextImageDataset


def train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    batch = [x.detach().clone().to(device) for x in batch]
    tokens, masks, x_imgs = batch
    x_imgs = x_imgs.permute(0, 3, 1, 2).float() / 127.5 - 1 # normalize to [-1, 1], shape: [batch_size, channels, side_x, side_y]
    timesteps = th.randint(0, len(glide_diffusion.betas)-1, (x_imgs.shape[0],), device=device) # random timesteps
    noise = th.randn_like(x_imgs, device=device) # random noise
    x_t = glide_diffusion.q_sample(x_imgs, timesteps, noise=noise) # sample from diffusion
    model_output = glide_model(x_t, timesteps, tokens=tokens, mask=masks) # forward pass
    epsilon = model_output[..., :3, :, :] # extract epsilon
    # delta = model_output[..., 3:, :, :] # extract delta
    # util.pred_to_pil(epsilon).save("current_epsilon.png") # save epsilon
    # util.pred_to_pil(x_t).save("current_x_t.png") # save x_t
    # util.pred_to_pil(delta).save("current_delta.png") # save delta
    return th.nn.functional.mse_loss(epsilon, noise.detach()) # compute loss


def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    grad_acc=1,
    learning_rate=1e-5,
    dropout=0.1,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,  # Tends to cause issues,not sure why as the paper states fp16 is stable.
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    weight_decay=0.0,
    project_name="glide_finetune",
    activation_checkpointing=False,
):
    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Start wandb logging
    wandb_run = util.wandb_setup(
        batch_size=batch_size,
        grad_acc=grad_acc,
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
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
    )
    glide_model.train()
    glide_model.to(device)

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
        force_reload=False,
    ) 
    assert len(dataset) > 0, "Dataset is empty"
    print(f"Dataset contains {len(dataset)} images")

    # Data loader setup
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Optimizer setup
    optimizer = th.optim.SGD(  # use bitsandbytes adam, supports 8-bit
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
        momentum=0.9,
    )

    # Training setup
    current_loss = 0  # hold gradients until all are accumulated
    accum_losses = []  # for determining grad_acc division

    # Training loop
    for train_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        current_loss += loss.item()  # sum up loss over grad_acc batches
        loss.backward()  # backpropagate the loss
        if train_idx % grad_acc == grad_acc - 1:
            optimizer.step()  # update the model parameters
            current_loss /= grad_acc  # finally average the loss over the grad_acc
            accum_losses.append(current_loss)
            wandb_run.log({"loss": current_loss})
            tqdm.write(f"Loss: {current_loss:.12f}")
            current_loss = 0
        if train_idx % 5000 == 0 and train_idx > 0:
            th.save(
                glide_model.state_dict(),
                os.path.join(checkpoints_dir, f"glide-ft-{train_idx}.pt"),
            )
            print(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--timestep_respacing", type=str, default="1000")
    parser.add_argument("--side_x", type=int, default=64)
    parser.add_argument("--side_y", type=int, default=64)
    parser.add_argument("--resize_ratio", type=float, default=0.8)
    parser.add_argument("--uncond_p", type=float, default=0.0)
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--checkpoints_dir", type=str, default="./glide_checkpoints/")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--freeze_transformer", action="store_true")
    parser.add_argument("--freeze_diffusion", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--project_name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")
    print(f"Resuming training checkpoint from {args.resume_ckpt} on device {device} with {args.grad_acc} gradients accumulated per step.")
    run_glide_finetune(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        side_x=args.side_x,
        side_y=args.side_y,
        resize_ratio=args.resize_ratio,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        use_fp16=args.use_fp16,
        device=device,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        weight_decay=args.weight_decay,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,
    )
