import os

import numpy as np
import PIL.Image
import torch as th
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def save_model(glide_model: th.nn.Module, checkpoints_dir: str, train_idx: int, epoch: int):
    th.save(
        glide_model.state_dict(),
        os.path.join(checkpoints_dir, f"glide-ft-{epoch}x{train_idx}.pt"),
    )
    tqdm.write(f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{epoch}x{train_idx}.pt")


def pred_to_pil(pred: th.Tensor) -> PIL.Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return PIL.Image.fromarray(reshaped.numpy())


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image).copy()).float().permute(2, 0, 1) / 127.5 - 1.0


def resize_for_upsample(
    original, low_res_x, low_res_y, upscale_factor: int = 4
) -> tuple[th.Tensor, th.Tensor]:
    """
    Resize/Crop an image to the size of the low resolution image. This is useful for upsampling.

    Args:
        original: A PIL.Image object to be cropped.
        low_res_x: The width of the low resolution image.
        low_res_y: The height of the low resolution image.
        upscale_factor: The factor by which to upsample the image.

    Returns:
        The downsampled image and the corresponding upscaled version cropped according to upscale_factor.
    """
    high_res_x, high_res_y = low_res_x * upscale_factor, low_res_y * upscale_factor
    high_res_image = original.resize((high_res_x, high_res_y), PIL.Image.LANCZOS)
    high_res_tensor = pil_image_to_norm_tensor(pil_image=high_res_image)
    low_res_image = high_res_image.resize((low_res_x, low_res_y), resample=PIL.Image.BICUBIC)
    low_res_tensor = pil_image_to_norm_tensor(pil_image=low_res_image)
    return low_res_tensor, high_res_tensor


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def create_warmup_scheduler(
    optimizer: th.optim.Optimizer,
    warmup_steps: int,
    warmup_start_lr: float = 7e-7,
    target_lr: float = 1e-5,
) -> LambdaLR | None:
    """Create learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup
        target_lr: Target learning rate after warmup

    Returns:
        LambdaLR scheduler or None if no warmup
    """
    if warmup_steps == 0:
        return None

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            progress = float(current_step) / float(max(1, warmup_steps))
            actual_lr = warmup_start_lr + progress * (target_lr - warmup_start_lr)
            return actual_lr / target_lr
        # After warmup, use full learning rate
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def wandb_setup(
    batch_size: int,
    side_x: int,
    side_y: int,
    learning_rate: float,
    use_fp16: bool,
    device: str,
    data_dir: str,
    base_dir: str,
    project_name: str = "glide-text2im-finetune",
    seed: int | None = None,
):
    return wandb.init(
        project=project_name,
        config={
            "batch_size": batch_size,
            "side_x": side_x,
            "side_y": side_y,
            "learning_rate": learning_rate,
            "use_fp16": use_fp16,
            "device": device,
            "data_dir": data_dir,
            "base_dir": base_dir,
            "seed": seed,
        },
    )
