import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import PIL.Image
import torch as th
from tqdm import tqdm

import wandb


def save_model(
    glide_model: th.nn.Module, checkpoints_dir: str, train_idx: int, epoch: int
):
    th.save(
        glide_model.state_dict(),
        os.path.join(checkpoints_dir, f"glide-ft-{epoch}x{train_idx}.pt"),
    )
    tqdm.write(
        f"Saved checkpoint {train_idx} to "
        f"{checkpoints_dir}/glide-ft-{epoch}x{train_idx}.pt"
    )


def pred_to_pil(pred: th.Tensor) -> PIL.Image.Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return PIL.Image.fromarray(reshaped.numpy())


def save_image_compressed(
    image: PIL.Image.Image,
    filepath: Union[str, Path],
    quality: int = 95,
    optimize: bool = True,
) -> str:
    """
    Save a PIL image as a compressed JPEG with high quality.

    Args:
        image: PIL Image to save
        filepath: Path to save the image (extension will be replaced with .jpg)
        quality: JPEG quality (1-100, default 95 for high quality)
        optimize: Whether to optimize the JPEG encoding (default True)

    Returns:
        The actual filepath where the image was saved (with .jpg extension)
    """
    # Convert Path to string and replace extension with .jpg
    filepath = str(filepath)
    if filepath.endswith((".png", ".PNG")):
        filepath = filepath[:-4] + ".jpg"
    elif not filepath.endswith((".jpg", ".jpeg", ".JPG", ".JPEG")):
        filepath = filepath + ".jpg"

    # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
    if image.mode in ("RGBA", "LA", "P"):
        # Create a white background
        background = PIL.Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(
            image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None
        )
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Save with high quality JPEG compression
    image.save(filepath, "JPEG", quality=quality, optimize=optimize)
    return filepath


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape
    [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def resize_for_upsample(
    original, low_res_x, low_res_y, upscale_factor: int = 4
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Resize/Crop an image to the size of the low resolution image. This is useful for
    upsampling.

    Args:
        original: A PIL.Image object to be cropped.
        low_res_x: The width of the low resolution image.
        low_res_y: The height of the low resolution image.
        upscale_factor: The factor by which to upsample the image.

    Returns:
        The downsampled image and the corresponding upscaled version cropped according
        to upscale_factor.
    """
    high_res_x, high_res_y = low_res_x * upscale_factor, low_res_y * upscale_factor
    high_res_image = original.resize((high_res_x, high_res_y), PIL.Image.LANCZOS)
    high_res_tensor = pil_image_to_norm_tensor(pil_image=high_res_image)
    low_res_image = high_res_image.resize(
        (low_res_x, low_res_y), resample=PIL.Image.BICUBIC
    )
    low_res_tensor = pil_image_to_norm_tensor(pil_image=low_res_image)
    return low_res_tensor, high_res_tensor


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


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
        },
    )
