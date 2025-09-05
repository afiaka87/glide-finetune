import os
from typing import Tuple

import numpy as np
from PIL import Image
import PIL
import torch as th
import wandb
from tqdm import tqdm


def save_model(
    glide_model: th.nn.Module, checkpoints_dir: str, train_idx: int, epoch: int
):
    th.save(
        glide_model.state_dict(),
        os.path.join(checkpoints_dir, f"glide-ft-{epoch}x{train_idx}.pt"),
    )
    tqdm.write(
        f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{epoch}x{train_idx}.pt"
    )


def pred_to_pil(pred: th.Tensor) -> Image.Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())


def make_grid(images: list, grid_size: int = None) -> Image.Image:
    """
    Create a grid of images from a list of PIL images.
    
    Args:
        images: List of PIL Image objects
        grid_size: Number of images per row/column (assumes square grid). 
                   If None, automatically calculates based on number of images.
    
    Returns:
        A single PIL Image containing the grid
    """
    n = len(images)
    
    if grid_size is None:
        # Calculate grid size - prefer square grids
        grid_size = int(np.ceil(np.sqrt(n)))
    
    # Get dimensions from first image
    if n == 0:
        return Image.new('RGB', (64, 64))
    
    img_width, img_height = images[0].size
    
    # Create grid dimensions
    grid_width = grid_size * img_width
    grid_height = grid_size * img_height
    
    # Create new image for grid
    grid_img = Image.new('RGB', (grid_width, grid_height))
    
    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        x = col * img_width
        y = row * img_height
        grid_img.paste(img, (x, y))
    
    return grid_img


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    # Copy the array to make it writable and avoid warnings
    arr = np.array(pil_image, dtype=np.float32)
    return th.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


def resize_for_upsample(
    original, low_res_x, low_res_y, upscale_factor: int = 4
) -> Tuple[th.Tensor, th.Tensor]:
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
