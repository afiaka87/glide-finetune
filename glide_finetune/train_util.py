import os
from typing import Tuple

import numpy as np
from PIL import Image
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


def save_ema_model(
    ema_model, checkpoints_dir: str, train_idx: int, epoch: int, ema_rate: float
):
    """Save EMA model checkpoint matching OpenAI's naming convention."""
    ema_path = os.path.join(
        checkpoints_dir, f"ema_{ema_rate}_{epoch}x{train_idx:06d}.pt"
    )
    th.save(
        ema_model.state_dict(),
        ema_path,
    )
    tqdm.write(f"Saved EMA checkpoint to {ema_path}")


def pred_to_pil(pred: th.Tensor) -> Image.Image:
    """Convert model prediction tensor to PIL Image.

    Args:
        pred: Tensor of shape [batch_size, channels, height, width] or [channels, height, width]
              Values expected to be in range [-1, 1]

    Returns:
        PIL Image (for single image)
    """
    # Handle both [C, H, W] and [B, C, H, W] tensors
    if pred.ndim == 3:
        # Add batch dimension if missing
        pred = pred.unsqueeze(0)

    # Only process first image if batch size > 1
    if pred.shape[0] > 1:
        print(
            f"Warning: pred_to_pil received batch of {pred.shape[0]} images, only converting first one"
        )
        pred = pred[:1]

    # Scale from [-1, 1] to [0, 255]
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()

    # Convert from [1, C, H, W] to [H, W, C]
    image_array = scaled.squeeze(0).permute(1, 2, 0).numpy()

    return Image.fromarray(image_array)


def find_optimal_grid_dims(n: int) -> Tuple[int, int]:
    """
    Find optimal grid dimensions that minimize empty space.

    Args:
        n: Number of images

    Returns:
        Tuple of (columns, rows) for the optimal grid layout
    """
    if n <= 0:
        return 1, 1

    # Special cases for small numbers
    if n == 1:
        return 1, 1
    elif n == 2:
        return 2, 1
    elif n == 3:
        return 3, 1
    elif n == 4:
        return 2, 2

    # Find factors that minimize empty space
    best_cols, best_rows = n, 1
    min_empty = float("inf")
    min_ratio_diff = float("inf")  # Prefer grids closer to square aspect ratio

    for rows in range(1, int(np.sqrt(n)) + 2):
        cols = (n + rows - 1) // rows  # Ceiling division
        empty = (rows * cols) - n

        # Calculate aspect ratio difference from square
        ratio_diff = abs(cols - rows)

        # Prefer less empty space, and among equal empty space, prefer squarer grids
        if empty < min_empty or (empty == min_empty and ratio_diff < min_ratio_diff):
            min_empty = empty
            min_ratio_diff = ratio_diff
            best_cols, best_rows = cols, rows

    return best_cols, best_rows


def next_power_of_2(n: int) -> int:
    """Return the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    # If n is already a power of 2, return it
    if n & (n - 1) == 0:
        return n
    # Find the next power of 2
    power = 1
    while power < n:
        power *= 2
    return power


def make_grid(
    images: list,
    grid_size: int | None = None,
    mode: str = "auto",
    pad_to_power_of_2: bool = False,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Create a grid of images from a list of PIL images.

    Args:
        images: List of PIL Image objects
        grid_size: Number of columns for the grid (only used in 'fixed' mode)
        mode: Grid layout mode - 'auto' (optimal), 'square' (force square),
              'wide' (prefer horizontal), 'tall' (prefer vertical), 'fixed' (use grid_size)
        pad_to_power_of_2: If True, pad the final image to power-of-2 dimensions
        background_color: RGB color tuple for background/padding

    Returns:
        A single PIL Image containing the grid
    """
    n = len(images)

    if n == 0:
        return Image.new("RGB", (64, 64), background_color)

    # Get dimensions from first image
    img_width, img_height = images[0].size

    # Determine grid dimensions based on mode
    if mode == "auto":
        cols, rows = find_optimal_grid_dims(n)
    elif mode == "square":
        size = int(np.ceil(np.sqrt(n)))
        cols, rows = size, size
    elif mode == "wide":
        # Prefer more columns than rows
        rows = max(1, int(np.sqrt(n / 2)))
        cols = (n + rows - 1) // rows
    elif mode == "tall":
        # Prefer more rows than columns
        cols = max(1, int(np.sqrt(n / 2)))
        rows = (n + cols - 1) // cols
    elif mode == "fixed" and grid_size is not None:
        cols = grid_size
        rows = (n + cols - 1) // cols
    else:
        # Fallback to auto mode
        cols, rows = find_optimal_grid_dims(n)

    # Calculate grid dimensions
    grid_width = cols * img_width
    grid_height = rows * img_height

    # Optionally pad to power of 2
    if pad_to_power_of_2:
        target_width = next_power_of_2(grid_width)
        target_height = next_power_of_2(grid_height)
    else:
        target_width = grid_width
        target_height = grid_height

    # Create new image for grid with background color
    grid_img = Image.new("RGB", (target_width, target_height), background_color)

    # Calculate offset for centering if we padded to power of 2
    x_offset = (target_width - grid_width) // 2
    y_offset = (target_height - grid_height) // 2

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Skip if we've placed all images (shouldn't happen with proper grid calculation)
        if row >= rows:
            break

        x = x_offset + col * img_width
        y = y_offset + row * img_height
        grid_img.paste(img, (x, y))

    return grid_img


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    # Copy the array to make it writable and avoid warnings
    arr = np.array(pil_image, dtype=np.float32)
    return th.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


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
