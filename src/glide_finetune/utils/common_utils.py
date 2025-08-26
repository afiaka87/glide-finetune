"""Common utility functions to reduce code duplication.

Consolidated utilities for image processing, tensor operations, and common transformations.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.common_utils")


# Image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_image_to_tensor(
    image: Image.Image | np.ndarray,
    target_range: str = "tanh",
) -> torch.Tensor:
    """Convert PIL image or numpy array to normalized tensor.
    
    Args:
        image: Input image (PIL or numpy)
        target_range: Target normalization range
            - "tanh": [-1, 1] range (for diffusion models)
            - "sigmoid": [0, 1] range
            - "imagenet": ImageNet normalization
            - "clip": CLIP model normalization
            
    Returns:
        Normalized tensor [C, H, W] or [B, C, H, W] if batched
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        image = np.asarray(image).copy()

    # Ensure float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Handle different input ranges
    if image.max() > 1.0:
        image = image / 255.0  # Convert from [0, 255] to [0, 1]

    # Convert to tensor
    if len(image.shape) == 2:  # Grayscale
        tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dim
    elif len(image.shape) == 3:  # RGB/RGBA
        tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
    elif len(image.shape) == 4:  # Batched
        tensor = torch.from_numpy(image).permute(0, 3, 1, 2)  # BHWC -> BCHW
    else:
        msg = f"Unsupported image shape: {image.shape}"
        raise ValueError(msg)

    # Apply target normalization
    if target_range == "tanh":
        # [-1, 1] range for diffusion models
        tensor = tensor * 2.0 - 1.0
    elif target_range == "sigmoid":
        # [0, 1] range (already in this range)
        pass
    elif target_range == "imagenet":
        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
        tensor = (tensor - mean) / std
    elif target_range == "clip":
        # CLIP model normalization
        mean = torch.tensor(CLIP_MEAN).view(-1, 1, 1)
        std = torch.tensor(CLIP_STD).view(-1, 1, 1)
        tensor = (tensor - mean) / std
    else:
        msg = f"Unknown target_range: {target_range}"
        raise ValueError(msg)

    return tensor


def denormalize_tensor_to_image(
    tensor: torch.Tensor,
    source_range: str = "tanh",
    return_pil: bool = True,
) -> Image.Image | np.ndarray:
    """Convert normalized tensor back to image.
    
    Args:
        tensor: Input tensor [C, H, W] or [B, C, H, W]
        source_range: Source normalization range
            - "tanh": [-1, 1] range (for diffusion models)
            - "sigmoid": [0, 1] range
            - "imagenet": ImageNet normalization
            - "clip": CLIP model normalization
        return_pil: Return PIL Image if True, numpy array if False
        
    Returns:
        Denormalized image (PIL or numpy)
    """
    # Remove batch dimension if present and single
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor[0]

    # Clone to avoid modifying original
    tensor = tensor.clone().cpu()

    # Apply inverse normalization
    if source_range == "tanh":
        # From [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
    elif source_range == "sigmoid":
        # Already in [0, 1]
        pass
    elif source_range == "imagenet":
        # Inverse ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
        tensor = tensor * std + mean
    elif source_range == "clip":
        # Inverse CLIP normalization
        mean = torch.tensor(CLIP_MEAN).view(-1, 1, 1)
        std = torch.tensor(CLIP_STD).view(-1, 1, 1)
        tensor = tensor * std + mean
    else:
        msg = f"Unknown source_range: {source_range}"
        raise ValueError(msg)

    # Clamp to valid range
    tensor = tensor.clamp(0, 1)

    # Convert to numpy
    if tensor.dim() == 3:
        # CHW -> HWC
        array = tensor.permute(1, 2, 0).numpy()
    elif tensor.dim() == 4:
        # BCHW -> BHWC
        array = tensor.permute(0, 2, 3, 1).numpy()
    else:
        msg = f"Unsupported tensor shape: {tensor.shape}"
        raise ValueError(msg)

    # Convert to uint8
    array = (array * 255).astype(np.uint8)

    if return_pil and array.ndim == 3:
        return Image.fromarray(array)
    return array


def pil_to_tensor(
    pil_image: Image.Image,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert PIL image to tensor with optional normalization.
    
    Simplified version for backward compatibility.
    
    Args:
        pil_image: PIL Image
        normalize: If True, normalize to [-1, 1]
        
    Returns:
        Tensor [C, H, W]
    """
    return normalize_image_to_tensor(
        pil_image,
        target_range="tanh" if normalize else "sigmoid",
    )


def tensor_to_pil(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> Image.Image:
    """Convert tensor to PIL image with optional denormalization.
    
    Simplified version for backward compatibility.
    
    Args:
        tensor: Input tensor
        denormalize: If True, assume input is in [-1, 1]
        
    Returns:
        PIL Image
    """
    return denormalize_tensor_to_image(
        tensor,
        source_range="tanh" if denormalize else "sigmoid",
        return_pil=True,
    )


def resize_image_aspect_ratio(
    image: Image.Image,
    target_size: int,
    method: str = "lanczos",
) -> Image.Image:
    """Resize image maintaining aspect ratio.
    
    Args:
        image: Input PIL image
        target_size: Target size for shortest edge
        method: Resize method ("lanczos", "bilinear", "nearest")
        
    Returns:
        Resized PIL image
    """
    width, height = image.size

    if width < height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)

    method_map = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }

    resample = method_map.get(method, Image.LANCZOS)
    return image.resize((new_width, new_height), resample)


def center_crop_image(
    image: Image.Image,
    crop_size: int | tuple[int, int],
) -> Image.Image:
    """Center crop image to specified size.
    
    Args:
        image: Input PIL image
        crop_size: Target crop size (int or (width, height))
        
    Returns:
        Cropped PIL image
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    width, height = image.size
    crop_width, crop_height = crop_size

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    return image.crop((left, top, right, bottom))


def smart_resize(
    image: Image.Image,
    target_size: int | tuple[int, int],
    method: str = "center_crop",
) -> Image.Image:
    """Smart resize with various strategies.
    
    Args:
        image: Input PIL image
        target_size: Target size
        method: Resize method
            - "center_crop": Resize then center crop
            - "pad": Resize and pad to maintain aspect ratio
            - "stretch": Stretch to exact size (may distort)
            
    Returns:
        Resized PIL image
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    if method == "center_crop":
        # Resize to cover target area, then crop
        width, height = image.size
        target_width, target_height = target_size

        scale = max(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize((new_width, new_height), Image.LANCZOS)
        return center_crop_image(image, target_size)

    if method == "pad":
        # Resize to fit within target area, then pad
        image.thumbnail(target_size, Image.LANCZOS)

        # Create new image with padding
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image

    if method == "stretch":
        # Direct resize (may distort)
        return image.resize(target_size, Image.LANCZOS)

    msg = f"Unknown resize method: {method}"
    raise ValueError(msg)


def batch_tensor_to_images(
    tensor: torch.Tensor,
    source_range: str = "tanh",
    return_pil: bool = True,
) -> list[Image.Image | np.ndarray]:
    """Convert batch of tensors to list of images.
    
    Args:
        tensor: Batch tensor [B, C, H, W]
        source_range: Source normalization range
        return_pil: Return PIL Images if True
        
    Returns:
        List of images
    """
    if tensor.dim() != 4:
        msg = f"Expected 4D tensor, got {tensor.dim()}D"
        raise ValueError(msg)

    images = []
    for i in range(tensor.size(0)):
        img = denormalize_tensor_to_image(
            tensor[i],
            source_range=source_range,
            return_pil=return_pil,
        )
        images.append(img)

    return images


def images_to_batch_tensor(
    images: list[Image.Image | np.ndarray],
    target_range: str = "tanh",
) -> torch.Tensor:
    """Convert list of images to batch tensor.
    
    Args:
        images: List of PIL images or numpy arrays
        target_range: Target normalization range
        
    Returns:
        Batch tensor [B, C, H, W]
    """
    tensors = []
    for img in images:
        tensor = normalize_image_to_tensor(img, target_range=target_range)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dim
        tensors.append(tensor)

    return torch.cat(tensors, dim=0)


def get_image_grid(
    images: list[Image.Image],
    rows: int | None = None,
    cols: int | None = None,
    padding: int = 2,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Create a grid of images.
    
    Args:
        images: List of PIL images
        rows: Number of rows (auto-calculated if None)
        cols: Number of columns (auto-calculated if None)
        padding: Padding between images
        background: Background color
        
    Returns:
        Grid image
    """
    n_images = len(images)

    if rows is None and cols is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))

    # Get max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create grid image
    grid_width = cols * max_width + (cols - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), background)

    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * (max_width + padding)
        y = row * (max_height + padding)

        # Center image in cell
        x += (max_width - img.width) // 2
        y += (max_height - img.height) // 2

        grid.paste(img, (x, y))

    return grid


# Tensor utilities

def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Safe division avoiding divide by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small epsilon for numerical stability
        
    Returns:
        Result of division
    """
    return numerator / (denominator + eps)


def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute cosine similarity between tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        dim: Dimension to compute similarity along
        eps: Epsilon for numerical stability
        
    Returns:
        Cosine similarity
    """
    x_norm = x / (x.norm(dim=dim, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=dim, keepdim=True) + eps)
    return (x_norm * y_norm).sum(dim=dim)


def lerp(
    start: torch.Tensor,
    end: torch.Tensor,
    weight: float | torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation between tensors.
    
    Args:
        start: Starting tensor
        end: Ending tensor
        weight: Interpolation weight (0 = start, 1 = end)
        
    Returns:
        Interpolated tensor
    """
    return start + weight * (end - start)


def slerp(
    start: torch.Tensor,
    end: torch.Tensor,
    weight: float | torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical linear interpolation between tensors.
    
    Args:
        start: Starting tensor
        end: Ending tensor
        weight: Interpolation weight (0 = start, 1 = end)
        eps: Epsilon for numerical stability
        
    Returns:
        Interpolated tensor
    """
    # Normalize inputs
    start_norm = start / (start.norm(dim=-1, keepdim=True) + eps)
    end_norm = end / (end.norm(dim=-1, keepdim=True) + eps)

    # Compute angle between vectors
    dot = (start_norm * end_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)

    # Use linear interpolation for small angles
    use_lerp = theta < 0.01

    # Spherical interpolation
    sin_theta = torch.sin(theta)
    a = torch.sin((1 - weight) * theta) / (sin_theta + eps)
    b = torch.sin(weight * theta) / (sin_theta + eps)

    result = a * start + b * end

    # Use lerp for small angles
    if use_lerp.any():
        lerp_result = lerp(start, end, weight)
        result = torch.where(use_lerp, lerp_result, result)

    return result
