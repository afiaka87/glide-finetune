"""
Shared image processing utilities for data loaders.

This module contains common image preprocessing functions used by both
the standard loader and WebDataset loader, including white padding removal
and random cropping utilities.
"""

import random
from typing import Optional, Tuple, Union

import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def trim_white_padding_tensor(
    img: torch.Tensor,
    white_thresh: Union[int, float] = 245,
    morph_kernel: Optional[int] = None,
) -> torch.Tensor:
    """
    Remove uniform white padding that was added to make a square canvas.

    Args:
        img: (C, H, W) torch.Tensor - Either uint8 in [0,255] or float in [0,1].
        white_thresh: threshold that defines "white". For uint8: 245, for float: 0.96
        morph_kernel: if set, performs a max-pool closing with this
                      square kernel (odd int) before bbox extraction.

    Returns:
        Cropped tensor (C, h', w') without the padded border.
        If no content is found, returns the original image.
    """
    assert img.ndim == 3, "expect (C,H,W)"
    c, h, w = img.shape

    # Adjust threshold based on tensor dtype
    if img.dtype == torch.uint8:
        # Use threshold as-is for uint8
        thresh_value = white_thresh
    else:
        # For float tensors, scale threshold to [0,1]
        thresh_value = white_thresh / 255.0 if white_thresh > 1 else white_thresh

    # Build boolean "content" mask - any channel below threshold
    content = (img < thresh_value).any(dim=0)  # (H,W) bool

    # Optional morphological closing in PyTorch
    if morph_kernel and morph_kernel > 1:
        pad = morph_kernel // 2
        # max_pool2d on the content mask = dilation
        content = (
            F.max_pool2d(
                content.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=morph_kernel,
                stride=1,
                padding=pad,
            )
            .squeeze()
            .bool()
        )

    # Find rows/cols that contain any non-white pixel
    rows = torch.where(content.any(dim=1))[0]
    cols = torch.where(content.any(dim=0))[0]
    
    if rows.numel() == 0 or cols.numel() == 0:
        # All-white edge case - return original
        return img

    # Get bounding box
    top, bottom = rows[0].item(), rows[-1].item() + 1
    left, right = cols[0].item(), cols[-1].item() + 1
    
    return img[:, int(top):int(bottom), int(left):int(right)]


def trim_white_padding_pil(
    pil_img: Image.Image, 
    white_thresh: int = 245,
    morph_kernel: Optional[int] = None,
) -> Image.Image:
    """
    Convenience wrapper: PIL.Image → tensor trim → PIL.Image.
    
    Args:
        pil_img: PIL Image to trim
        white_thresh: threshold for white detection (0-255)
        morph_kernel: optional morphological closing kernel size
        
    Returns:
        Trimmed PIL Image
    """
    # Convert to tensor, trim, convert back
    tensor = TF.pil_to_tensor(pil_img.convert("RGB"))  # (C,H,W) uint8
    trimmed = trim_white_padding_tensor(
        tensor, white_thresh=white_thresh, morph_kernel=morph_kernel
    )
    result: Image.Image = TF.to_pil_image(trimmed)
    return result


def random_center_crop(
    img: Image.Image,
    min_scale: float = 0.8,
    jitter_frac: float = 0.1,
    out_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Take a crop roughly around the centre, with mild random scale + offset.
    
    This provides additional augmentation after white padding removal,
    helping the model learn to handle various object positions.

    Args:
        img: PIL Image to crop
        min_scale: lower bound of the side-length as a fraction of original (0-1)
        jitter_frac: how far (fraction of leftover border) the crop can shift
        out_size: if given, resize the crop to this (w, h) with bicubic

    Returns:
        Cropped (and optionally resized) PIL Image
    """
    w, h = img.size
    assert 0.0 < min_scale <= 1.0, "min_scale must be in (0, 1]"

    # Choose random scale
    scale = random.uniform(min_scale, 1.0)
    crop_w = int(w * scale)
    crop_h = int(h * scale)

    # Center coordinates
    cx, cy = w // 2, h // 2
    left = cx - crop_w // 2
    top = cy - crop_h // 2

    # Add jitter within allowed range
    max_dx = int((w - crop_w) * jitter_frac)
    max_dy = int((h - crop_h) * jitter_frac)
    left += random.randint(-max_dx, max_dx)
    top += random.randint(-max_dy, max_dy)

    # Clamp to image bounds
    left = max(0, min(left, w - crop_w))
    top = max(0, min(top, h - crop_h))

    # Perform crop
    crop = img.crop((left, top, left + crop_w, top + crop_h))
    
    # Optionally resize to target size
    if out_size is not None:
        crop = crop.resize(out_size, resample=PIL.Image.BICUBIC)
        
    return crop


def preprocess_image_with_padding_removal(
    pil_img: Image.Image,
    target_size: Tuple[int, int],
    white_thresh: int = 245,
    use_random_crop: bool = True,
    min_crop_scale: float = 0.8,
    crop_jitter: float = 0.1,
) -> Image.Image:
    """
    Complete preprocessing pipeline: trim white padding, optional random crop, resize.
    
    This is the recommended preprocessing flow for images with white padding.
    
    Args:
        pil_img: Input PIL Image
        target_size: Final output size (width, height)
        white_thresh: Threshold for white detection (0-255)
        use_random_crop: Whether to apply random center cropping
        min_crop_scale: Minimum scale for random crop
        crop_jitter: Jitter fraction for random crop
        
    Returns:
        Preprocessed PIL Image at target_size
    """
    # Step 1: Remove white padding
    trimmed = trim_white_padding_pil(pil_img, white_thresh=white_thresh)
    
    # Step 2: Optional random center crop (only if image is larger than target)
    if (use_random_crop and 
        trimmed.width > target_size[0] and 
        trimmed.height > target_size[1]):
        trimmed = random_center_crop(
            trimmed, 
            min_scale=min_crop_scale, 
            jitter_frac=crop_jitter
        )
    
    # Step 3: Resize to final target size
    final = trimmed.resize(target_size, resample=PIL.Image.BICUBIC)
    
    return final