import os
import time
from pathlib import Path
from random import choice, randint, random
from typing import Union

import PIL
import torch
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor


def trim_white_padding_tensor(
    img: torch.Tensor,
    white_thresh: Union[int, float] = 0.95,  # 0-1 for float32 / 0-255 for uint8
    morph_kernel: int | None = None,
) -> torch.Tensor:
    """
    Remove uniform white padding that was added to make a square canvas.

    Args
    ----
    img : (C, H, W) torch.Tensor
          Either uint8 in [0,255] or float in [0,1].
    white_thresh : threshold that defines "white".
    morph_kernel : if set, performs a max-pool closing with this
                   square kernel (odd int) before bbox extraction.

    Returns
    -------
    Cropped tensor (C, h', w') without the padded border.
    """
    assert img.ndim == 3, "expect (C,H,W)"
    c, h, w = img.shape

    # --- 1. build boolean “content” mask ------------------------------
    if img.dtype == torch.uint8:
        content = (img < white_thresh).any(dim=0)  # (H,W) bool
    else:  # float
        content = (img < white_thresh).any(dim=0)

    # --- 2. optional morphological closing in PyTorch -----------------
    if morph_kernel and morph_kernel > 1:
        pad = morph_kernel // 2
        # max_pool2d on the NOT-content mask = dilation of content
        content = (
            torch.nn.functional.max_pool2d(  # ↓ F.max_pool2d docs
                content.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=morph_kernel,
                stride=1,
                padding=pad,
            )
            .squeeze()
            .bool()
        )

    # --- 3. find rows / cols that contain any non-white pixel ----------
    rows = torch.where(content.any(dim=1))[0]  # torch.any docs
    cols = torch.where(content.any(dim=0))[0]  # torch.any docs
    if rows.numel() == 0 or cols.numel() == 0:  # all-white edge-case
        return img

    top, bottom = rows[0].item(), rows[-1].item() + 1  # +1 for slicing
    left, right = cols[0].item(), cols[-1].item() + 1
    return img[:, int(top):int(bottom), int(left):int(right)]


def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(
        shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0)
    )
    return image_transform(image)


def get_image_files_dict(base_path):
    image_files = [
        *base_path.glob("**/*.png"),
        *base_path.glob("**/*.jpg"),
        *base_path.glob("**/*.jpeg"),
        *base_path.glob("**/*.bmp"),
    ]
    return {image_file.stem: image_file for image_file in image_files}


def get_text_files_dict(base_path):
    text_files = [*base_path.glob("**/*.txt")]
    return {text_file.stem: text_file for text_file in text_files}


def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)


class TextImageDataset(Dataset):
    def __init__(
        self,
        folder="",
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        shuffle=False,
        tokenizer=None,
        text_ctx_len=128,
        uncond_p=0.0,
        use_captions=False,
        enable_glide_upsample=False,
        upscale_factor=4,
    ):
        super().__init__()
        folder = Path(folder)

        self.image_files = get_image_files_dict(folder)
        if use_captions:
            self.text_files = get_text_files_dict(folder)
            self.keys = get_shared_stems(self.image_files, self.text_files)
            print(f"Found {len(self.keys)} images.")
            print(f"Using {len(self.text_files)} text files.")
        else:
            self.text_files = None
            self.keys = list(self.image_files.keys())
            print(f"Found {len(self.keys)} images.")
            print("NOT using text files. Restart with --use_captions to enable...")
            time.sleep(3)

        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len

        self.shuffle = shuffle
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y
        self.resolution = side_x  # Assuming square images
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.enable_upsample = enable_glide_upsample
        self.upscale_factor = upscale_factor
        
        # Create transform for final resizing
        self.transform = T.Resize((self.resolution, self.resolution), interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_caption(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        
        # Check if file exists and is readable
        if not os.path.exists(text_file):
            print(f"Caption file not found: {text_file} (index {ind}, key {key})")
            print("  -> Using unconditional tokens")
            return get_uncond_tokens_mask(self.tokenizer)
            
        # Check file size
        file_size = os.path.getsize(text_file)
        if file_size == 0:
            print(f"Empty caption file: {text_file} (index {ind}, key {key})")
            print("  -> Using unconditional tokens")
            return get_uncond_tokens_mask(self.tokenizer)
            
        try:
            descriptions = open(text_file, "r").readlines()
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            
            if not descriptions:
                print(f"No valid captions in file: {text_file} (index {ind}, key {key})")
                print("  -> Using unconditional tokens")
                return get_uncond_tokens_mask(self.tokenizer)
                
            description = choice(descriptions).strip()
            return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
        except Exception as e:
            print(f"Error reading caption file: {text_file} (index {ind}, key {key})")
            print(f"  Error: {type(e).__name__}: {e}")
            print("  -> Using unconditional tokens")
            return get_uncond_tokens_mask(self.tokenizer)

    def __getitem__(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]
        if self.text_files is None or random() < self.uncond_p:
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            tokens, mask = self.get_caption(ind)

        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError):
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        if self.enable_upsample:  # base image from cropped high-res image
            upsample_pil_image = random_resized_crop(
                original_pil_image,
                (self.side_x * self.upscale_factor, self.side_y * self.upscale_factor),
                resize_ratio=self.resize_ratio,
            )
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
            base_pil_image = upsample_pil_image.resize(
                (self.side_x, self.side_y), resample=PIL.Image.BICUBIC
            )
            base_tensor = pil_image_to_norm_tensor(base_pil_image)
            return (
                tokens,
                mask,
                base_tensor,
                upsample_tensor,
            )

        base_pil_image = random_resized_crop(
            original_pil_image,
            (self.side_x, self.side_y),
            resize_ratio=self.resize_ratio,
        )
        # Crop the image to remove white padding
        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        base_tensor = trim_white_padding_tensor(base_tensor)
        
        # Verify image dimensions after trimming
        expected_shape = (3, self.resolution, self.resolution)
        if base_tensor.shape != expected_shape:
            print(f"WARNING: Image shape mismatch after trimming for {image_file}")
            print(f"  Expected: {expected_shape}, Got: {base_tensor.shape}")
            print(f"  Index: {ind}, Key: {key}")
            # Resize to correct dimensions
            if base_tensor.shape[0] == 3:  # Correct channels
                base_tensor = F.interpolate(
                    base_tensor.unsqueeze(0), 
                    size=(self.resolution, self.resolution), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                print(f"  -> Resized to {base_tensor.shape}")
            else:
                print("  -> Skipping due to incorrect channels")
                return self.skip_sample(ind)
                
        return tokens, mask, base_tensor
