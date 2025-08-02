import os
import time
from pathlib import Path
from random import choice, randint, random

import PIL
from torch.utils.data import Dataset
from torchvision import transforms as T

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.image_processing import (
    preprocess_image_with_padding_removal,
    random_center_crop,
    trim_white_padding_pil,
)
from glide_finetune.train_util import pil_image_to_norm_tensor


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
        # CLIP embedding cache parameters
        use_clip_cache=False,
        clip_model_name="ViT-L/14",
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

        # CLIP embedding cache
        self.use_clip_cache = use_clip_cache
        self.clip_model_name = clip_model_name
        self.clip_cache_stats = {"hits": 0, "misses": 0, "errors": 0}

        if use_clip_cache:
            print(f"CLIP embedding cache enabled for model: {clip_model_name}")
            # Count available clip files
            clip_files = list(folder.glob("**/*.clip"))
            print(f"Found {len(clip_files)} cached CLIP embeddings")

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
                print(
                    f"No valid captions in file: {text_file} (index {ind}, key {key})"
                )
                print("  -> Using unconditional tokens")
                return get_uncond_tokens_mask(self.tokenizer)

            description = choice(descriptions).strip()
            return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
        except Exception as e:
            print(f"Error reading caption file: {text_file} (index {ind}, key {key})")
            print(f"  Error: {type(e).__name__}: {e}")
            print("  -> Using unconditional tokens")
            return get_uncond_tokens_mask(self.tokenizer)

    def load_clip_embedding(self, ind):
        """Load CLIP embedding from cache file if available."""
        if not self.use_clip_cache:
            return None

        key = self.keys[ind]
        if not self.text_files or key not in self.text_files:
            return None

        # Get corresponding .clip file path
        text_file = self.text_files[key]
        clip_file = text_file.with_suffix(".clip")

        if not clip_file.exists():
            self.clip_cache_stats["misses"] += 1
            return None

        try:
            import torch

            clip_data = torch.load(clip_file, map_location="cpu")

            # Validate clip model matches
            if clip_data.get("clip_model") != self.clip_model_name:
                self.clip_cache_stats["misses"] += 1
                return None

            # Return embedding tensor
            embedding = clip_data["embedding"]
            self.clip_cache_stats["hits"] += 1
            return embedding

        except Exception as e:
            self.clip_cache_stats["errors"] += 1
            print(f"Error loading CLIP embedding from {clip_file}: {e}")
            return None

    def __getitem__(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]
        if self.text_files is None or random() < self.uncond_p:
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            tokens, mask = self.get_caption(ind)

        # Load CLIP embedding if available
        clip_embedding = self.load_clip_embedding(ind)

        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError):
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        if self.enable_upsample:
            # First apply white padding removal
            trimmed_pil = trim_white_padding_pil(original_pil_image, white_thresh=245)

            # Apply random center crop if image is large enough
            if trimmed_pil.width > self.side_x and trimmed_pil.height > self.side_y:
                trimmed_pil = random_center_crop(
                    trimmed_pil, min_scale=self.resize_ratio, jitter_frac=0.1
                )

            # Create high-res version for upsampling
            upsample_pil_image = trimmed_pil.resize(
                (self.side_x * self.upscale_factor, self.side_y * self.upscale_factor),
                resample=PIL.Image.BICUBIC,
            )
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)

            # Create base low-res version
            base_pil_image = trimmed_pil.resize(
                (self.side_x, self.side_y), resample=PIL.Image.BICUBIC
            )
            base_tensor = pil_image_to_norm_tensor(base_pil_image)

            if self.use_clip_cache:
                return (
                    tokens,
                    mask,
                    base_tensor,
                    upsample_tensor,
                    clip_embedding,
                )
            else:
                return (
                    tokens,
                    mask,
                    base_tensor,
                    upsample_tensor,
                )

        # Apply preprocessing: trim white padding first, then resize
        processed_pil_image = preprocess_image_with_padding_removal(
            original_pil_image,
            target_size=(self.side_x, self.side_y),
            white_thresh=245,
            use_random_crop=True,
            min_crop_scale=self.resize_ratio,  # Use resize_ratio as min crop scale
            crop_jitter=0.1,
        )

        # Convert to tensor
        base_tensor = pil_image_to_norm_tensor(processed_pil_image)

        if self.use_clip_cache:
            return tokens, mask, base_tensor, clip_embedding
        else:
            return tokens, mask, base_tensor

    def get_clip_cache_stats(self):
        """Get CLIP embedding cache statistics."""
        return self.clip_cache_stats.copy()
