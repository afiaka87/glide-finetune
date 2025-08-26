import random
import time
from pathlib import Path
from typing import Any

import PIL
import PIL.Image
import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T

from glide_finetune.utils.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.utils.image_processing import (
    trim_white_padding_pil,
)
from glide_finetune.clip_features_loader import load_clip_features

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.train_util import pil_image_to_norm_tensor

# Initialize logger
logger = get_logger("glide_finetune.loader")


def random_resized_crop(
    image: PIL.Image.Image, shape: tuple[int, int], resize_ratio: float = 1.0
) -> PIL.Image.Image:
    """
    Randomly resize and crop an image to a given size.

    Args:
        image: The image to be resized and cropped.
        shape: The desired output shape.
        resize_ratio: The ratio to resize the image.
    
    Returns:
        The transformed image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    return image_transform(image)


def get_image_files_dict(base_path: Path) -> dict[str, Path]:
    """Get dictionary of image files keyed by stem."""
    image_files = [
        *base_path.glob("**/*.png"),
        *base_path.glob("**/*.jpg"),
        *base_path.glob("**/*.jpeg"),
        *base_path.glob("**/*.bmp"),
    ]
    return {image_file.stem: image_file for image_file in image_files}


def get_text_files_dict(base_path: Path) -> dict[str, Path]:
    """Get dictionary of text files keyed by stem."""
    text_files = [*base_path.glob("**/*.txt")]
    return {text_file.stem: text_file for text_file in text_files}


def get_shared_stems(image_files_dict: dict[str, Path], text_files_dict: dict[str, Path]) -> list[str]:
    """Get list of stems that have both image and text files."""
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)


class TextImageDataset(Dataset):
    def __init__(
        self,
        folder: str = "",
        side_x: int = 64,
        side_y: int = 64,
        resize_ratio: float = 0.75,
        shuffle: bool = False,
        tokenizer: Any | None = None,  # Tokenizer type from glide_text2im
        text_ctx_len: int = 128,
        uncond_p: float = 0.0,
        use_captions: bool = False,
        enable_glide_upsample: bool = False,
        upscale_factor: int = 4,
        trim_white_padding: bool = False,
        white_thresh: int = 245,
        skip_samples: int = 0,  # Number of samples to skip for resumption
        resampling_method: str = "bicubic",  # Resampling method for image resizing
        clip_features_path: str | None = None,  # Path to precomputed CLIP features
    ) -> None:
        super().__init__()
        folder = Path(folder)

        self.image_files = get_image_files_dict(folder)
        if use_captions:
            self.text_files = get_text_files_dict(folder)
            self.keys = get_shared_stems(self.image_files, self.text_files)
            logger.info(f"Found {len(self.keys)} images.")
            logger.info(f"Using {len(self.text_files)} text files.")
        else:
            self.text_files = None
            self.keys = list(self.image_files.keys())
            logger.info(f"Found {len(self.keys)} images.")
            logger.info("NOT using text files. Restart with --use_captions to enable...")
            time.sleep(3)

        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len

        self.shuffle = shuffle
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.enable_upsample = enable_glide_upsample
        self.upscale_factor = upscale_factor
        self.trim_white_padding = trim_white_padding
        self.white_thresh = white_thresh
        
        # Set resampling method
        if resampling_method.lower() == "lanczos":
            self.resample = PIL.Image.LANCZOS
        else:  # default to bicubic
            self.resample = PIL.Image.BICUBIC
        
        # Load precomputed CLIP features if provided
        self.clip_features_loader = None
        if clip_features_path:
            self.clip_features_loader = load_clip_features(clip_features_path)
            if self.clip_features_loader:
                logger.info(f"Loaded CLIP features from {clip_features_path}")
                logger.info(f"CLIP dimension: {self.clip_features_loader.clip_dim}")
            else:
                logger.warning(f"Failed to load CLIP features from {clip_features_path}")

        # Handle dataset resumption by rotating keys
        self.skip_samples = skip_samples
        if skip_samples > 0:
            # Rotate the keys list to start from the skip position
            skip_idx = skip_samples % len(self.keys)
            self.keys = self.keys[skip_idx:] + self.keys[:skip_idx]
            logger.info(
                f"📊 Rotated dataset to skip {skip_samples} samples (starting at index {skip_idx})"
            )

    def __len__(self) -> int:
        return len(self.keys)

    def random_sample(self) -> tuple[th.Tensor, ...]:
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind: int) -> tuple[th.Tensor, ...]:
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind: int) -> tuple[th.Tensor, ...]:
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_caption(self, ind: int) -> tuple[th.Tensor, th.Tensor]:
        key = self.keys[ind]
        text_file = self.text_files[key]
        descriptions = open(text_file).readlines()
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions).strip()
            return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
        except IndexError:
            logger.info(f"An exception occurred trying to load file {text_file}.")
            logger.info(f"Skipping index {ind}")
            return self.skip_sample(ind)

    def __getitem__(self, ind: int) -> tuple[th.Tensor, ...]:
        key = self.keys[ind]
        image_file = self.image_files[key]
        if self.text_files is None or random.random() < self.uncond_p:  # noqa: S311 - Pseudorandom is appropriate for ML training
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            tokens, mask = self.get_caption(ind)

        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError):
            logger.info(f"An exception occurred trying to load file {image_file}.")
            logger.info(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Apply white-padding removal if enabled
        if self.trim_white_padding:
            original_pil_image = trim_white_padding_pil(
                original_pil_image, white_thresh=self.white_thresh
            )
        
        # Load CLIP features if available
        clip_features = None
        if self.clip_features_loader:
            clip_features = self.clip_features_loader.get_feature(key)
            if clip_features is None:
                # Feature not found for this image
                # Create a zero tensor as placeholder - training code should handle this
                clip_features = th.zeros(self.clip_features_loader.clip_dim)

        if (
            self.enable_upsample
        ):  # the base image used should be derived from the cropped high-resolution image.
            upsample_pil_image = random_resized_crop(
                original_pil_image,
                (self.side_x * self.upscale_factor, self.side_y * self.upscale_factor),
                resize_ratio=self.resize_ratio,
            )
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
            base_pil_image = upsample_pil_image.resize(
                (self.side_x, self.side_y), resample=self.resample
            )
            base_tensor = pil_image_to_norm_tensor(base_pil_image)
            if clip_features is not None:
                return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, upsample_tensor, clip_features
            return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, upsample_tensor

        base_pil_image = random_resized_crop(
            original_pil_image, (self.side_x, self.side_y), resize_ratio=self.resize_ratio
        )
        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        if clip_features is not None:
            return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, clip_features
        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor
