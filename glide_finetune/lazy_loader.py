from pathlib import Path
from random import randint, random

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T
from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor
from glide_finetune.loader import random_resized_crop


def caption_from_filename(path):
    """Extract caption from filename like '1626461141_everything_about_hunting.jpg'.

    Strips extension, removes the leading timestamp (split on first '_'),
    and replaces remaining underscores with spaces.
    """
    stem = path.stem
    # Split on first underscore to remove timestamp
    parts = stem.split("_", 1)
    if len(parts) > 1:
        caption = parts[1]
    else:
        caption = parts[0]
    return caption.replace("_", " ").strip()


class LazyImageDataset(Dataset):
    """Dataset that extracts captions from filenames (timestamp_caption_words.jpg)."""

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
        enable_glide_upsample=False,
        upscale_factor=4,
        random_hflip=False,
    ):
        super().__init__()
        folder = Path(folder)

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in extensions
        )
        print(f"Found {len(self.image_files)} images in {folder}")

        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len
        self.shuffle = shuffle
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.enable_upsample = enable_glide_upsample
        self.upscale_factor = upscale_factor
        self.random_hflip = random_hflip
        self.color_jitter = T.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02
        )

    def __len__(self):
        return len(self.image_files)

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

    def __getitem__(self, ind):
        image_path = self.image_files[ind]

        if random() < self.uncond_p:
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            caption = caption_from_filename(image_path)
            tokens, mask = get_tokens_and_mask(tokenizer=self.tokenizer, prompt=caption)

        try:
            original_pil_image = PIL.Image.open(image_path).convert("RGB")
        except (OSError, ValueError):
            print(f"An exception occurred trying to load file {image_path}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        if random() < 0.5:
            original_pil_image = original_pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        original_pil_image = self.color_jitter(original_pil_image)

        if self.enable_upsample:
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
                tokens.clone(),
                mask.clone(),
                base_tensor,
                upsample_tensor,
            )

        base_pil_image = random_resized_crop(
            original_pil_image,
            (self.side_x, self.side_y),
            resize_ratio=self.resize_ratio,
        )
        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        return tokens.clone(), mask.clone(), base_tensor
