import time
from pathlib import Path
from random import randint, choice, random

import PIL

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T
from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
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
        random_hflip=False,
        random_brightness=False,
        random_contrast=False,
        random_color_jitter=False,
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
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.enable_upsample = enable_glide_upsample
        self.upscale_factor = upscale_factor
        self.random_hflip = random_hflip
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_color_jitter = random_color_jitter

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
        descriptions = open(text_file, "r").readlines()
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions).strip()
            return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
        except IndexError:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

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

        # Apply random horizontal flip if enabled
        if self.random_hflip and random() < 0.5:
            original_pil_image = original_pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Apply color augmentations if enabled
        if self.random_brightness and random() < 0.5:
            from torchvision.transforms import functional as TF
            brightness_factor = 0.5 + random() * 1.0  # 0.5 to 1.5
            original_pil_image = TF.adjust_brightness(original_pil_image, brightness_factor)

        if self.random_contrast and random() < 0.5:
            from torchvision.transforms import functional as TF
            contrast_factor = 0.5 + random() * 1.0  # 0.5 to 1.5
            original_pil_image = TF.adjust_contrast(original_pil_image, contrast_factor)

        if self.random_color_jitter and random() < 0.5:
            from torchvision.transforms import functional as TF
            saturation_factor = 0.5 + random() * 1.0  # 0.5 to 1.5
            hue_factor = -0.1 + random() * 0.2  # -0.1 to 0.1
            original_pil_image = TF.adjust_saturation(original_pil_image, saturation_factor)
            original_pil_image = TF.adjust_hue(original_pil_image, hue_factor)

        if self.enable_upsample:  # the base image used should be derived from the cropped high-resolution image.
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
