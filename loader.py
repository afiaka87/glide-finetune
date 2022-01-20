
from functools import lru_cache
import os
from pathlib import Path
from random import randint, choice, random

from os.path import expanduser
from typing import Tuple
import PIL
import numpy as np

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T

from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer
from glide_text2im.model_creation import get_encoder
from glide_text2im.tokenizer.bpe import Encoder

DEFAULT_KEY_CACHE_PATH = Path(expanduser('~/.cache/glide/keys.txt'))
DEFAULT_IMAGES_CACHE_PATH = Path(expanduser('~/.cache/glide/images.txt'))
DEFAULT_CAPTIONS_CACHE_PATH = Path(expanduser('~/.cache/glide/captions.txt'))

@lru_cache(maxsize=1)
def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)

def get_tokens_and_mask(tokenizer: Encoder, prompt: str = '') -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, 128) # TODO: make this a parameter
        tokens = th.tensor(tokens)# + uncond_tokens)
        mask = th.tensor(mask, dtype=th.bool)# + uncond_mask, dtype=th.bool)
        return tokens, mask
    
def get_image_files_dict(base_path):
    image_files = [
        *base_path.glob('**/*.png'), *base_path.glob('**/*.jpg'),
        *base_path.glob('**/*.jpeg'), *base_path.glob('**/*.bmp')
    ]
    return {image_file.stem: image_file for image_file in image_files}

def get_text_files_dict(base_path):
    text_files = [*base_path.glob('**/*.txt')]
    return {text_file.stem: text_file for text_file in text_files}

def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)

def get_keys_from_cache(folder, image_cache_path, text_cache_path, keys_cache_path):
    image_cache_path, text_cache_path, keys_cache_path = Path(image_cache_path), Path(text_cache_path), Path(keys_cache_path)
    if not image_cache_path.exists(): 
        print(f'Creating image cache at {image_cache_path}...')
        image_files = get_image_files_dict(folder)
    else:
        print(f'Loading image cache from {image_cache_path}...')
        image_files = open(image_cache_path, 'r').read().splitlines()
        image_files = {Path(image_file).stem: image_file for image_file in image_files}

    if not text_cache_path.exists():
        print(f'Creating text cache at {text_cache_path}...')
        text_files = get_text_files_dict(folder)
    else:
        print(f'Loading text cache from {text_cache_path}...')
        text_files = open(text_cache_path, 'r').read().splitlines()
        text_files = {Path(text_file).stem: text_file for text_file in text_files}

    if not keys_cache_path.exists():
        print(f'Creating keys cache at {keys_cache_path}...')
        keys = get_shared_stems(image_files, text_files)
    else:
        print(f'Loading keys cache from {keys_cache_path}...')
        keys = open(keys_cache_path, 'r').read().splitlines()
    return keys, image_files, text_files

class TextImageDataset(Dataset):
    def __init__(self,
                 folder='',
                 side_x=64,
                 side_y=64,
                 resize_ratio=0.75,
                 shuffle=False,
                 tokenizer=None,
                 text_ctx_len=128,
                 uncond_p=0.0,
                 image_files_cache=DEFAULT_IMAGES_CACHE_PATH,
                 text_files_cache=DEFAULT_CAPTIONS_CACHE_PATH,
                 shared_keys_cache=DEFAULT_KEY_CACHE_PATH,
                 force_reload=False):
        super().__init__()
        image_cache_path = Path(image_files_cache)
        text_cache_path = Path(text_files_cache)
        keys_cache_path = Path(shared_keys_cache)
        folder = Path(folder)

        if force_reload:
            if image_cache_path.exists():
                os.remove(image_cache_path)
            if text_cache_path.exists():
                os.remove(text_cache_path)
            if keys_cache_path.exists():
                os.remove(keys_cache_path)

        keys, image_files, text_files = get_keys_from_cache(folder, image_cache_path, text_cache_path, keys_cache_path)
        self.keys = list(keys)
        self.image_files = image_files
        self.text_files = text_files
        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len

        self.shuffle = shuffle
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.imagepreproc = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize(int(self.side_x * self.resize_ratio)),
                T.RandomResizedCrop((self.side_x, self.side_y),
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.),
                                    interpolation=T.InterpolationMode.LANCZOS),
                T.ToTensor(),
            ])

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


    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]
        descriptions = open(text_file, 'r').readlines()
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions).strip()
            if random() < self.uncond_p:
                description = '' # uses the uncond tokens and masks
            tokens, mask = get_tokens_and_mask(
                tokenizer=self.tokenizer,
                prompt=description)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        try:
            x_img = PIL.Image.open(os.path.join(self.prefix, image_file))
            x_img = (self.imagepreproc(x_img)+1).round()/127.5 - 1
        except OSError as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        return tokens, mask, x_img