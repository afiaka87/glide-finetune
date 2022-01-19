
from functools import lru_cache
import os
from pathlib import Path
from random import randint, choice

from os.path import expanduser
import PIL
import numpy as np

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def show_image(batch):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])

@lru_cache(maxsize=1)
def get_keys(base_path):
    if os.path.isdir(base_path):
        text_files = [*base_path.glob('**/*.txt')]
        image_files = [
            *base_path.glob('**/*.png'), *base_path.glob('**/*.jpg'),
            *base_path.glob('**/*.jpeg'), *base_path.glob('**/*.bmp')
        ]
    
    text_files = {text_file.stem: text_file for text_file in text_files}
    image_files = {image_file.stem: image_file for image_file in image_files}
    keys = (image_files.keys() & text_files.keys())
    return list(keys), image_files, text_files

KEY_CACHE_PATH = Path(expanduser('~/.cache/glide/keys.txt'))

@lru_cache(maxsize=1)
def get_keys_from_cache(image_files_cache, text_files_cache):
    """
    Get the keys from a file containing the paths to the image and text files instead of globbing.

    Args:
        image_files_cache: The path to the image files cache. Must end in .txt
        text_files_cache: The path to the text files cache. Must end in .txt
    """
    image_files_cache, text_files_cache = Path(image_files_cache), Path(text_files_cache)

    print(f'Loading keys from cache: {image_files_cache}...')
    image_files = open(image_files_cache, 'r').read().splitlines()
    image_files = { Path(image_file).stem: Path(image_file) for image_file in image_files }

    print(f'Loading keys from cache: {text_files_cache}...')
    text_files = open(text_files_cache,'r').read().splitlines()
    text_files = { Path(text_file).stem: Path(text_file) for text_file in text_files }

    if not KEY_CACHE_PATH.exists():
        print(f"Finding keys from path basename/stem...")
        keys = (image_files.keys() & text_files.keys())
        KEY_CACHE_PATH.write_text('\n'.join(keys))
        return list(keys), image_files, text_files
    else:
        keys = open(KEY_CACHE_PATH, 'r').read().splitlines()
        print(f"Loaded {len(keys)} keys from cache")
        return keys, image_files, text_files

class TextImageDataset(Dataset):
    def __init__(self,
                 folder='',
                 batch_size=1,
                 side_x=64,
                 side_y=64,
                 image_files_cache='./image_files.txt',
                 text_files_cache='./text_files.txt',
                 shuffle=False,
                 device='cpu',
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        if '~' in folder:
            folder = expanduser(folder)
        image_cache_path = Path(image_files_cache)
        text_cache_path = Path(text_files_cache)
        folder = Path(folder)
        if image_cache_path.exists() and text_cache_path.exists():
            print(f"Found image and text cache files, loading from them...")
            self.keys, self.image_files, self.text_files = get_keys_from_cache(image_cache_path, text_cache_path)
        else:
            raise FileNotFoundError(f"Could not find image and text cache files, {image_cache_path} and {text_cache_path}")
        self.shuffle = shuffle
        self.device = device
        self.batch_size = batch_size
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y

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
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        
        try:
            x_img = PIL.Image.open(os.path.join(self.prefix, image_file))
            preprocess = lambda x: th.from_numpy(np.asarray(x.resize((self.side_x, self.side_y)).convert("RGB"))).unsqueeze(0).permute(0, 3, 1, 2) / 127. - 1.
            x_img = preprocess(x_img).to(self.device)
        except OSError as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        return description, x_img