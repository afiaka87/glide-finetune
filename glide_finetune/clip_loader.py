from glob import glob
import webdataset as wds  # pylint: disable=import-outside-toplevel
from tqdm import tqdm
from functools import lru_cache
import json
import io
import time
from pathlib import Path
from random import randint, choice, random

from typing import Tuple
import PIL
import numpy as np

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T

from glide_text2im.tokenizer.bpe import Encoder


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
        resize_ratio=0.75,
        shuffle=False,
        imagepreproc=None,
    ):
        super().__init__()
        folder = Path(folder)

        self.image_files = get_image_files_dict(folder)
        self.text_files = get_text_files_dict(folder)
        self.keys = get_shared_stems(self.image_files, self.text_files)
        print(f"Found {len(self.keys)} images.")
        print(f"Using {len(self.text_files)} text files.")

        self.resize_ratio = resize_ratio

        self.shuffle = shuffle
        self.prefix = folder
        self.imagepreproc =  imagepreproc

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
            return choice(descriptions).strip()
        except IndexError as zero_captions_in_file_ex:
            print(f"Exception: {zero_captions_in_file_ex}")
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

    def __getitem__(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]
        prompt = self.get_caption(ind)
        print(f"An exception occurred trying to load file {image_file}.")
        try:
            x_img = self.imagepreproc(PIL.Image.open(image_file).convert("RGB"))
            x_img = th.from_numpy(np.asarray(x_img)).float()
            # x_img = th.from_numpy(np.asarray(x_img)).float().permute(2, 0, 1)/ 127.5 - 1.
        except OSError as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            print(f"Exception: {e}")
            return self.skip_sample(ind)
        return prompt, x_img

def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    urls = "/mnt/10TB_HDD_OLDER/LAION/laion400m-dat-release/"
    archives = glob(f"{urls}*.tar")

    dataset = wds.WebDataset(archives, cache_dir=cache_path, cache_size=10 ** 10, handler=wds.handlers.warn_and_continue)
    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        else:
            metadata = json.loads(item["json"].decode("utf-8"))
            # if metadata["original_height"] != metadata["original_width"]:
            if metadata["NSFW"] in ["LIKELY", "NSFW"]:
                tqdm.write(f"Skipping image because it is not square.")
                tqdm.write(f"{metadata}")
                return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        prompt, x_img = None, None
        image_data = item[image_key]
        x_img = image_transform(PIL.Image.open(io.BytesIO(image_data)))
        x_img = th.from_numpy(np.asarray(x_img)).float().permute(2, 0, 1) / 127.5 - 1.
        text = item[caption_key]
        prompt = text.decode("utf-8")
        return prompt, x_img

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_stop)
    return transformed_dataset