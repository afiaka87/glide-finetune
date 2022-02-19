import io
from random import random

import numpy as np
import PIL
import torch as th
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask


def glide_wds_loader(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    enable_metadata=True,
    image_key="jpg",
    caption_key="txt",
    cache_path=None,
    tokenizer=None,
    uncond_p=0.2,
):
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.reraise_exception,
    )

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        tokens, mask, x_img = None, None, None
        if enable_image:
            image_data = item[image_key]
            x_img = image_transform(PIL.Image.open(io.BytesIO(image_data)))
            x_img = (
                th.from_numpy(np.asarray(x_img)).float().permute(2, 0, 1) / 127.5 - 1.0
            )
        if enable_text:
            if random() < uncond_p:
                tokens, mask = get_uncond_tokens_mask(tokenizer)
            else:
                text = item[caption_key]
                caption = text.decode("utf-8")
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
        else:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        return tokens, mask, x_img

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.reraise_exception
    )
    return transformed_dataset
