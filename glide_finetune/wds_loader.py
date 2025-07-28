import io
import json
from random import random

import PIL
import torch as th
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor


def glide_wds_loader(
    urls,
    enable_text=True,
    enable_image=True,
    enable_metadata=True,
    image_key="jpg",
    caption_key="txt",
    metadata_key="json",
    cache_path=None,
    tokenizer=None,
    base_x=64,
    base_y=64,
    uncond_p=0.2,
    nsfw_filter=True,
    ar_lower=0.5,
    ar_upper=2.0,
    min_original_height=256,
    min_original_width=256,
    enable_upsample=False,
    similarity_threshold_upper=0.0,
    similarity_threshold_lower=0.5,
    words_to_skip=[],
    dataset_name="laion",  # can be laion, alamy.
    upscale_factor=4,
    laion_no_filter=False,
):
    print(f"[WebDataset] Creating loader with dataset_name='{dataset_name}', laion_no_filter={laion_no_filter}")
    print(f"[WebDataset] URLs: {urls[:3] if isinstance(urls, list) else urls} (total: {len(urls) if isinstance(urls, list) else 'unknown'} files)")
    
    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))
    
    print("[WebDataset] Creating WebDataset object...")
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.reraise_exception,
    )
    print("[WebDataset] WebDataset object created")

    def filter_dataset_laion(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False

        metadata = json.loads(item["json"].decode("utf-8"))

        similarity = float(metadata["similarity"])
        original_height = float(metadata["original_height"])
        original_width = float(metadata["original_width"])
        aspect_ratio = original_width / original_height
        caption = item[caption_key].decode("utf-8").lower()
        nsfw_rating = metadata["NSFW"]

        if original_height < min_original_height or original_width < min_original_width:
            return False
        if aspect_ratio < ar_lower or aspect_ratio > ar_upper:
            return False
        if (
            similarity < similarity_threshold_lower
            or similarity > similarity_threshold_upper
        ):
            return False
        if nsfw_filter and nsfw_rating in ["NSFW", "LIKELY"]:
            return False
        if any(slur.lower() in caption for slur in words_to_skip):
            return False
        return True

    def filter_dataset_alamy(item):
        if enable_image and "jpg" not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        metadata = json.loads(item["json"].decode("utf-8"))
        language_code = metadata["lc"]
        if language_code != "en":
            return False
        if enable_text and "caption" not in metadata:
            return False
        return True  # all good

    print(f"[WebDataset] Applying filtering for dataset_name='{dataset_name}'...")
    if dataset_name == "laion" and not laion_no_filter:
        filtered_dataset = dataset.select(filter_dataset_laion)
    elif dataset_name == "laion" and laion_no_filter:
        # Skip LAION filtering when laion_no_filter is True
        print("[WebDataset] LAION filtering disabled (--laion_no_filter)")
        filtered_dataset = dataset
    elif dataset_name == "alamy":
        filtered_dataset = dataset.select(filter_dataset_alamy)
    elif dataset_name == "webdataset":
        # No filtering - use dataset as-is
        print("[WebDataset] No filtering applied (webdataset mode)")
        filtered_dataset = dataset
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of 'laion', 'alamy', or 'webdataset'."
        )
    print("[WebDataset] Filtering complete")

    def preprocess_dataset(item):
        tokens, mask, base_tensor, upsample_tensor = None, None, None, None

        # 20%, the empty token is used to represent the unconditional token.
        # This lets classifier-free guidance work after training.
        if not enable_text or random() < uncond_p:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            caption = item[caption_key].decode("utf-8")
            tokens, mask = get_tokens_and_mask(tokenizer, caption)

        image_data = item[image_key]
        original_pil_image = PIL.Image.open(io.BytesIO(image_data))

        base_pil_image = original_pil_image.resize(
            base_image_shape, resample=PIL.Image.BICUBIC
        ).convert("RGB")
        base_tensor = pil_image_to_norm_tensor(base_pil_image)

        # Upsample model needs both base and upsample images e.g. 64x64 and 256x256
        if enable_upsample:
            upsample_pil_image = original_pil_image.resize(
                upsample_image_shape
            ).convert("RGB")
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
            return (
                th.tensor(tokens),
                th.tensor(mask, dtype=th.bool),
                base_tensor,
                upsample_tensor,
            )
        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor

    print("[WebDataset] Applying preprocessing transform...")
    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.reraise_exception
    )
    print("[WebDataset] Preprocessing transform applied")
    return transformed_dataset
