#!/usr/bin/env python3
"""
glide_wds_loader.py

WebDataset loader with automatic removal of white-padding *before* resizing,
plus a light random-centre crop so the model never sees the synthetic borders.
All tensors returned to the collator are fixed-size, so batching works out
of the box.
"""

import io
import json
import random
from typing import Tuple, Union

import PIL
import torch as th
import torchvision.transforms.functional as TF
import webdataset as wds
from PIL import Image

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor

# -----------------------------------------------------------------------------#
# -------------  white-padding removal (pure torch, no OpenCV) ----------------#
# -----------------------------------------------------------------------------#


def _trim_white_padding_tensor(
    img: th.Tensor,
    white_thresh: Union[int, float] = 245,
    morph_kernel: int | None = None,
) -> th.Tensor:
    """
    Remove uniform white padding that was added to make a square canvas.

    Args
    ----
    img          :  (C, H, W) uint8 tensor in [0, 255].
    white_thresh :  pixel values > white_thresh are treated as white.
    morph_kernel :  optional closing (odd int, e.g. 3 or 5) to bridge gaps.

    Returns
    -------
    Cropped tensor (C, h', w').  If no content is found, returns the input img.
    """
    assert img.ndim == 3 and img.dtype == th.uint8, "expect (C,H,W) uint8 tensor"

    # 1. binary mask where *any* channel is non-white
    content = (img < white_thresh).any(dim=0)  # (H,W) bool

    # 2. optional morphological closing to fill tiny holes
    if morph_kernel and morph_kernel > 1:
        pad = morph_kernel // 2
        content = (
            th.nn.functional.max_pool2d(
                content.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=morph_kernel,
                stride=1,
                padding=pad,
            )
            .squeeze()
            .bool()
        )

    # 3. bounding box of non-white rows / cols
    rows = th.where(content.any(dim=1))[0]
    cols = th.where(content.any(dim=0))[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return img  # all-white edge-case

    top, bottom = rows[0].item(), rows[-1].item() + 1
    left, right = cols[0].item(), cols[-1].item() + 1
    return img[:, int(top):int(bottom), int(left):int(right)]


def trim_white_padding_pil(pil_img: Image.Image, thresh: int = 245) -> Image.Image:
    """
    Convenience wrapper: PIL.Image → tensor trim → PIL.Image.
    """
    t = TF.pil_to_tensor(pil_img.convert("RGB"))  # (C,H,W) uint8
    t = _trim_white_padding_tensor(t, white_thresh=thresh)
    return TF.to_pil_image(t)  # back to PIL


# -----------------------------------------------------------------------------#
# ------------------  light random-centre crop (PIL)  -------------------------#
# -----------------------------------------------------------------------------#


def random_center_crop(
    img: Image.Image,
    min_scale: float = 0.7,
    jitter_frac: float = 0.1,
    out_size: Tuple[int, int] | None = None,
) -> Image.Image:
    """
    Take a crop roughly around the centre, with mild random scale + offset.

    * min_scale   : lower bound of the side-length as a fraction of original.
    * jitter_frac : how far (fraction of leftover border) the crop can shift.
    * out_size    : if given, resize the crop to this (w, h) with bicubic.

    Returns a PIL.Image.
    """
    w, h = img.size
    assert 0.0 < min_scale <= 1.0

    # choose random scale
    scale = random.uniform(min_scale, 1.0)
    crop_w = int(w * scale)
    crop_h = int(h * scale)

    # centre coordinates
    cx, cy = w // 2, h // 2
    left = cx - crop_w // 2
    top = cy - crop_h // 2

    # jitter within allowed range
    max_dx = int((w - crop_w) * jitter_frac)
    max_dy = int((h - crop_h) * jitter_frac)
    left += random.randint(-max_dx, max_dx)
    top += random.randint(-max_dy, max_dy)

    # clamp to image bounds
    left = max(0, min(left, w - crop_w))
    top = max(0, min(top, h - crop_h))

    crop = img.crop((left, top, left + crop_w, top + crop_h))
    if out_size is not None:
        crop = crop.resize(out_size, resample=PIL.Image.BICUBIC)
    return crop


# -----------------------------------------------------------------------------#
# ---------------------------  WebDataset loader  -----------------------------#
# -----------------------------------------------------------------------------#


def glide_wds_loader(
    urls,
    *,
    enable_text: bool = True,
    enable_image: bool = True,
    enable_metadata: bool = True,
    image_key: str = "jpg",
    caption_key: str = "txt",
    metadata_key: str = "json",
    cache_path: str | None = None,
    tokenizer=None,
    base_x: int = 64,
    base_y: int = 64,
    uncond_p: float = 0.2,
    nsfw_filter: bool = False,
    ar_lower: float = 0.5,
    ar_upper: float = 2.0,
    min_original_height: int = 64,
    min_original_width: int = 64,
    enable_upsample: bool = False,
    similarity_threshold_upper: float = 0.8,
    similarity_threshold_lower: float = 0.2,
    words_to_skip: list[str] | None = None,
    dataset_name: str = "laion",  # 'laion', 'alamy', or 'webdataset'
    upscale_factor: int = 4,
    laion_no_filter: bool = False,
):
    words_to_skip = words_to_skip or []

    # ------------------------------------------------------------------#
    #  dataset creation + optional filtering
    # ------------------------------------------------------------------#
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.reraise_exception,
    )

    def _select_laion(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False

        metadata = json.loads(item[metadata_key].decode("utf-8"))  # noqa: F841

        # NSFW filtering (categories are "UNLIKELY", "UNSURE", "NSFW" - also many)
        nsfw_likelihood: str = metadata.get("NSFW")
        if nsfw_filter and nsfw_likelihood is not None and nsfw_likelihood == "NSFW":
            return False

        # Similarity filtering
        similarity = float(metadata["similarity"])
        if (similarity < similarity_threshold_lower or 
            similarity > similarity_threshold_upper):
            return False

        # Aspect ratio filtering
        orig_h = float(metadata["original_height"])
        orig_w = float(metadata["original_width"])
        ar = orig_w / orig_h
        if orig_h < min_original_height or orig_w < min_original_width:
            return False
        if ar < ar_lower or ar > ar_upper:
            return False
        return True

    def _select_alamy(item):
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False
        metadata = json.loads(item[metadata_key].decode("utf-8"))
        if metadata.get("lc") != "en":
            return False
        return True

    if dataset_name == "laion" and not laion_no_filter:
        dataset = dataset.select(_select_laion)
    elif dataset_name == "alamy":
        dataset = dataset.select(_select_alamy)
    # else: keep as-is for webdataset or laion_no_filter

    # ------------------------------------------------------------------#
    #  per-sample preprocessing
    # ------------------------------------------------------------------#
    base_size = (base_x, base_y)
    up_size = (base_x * upscale_factor, base_y * upscale_factor)

    def _preprocess(item):
        # ---------- text conditioning ----------
        if not enable_text or random.random() < uncond_p:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            caption = item.get(caption_key, b"").decode("utf-8")
            if caption:
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
            else:
                tokens, mask = get_uncond_tokens_mask(tokenizer)

        # ---------- image processing ----------
        img_bytes = item[image_key]
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 1. remove white borders
        pil = trim_white_padding_pil(pil, thresh=245)

        # 2. light random-centre crop (only if large enough)
        if pil.width > base_x and pil.height > base_y:
            pil = random_center_crop(pil, min_scale=0.8, jitter_frac=0.1)

        # 3. resize to target sizes
        base_pil = pil.resize(base_size, resample=PIL.Image.BICUBIC)
        base_tensor = pil_image_to_norm_tensor(base_pil)  # (3,64,64) float32

        if enable_upsample:
            up_pil = pil.resize(up_size, resample=PIL.Image.BICUBIC)
            up_tensor = pil_image_to_norm_tensor(up_pil)
            return (
                th.tensor(tokens),
                th.tensor(mask, dtype=th.bool),
                base_tensor,
                up_tensor,
            )

        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor

    # warn_and_continue: skip corrupt samples, keep training rolling
    dataset = dataset.map(_preprocess, handler=wds.handlers.warn_and_continue)
    return dataset
