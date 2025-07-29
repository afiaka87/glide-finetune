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
import time
from typing import Any, Dict, Optional, Tuple, Union

import PIL.Image
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
    return img[:, int(top) : int(bottom), int(left) : int(right)]


def trim_white_padding_pil(pil_img: Image.Image, thresh: int = 245) -> Image.Image:
    """
    Convenience wrapper: PIL.Image → tensor trim → PIL.Image.
    """
    t = TF.pil_to_tensor(pil_img.convert("RGB"))  # (C,H,W) uint8
    t = _trim_white_padding_tensor(t, white_thresh=thresh)
    result: Image.Image = TF.to_pil_image(t)  # back to PIL
    return result


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
# ----------------------  WebDataset Statistics Tracking  ---------------------#
# -----------------------------------------------------------------------------#


class WebDatasetStats:
    """Track statistics for WebDataset loading and preprocessing."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.samples_processed = 0
        self.samples_skipped = 0
        self.uncond_count = 0
        self.caption_empty_count = 0
        self.filter_rejected_count = 0
        self.preprocessing_errors = 0
        self.total_processing_time = 0.0
        self.batch_count = 0

        # Image statistics
        self.original_sizes: list[tuple[int, int]] = []
        self.aspect_ratios: list[float] = []
        self.white_padding_removed_count = 0
        self.random_crop_applied_count = 0

        # Timing statistics
        self.load_times: list[float] = []
        self.preprocess_times: list[float] = []

        # Metadata statistics (for LAION)
        self.nsfw_filtered_count = 0
        self.similarity_filtered_count = 0
        self.size_filtered_count = 0
        self.ar_filtered_count = 0

        # Metadata field tracking
        self.metadata_fields: Dict[str, list[Any]] = {}

    def update_sample(
        self,
        processed: bool,
        processing_time: float,
        original_size: Optional[tuple[int, int]] = None,
        aspect_ratio: Optional[float] = None,
        is_uncond: bool = False,
        caption_empty: bool = False,
        white_padding_removed: bool = False,
        random_crop_applied: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update statistics for a single sample."""
        if processed:
            self.samples_processed += 1
            self.preprocess_times.append(processing_time)

            if is_uncond:
                self.uncond_count += 1
            if caption_empty:
                self.caption_empty_count += 1
            if white_padding_removed:
                self.white_padding_removed_count += 1
            if random_crop_applied:
                self.random_crop_applied_count += 1

            if original_size is not None:
                self.original_sizes.append(original_size)
            if aspect_ratio is not None:
                self.aspect_ratios.append(aspect_ratio)

            # Track metadata fields
            if metadata is not None:
                for key, value in metadata.items():
                    if key not in self.metadata_fields:
                        self.metadata_fields[key] = []
                    self.metadata_fields[key].append(value)
        else:
            self.samples_skipped += 1

    def update_filter_rejection(self, reason: str) -> None:
        """Update filter rejection statistics."""
        self.filter_rejected_count += 1
        if reason == "nsfw":
            self.nsfw_filtered_count += 1
        elif reason == "similarity":
            self.similarity_filtered_count += 1
        elif reason == "size":
            self.size_filtered_count += 1
        elif reason == "aspect_ratio":
            self.ar_filtered_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        total_samples = self.samples_processed + self.samples_skipped

        stats = {
            "samples_processed": self.samples_processed,
            "samples_skipped": self.samples_skipped,
            "total_samples": total_samples,
            "processing_rate": self.samples_processed / total_samples
            if total_samples > 0
            else 0.0,
            "uncond_rate": self.uncond_count / self.samples_processed
            if self.samples_processed > 0
            else 0.0,
            "caption_empty_rate": self.caption_empty_count / self.samples_processed
            if self.samples_processed > 0
            else 0.0,
            "white_padding_removal_rate": (
                self.white_padding_removed_count / self.samples_processed
                if self.samples_processed > 0
                else 0.0
            ),
            "random_crop_rate": (
                self.random_crop_applied_count / self.samples_processed
                if self.samples_processed > 0
                else 0.0
            ),
        }

        # Add filtering statistics if applicable
        if self.filter_rejected_count > 0:
            stats.update(
                {
                    "filter_rejected_total": self.filter_rejected_count,
                    "nsfw_filtered": self.nsfw_filtered_count,
                    "similarity_filtered": self.similarity_filtered_count,
                    "size_filtered": self.size_filtered_count,
                    "ar_filtered": self.ar_filtered_count,
                }
            )

        # Add timing statistics
        if self.preprocess_times:
            stats.update(
                {
                    "avg_preprocess_time_ms": (
                        sum(self.preprocess_times) / len(self.preprocess_times) * 1000
                    ),
                    "total_processing_time_s": sum(self.preprocess_times),
                }
            )

        # Add image size statistics
        if self.original_sizes:
            avg_width = sum(w for w, _ in self.original_sizes) / len(
                self.original_sizes
            )
            avg_height = sum(h for _, h in self.original_sizes) / len(
                self.original_sizes
            )
            stats.update(
                {
                    "avg_original_width": avg_width,
                    "avg_original_height": avg_height,
                }
            )

        if self.aspect_ratios:
            stats["avg_aspect_ratio"] = sum(self.aspect_ratios) / len(
                self.aspect_ratios
            )

        # Add metadata field statistics
        if self.metadata_fields:
            metadata_stats = {}
            for field, values in self.metadata_fields.items():
                if not values:
                    continue

                # For numeric fields, calculate statistics
                if field in [
                    "similarity",
                    "width",
                    "height",
                    "original_width",
                    "original_height",
                    "key",
                    "shard_id",
                ]:
                    numeric_values = [
                        v
                        for v in values
                        if v is not None and isinstance(v, (int, float))
                    ]
                    if numeric_values:
                        metadata_stats[f"metadata_{field}_avg"] = sum(
                            numeric_values
                        ) / len(numeric_values)
                        metadata_stats[f"metadata_{field}_min"] = min(numeric_values)
                        metadata_stats[f"metadata_{field}_max"] = max(numeric_values)

                # For NSFW field, count categories
                elif field == "NSFW":
                    nsfw_counts = {}
                    for v in values:
                        if v is not None:
                            nsfw_counts[str(v)] = nsfw_counts.get(str(v), 0) + 1
                    metadata_stats["metadata_nsfw_distribution"] = nsfw_counts

                # For LICENSE field, count unique licenses
                elif field == "LICENSE":
                    unique_licenses = set(v for v in values if v is not None)
                    metadata_stats["metadata_unique_licenses"] = len(unique_licenses)

                # For status field, count success/failure
                elif field == "status":
                    status_counts = {}
                    for v in values:
                        if v is not None:
                            status_counts[str(v)] = status_counts.get(str(v), 0) + 1
                    metadata_stats["metadata_status_distribution"] = status_counts

            stats.update(metadata_stats)

        return stats


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
) -> Tuple[Any, WebDatasetStats]:
    words_to_skip = words_to_skip or []

    # Create statistics tracker
    stats = WebDatasetStats()

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
            stats.update_filter_rejection("nsfw")
            return False

        # Similarity filtering
        similarity = float(metadata["similarity"])
        if (
            similarity < similarity_threshold_lower
            or similarity > similarity_threshold_upper
        ):
            stats.update_filter_rejection("similarity")
            return False

        # Aspect ratio filtering
        orig_h = float(metadata["original_height"])
        orig_w = float(metadata["original_width"])
        ar = orig_w / orig_h
        if orig_h < min_original_height or orig_w < min_original_width:
            stats.update_filter_rejection("size")
            return False
        if ar < ar_lower or ar > ar_upper:
            stats.update_filter_rejection("aspect_ratio")
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
        start_time = time.time()

        # Extract metadata if available
        metadata = None
        if enable_metadata and metadata_key in item:
            try:
                metadata = json.loads(item[metadata_key].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                metadata = None

        # ---------- text conditioning ----------
        is_uncond = False
        caption_empty = False

        if not enable_text or random.random() < uncond_p:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
            is_uncond = True
        else:
            caption = item.get(caption_key, b"").decode("utf-8")
            if caption:
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
            else:
                tokens, mask = get_uncond_tokens_mask(tokenizer)
                caption_empty = True

        # ---------- image processing ----------
        img_bytes = item[image_key]
        pil = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Track original size
        original_size = (pil.width, pil.height)
        aspect_ratio = pil.width / pil.height

        # 1. remove white borders
        original_pil = pil
        pil = trim_white_padding_pil(pil, thresh=245)
        white_padding_removed = pil.size != original_pil.size

        # 2. light random-centre crop (only if large enough)
        random_crop_applied = False
        if pil.width > base_x and pil.height > base_y:
            pil = random_center_crop(pil, min_scale=0.8, jitter_frac=0.1)
            random_crop_applied = True

        # 3. resize to target sizes
        base_pil = pil.resize(base_size, resample=PIL.Image.BICUBIC)
        base_tensor = pil_image_to_norm_tensor(base_pil)  # (3,64,64) float32

        # Update statistics
        processing_time = time.time() - start_time
        stats.update_sample(
            processed=True,
            processing_time=processing_time,
            original_size=original_size,
            aspect_ratio=aspect_ratio,
            is_uncond=is_uncond,
            caption_empty=caption_empty,
            white_padding_removed=white_padding_removed,
            random_crop_applied=random_crop_applied,
            metadata=metadata,
        )

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
    return dataset, stats
