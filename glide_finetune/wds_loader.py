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
from typing import Any, Dict, Optional, Tuple

import PIL.Image
import torch as th
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.image_processing import (
    random_center_crop,
    trim_white_padding_pil,
)
from glide_finetune.train_util import pil_image_to_norm_tensor

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
                    nsfw_counts: Dict[str, int] = {}
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
                    status_counts: Dict[str, int] = {}
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
        pil = trim_white_padding_pil(pil, white_thresh=245)
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
