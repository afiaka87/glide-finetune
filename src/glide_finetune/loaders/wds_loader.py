import io
import json
from collections.abc import Iterator
from random import random
from typing import Any

import PIL
import PIL.Image
import torch as th
import webdataset as wds
from PIL import Image

from glide_finetune.utils.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.utils.image_processing import trim_white_padding_pil

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.train_util import pil_image_to_norm_tensor

# Initialize logger
logger = get_logger("glide_finetune.wds_loader")


def glide_wds_loader(
    urls: str | list[str],
    enable_text: bool = True,
    enable_image: bool = True,
    enable_metadata: bool = True,
    image_key: str = "jpg",
    caption_key: str = "txt",
    metadata_key: str = "json",
    cache_path: str | None = None,
    tokenizer: Any | None = None,
    base_x: int = 64,
    base_y: int = 64,
    uncond_p: float = 0.2,
    nsfw_filter: bool = True,
    ar_lower: float = 0.5,
    ar_upper: float = 2.0,
    min_original_height: int = 256,
    min_original_width: int = 256,
    enable_upsample: bool = False,
    similarity_threshold_upper: float = 1.0,  # Maximum similarity (inclusive)
    similarity_threshold_lower: float = 0.25,  # Minimum similarity (inclusive)
    words_to_skip: list[str] | None = None,
    dataset_name: str = "laion",  # can be laion, alamy, synthetic, webdataset, generic, or custom
    upscale_factor: int = 4,
    trim_white_padding: bool = False,
    white_thresh: int = 245,
    resampling_method: str = "bicubic",  # Resampling method for image resizing
    disable_laion_filters: bool = False,  # Disable all LAION quality/NSFW/similarity filters
) -> Iterator[tuple[th.Tensor, th.Tensor, th.Tensor]]:
    if words_to_skip is None:
        words_to_skip = []
    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))
    
    # Set resampling method
    if resampling_method.lower() == "lanczos":
        resample = Image.Resampling.LANCZOS
    else:  # default to bicubic
        resample = Image.Resampling.BICUBIC

    # Custom handler that warns about duplicates but continues
    def handle_duplicates(exn: Exception) -> bool:
        """Handle duplicate keys by logging and continuing."""
        if "duplicate" in str(exn):
            # Log the duplicate but continue processing
            logger.warning(f"Warning: Skipping duplicate file in tar: {exn}")
            return True  # Continue processing
        # For other exceptions, reraise
        return False

    # For distributed training, we need to split shards properly
    # WebDataset handles this via nodesplitter parameter
    # Note: We don't check dist.is_initialized() here because DataLoader workers
    # spawn fresh processes without the distributed context
    
    # Instead, we'll use resampled=True which handles distributed automatically
    # and works correctly with Accelerate
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=handle_duplicates,  # Use custom handler for duplicates
        resampled=True,  # This enables proper distributed sampling
        shardshuffle=True,  # Enable shard shuffling with resampled
    )
    
    # Log if we detect we're in distributed mode (main process only)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if rank == 0:  # Only log on main process
                logger.info(f"WebDataset configured for distributed training: {world_size} processes")
    except Exception:
        # Not in distributed mode, that's fine
        pass

    def filter_dataset_laion(item: dict[str, Any]) -> bool:
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False

        # If filters are disabled, accept all samples that have the required keys
        if disable_laion_filters:
            return True

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
        if similarity < similarity_threshold_lower or similarity > similarity_threshold_upper:
            return False
        if nsfw_filter and nsfw_rating in ["NSFW", "LIKELY"]:
            return False
        return not any(slur.lower() in caption for slur in words_to_skip)

    def filter_dataset_alamy(item: dict[str, Any]) -> bool:
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

    def filter_dataset_synthetic(item: dict[str, Any]) -> bool:
        if enable_image and "jpg" not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        metadata = json.loads(item["json"].decode("utf-8"))

        # Check if we have the required fields for synthetic dataset
        if enable_text and "short_caption" not in metadata and "long_caption" not in metadata:
            return False

        # Check image dimensions if available
        if "width" in metadata and "height" in metadata:
            width = float(metadata["width"])
            height = float(metadata["height"])
            aspect_ratio = width / height

            if width < min_original_width or height < min_original_height:
                return False
            if aspect_ratio < ar_lower or aspect_ratio > ar_upper:
                return False

        return True

    if dataset_name == "laion":
        if disable_laion_filters:
            logger.warning("⚠️  LAION filters DISABLED - accepting all samples with required keys!")
        filtered_dataset = dataset.select(filter_dataset_laion)
    elif dataset_name == "alamy":
        filtered_dataset = dataset.select(filter_dataset_alamy)
    elif dataset_name == "synthetic":
        filtered_dataset = dataset.select(filter_dataset_synthetic)
    elif dataset_name in ["webdataset", "generic", "custom"]:
        # Generic dataset - only check for image and caption keys, no other filtering
        def filter_dataset_generic(item: dict[str, Any]) -> bool:
            # Only require the keys that are actually needed
            if enable_image and image_key not in item:
                # Log detailed information about what keys are available
                available_keys = list(item.keys())
                logger.debug(f"Sample missing image key '{image_key}'. Available keys: {available_keys}")
                return False
            if enable_text and caption_key not in item:
                available_keys = list(item.keys())
                logger.debug(f"Sample missing caption key '{caption_key}'. Available keys: {available_keys}")
                return False
            # No metadata requirement for generic datasets
            return True
        filtered_dataset = dataset.select(filter_dataset_generic)
        
        # Check if all samples were filtered out
        sample_count = 0
        max_check = 100  # Check first 100 samples to see if any pass
        for i, sample in enumerate(filtered_dataset):
            sample_count += 1
            if i >= max_check - 1:
                break
        
        if sample_count == 0:
            # Try to get one sample from unfiltered dataset to show available keys
            try:
                first_sample = next(iter(dataset))
                available_keys = list(first_sample.keys())
                error_msg = (
                    f"All samples were filtered out! No samples have the required keys.\n"
                    f"  Looking for image_key='{image_key}' and caption_key='{caption_key}'\n"
                    f"  Available keys in dataset: {available_keys}\n"
                    f"  Dataset: {dataset_name}\n"
                    f"  Hint: Use --wds_image_key and --wds_caption_key to specify correct keys"
                )
            except Exception:
                error_msg = (
                    f"All samples were filtered out! No samples have the required keys.\n"
                    f"  Looking for image_key='{image_key}' and caption_key='{caption_key}'\n"
                    f"  Dataset: {dataset_name}\n"
                    f"  Hint: Use --wds_image_key and --wds_caption_key to specify correct keys"
                )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"WebDataset filter check passed - found valid samples with keys: image_key='{image_key}', caption_key='{caption_key}'")
    else:
        msg = f"Unknown dataset: {dataset_name}. Must be one of 'laion', 'alamy', 'synthetic', 'webdataset', 'generic', or 'custom'."
        raise ValueError(
            msg
        )

    def preprocess_dataset(item: dict[str, Any]) -> tuple[th.Tensor, th.Tensor, th.Tensor] | tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        tokens, mask, base_tensor, upsample_tensor = None, None, None, None

        # 20%, the empty token is used to represent the unconditional token.
        # This lets classifier-free guidance work after training.
        if not enable_text or random() < uncond_p:  # noqa: S311 - Pseudorandom appropriate for unconditional token dropout
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            # Handle different dataset formats for captions
            if dataset_name == "synthetic":
                # For synthetic dataset, caption is in the JSON metadata
                metadata = json.loads(item["json"].decode("utf-8"))
                # Prefer short_caption, fallback to long_caption
                caption = metadata.get("short_caption", metadata.get("long_caption", ""))
            elif dataset_name == "alamy":
                # For alamy dataset, caption is in the JSON metadata
                metadata = json.loads(item["json"].decode("utf-8"))
                caption = metadata.get("caption", "")
            else:
                # For laion dataset, caption is in separate txt file
                caption = item[caption_key].decode("utf-8")

            tokens, mask = get_tokens_and_mask(tokenizer, caption)

        image_data = item[image_key]
        original_pil_image: Image.Image = PIL.Image.open(io.BytesIO(image_data))

        # Apply white-padding removal if enabled
        if trim_white_padding:
            original_pil_image = trim_white_padding_pil(
                original_pil_image.convert("RGB"), white_thresh=white_thresh
            )
        else:
            original_pil_image = original_pil_image.convert("RGB")

        base_pil_image = original_pil_image.resize(base_image_shape, resample=resample)
        base_tensor = pil_image_to_norm_tensor(base_pil_image)  # type: ignore[no-untyped-call]

        # The upsample model needs both the base and the upsample images e.g. 64x64 and 256x256.
        if enable_upsample:
            upsample_pil_image = original_pil_image.resize(upsample_image_shape, resample=resample)
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)  # type: ignore[no-untyped-call]
            return (
                tokens.clone(),
                mask.clone(),
                base_tensor,
                upsample_tensor,
            )
        return tokens.clone(), mask.clone(), base_tensor

    return filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.reraise_exception
    )

