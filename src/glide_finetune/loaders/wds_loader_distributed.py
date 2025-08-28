"""
Distributed WebDataset loader for multi-GPU training.

Handles proper sharding across multiple GPUs to ensure each sample is seen exactly once.
"""

import io
import json
import random
from typing import Any

import torch as th
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader

from glide_finetune.utils.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.utils.image_processing import trim_white_padding_pil
from glide_finetune.utils.train_util import pil_image_to_norm_tensor

# Augmentations module not yet implemented
# from glide_finetune.utils.augmentations import create_laion_augmentor, DiffusionAugmentor


def distributed_wds_loader(
    urls: str | list[str],
    batch_size: int = 1,
    side_x: int = 64,
    side_y: int = 64,
    resize_ratio: float = 1.0,
    uncond_p: float = 0.0,
    image_key: str = "jpg",
    caption_key: str = "txt",
    enable_metadata: bool = True,
    metadata_key: str = "json",
    enable_upsample: bool = False,
    upscale_factor: int = 4,
    wds_dataset_name: str | None = None,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    seed: int = 0,
    epoch_samples: int | None = None,
    tokenizer: Any | None = None,  # Add tokenizer parameter
    trim_white_padding: bool = False,
    white_thresh: int = 245,
    use_augmentations: bool = True,
    augmentor: Any | None = None,  # DiffusionAugmentor when implemented
) -> DataLoader[Any]:
    """
    Create a distributed WebDataset loader for multi-GPU training.

    Args:
        urls: Path pattern or list of paths to tar files
        batch_size: Batch size per GPU
        side_x, side_y: Target image dimensions
        resize_ratio: Random resize ratio for augmentation
        uncond_p: Probability of dropping captions for CFG
        image_key: Key for images in WebDataset
        caption_key: Key for captions in WebDataset
        enable_metadata: Whether to load metadata
        metadata_key: Key for metadata in WebDataset
        enable_upsample: Whether to prepare data for upsampling
        upscale_factor: Factor for upsampling
        wds_dataset_name: Name of dataset for special handling
        world_size: Number of GPUs/processes
        rank: Current GPU/process rank
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Keep workers alive between epochs
        seed: Random seed for shuffling
        epoch_samples: Number of samples per epoch (for infinite datasets)
        tokenizer: Tokenizer for text encoding
        trim_white_padding: Whether to remove white padding from images
        white_thresh: Threshold for white detection (0-255)
        use_augmentations: Whether to apply augmentations
        augmentor: Custom augmentor instance (if None, uses LAION defaults)

    Returns:
        DataLoader for distributed training
    """

    # Convert single URL to list - handle properly
    if isinstance(urls, str):
        if "*" in urls or "{" in urls:
            # It's a pattern, expand it
            from glob import glob

            urls = sorted(glob(urls))
        else:
            urls = [urls]
    elif not isinstance(urls, list):
        # This should never be reached given the type hints but handle it
        pass  # urls is already a list

    # For distributed training, we need to ensure each worker gets different shards
    # WebDataset's nodesplitter handles this automatically
    if world_size > 1:
        # Each process will automatically get different shards
        # The nodesplitter ensures no overlap
        dataset = wds.WebDataset(
            urls,
            nodesplitter=wds.split_by_node,  # Automatically splits by rank
            shardshuffle=True,  # Shuffle shards
        )
    else:
        # Single GPU, use all shards
        dataset = wds.WebDataset(urls, shardshuffle=True)

    # Shuffle samples within shards
    dataset = dataset.shuffle(1000)

    # Decode the raw WebDataset samples first
    dataset = dataset.decode("pil")

    # Create augmentor if needed
    if use_augmentations and augmentor is None:
        # Augmentations not yet implemented
        # augmentor = create_laion_augmentor()
        augmentor = None

    # Process decoded samples for training
    def decode_sample(sample: dict[str, Any]) -> tuple[str, th.Tensor] | tuple[str, th.Tensor, th.Tensor]:
        """Decode and process a single sample."""
        # Get image (already decoded to PIL by .decode("pil"))
        if image_key in sample:
            image = sample[image_key]
            if not isinstance(image, Image.Image):
                # Fallback if not a PIL image
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")
        else:
            msg = f"Image key '{image_key}' not found in sample"
            raise KeyError(msg)

        # Get caption (already decoded to string by .decode("pil"))
        caption = ""
        if caption_key in sample:
            caption = sample[caption_key]
            if isinstance(caption, bytes):
                caption = caption.decode("utf-8")
            elif not isinstance(caption, str):
                caption = str(caption)

        # Handle dataset-specific caption formats
        if wds_dataset_name == "synthetic":
            # Synthetic dataset has multiple caption types
            if metadata_key in sample:
                try:
                    metadata_raw = sample[metadata_key]
                    if isinstance(metadata_raw, bytes):
                        metadata = json.loads(metadata_raw.decode("utf-8"))
                    elif isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw)
                    else:
                        metadata = metadata_raw  # Already decoded dict
                    # Prefer long_caption, fallback to short_caption
                    caption = metadata.get("long_caption", metadata.get("short_caption", caption))
                except Exception:
                    pass
        elif wds_dataset_name == "laion":
            # LAION captions are in the txt field
            pass
        elif wds_dataset_name == "alamy":
            # Alamy might have structured metadata
            if metadata_key in sample:
                try:
                    metadata_raw = sample[metadata_key]
                    if isinstance(metadata_raw, bytes):
                        metadata = json.loads(metadata_raw.decode("utf-8"))
                    elif isinstance(metadata_raw, str):
                        metadata = json.loads(metadata_raw)
                    else:
                        metadata = metadata_raw  # Already decoded dict
                    caption = metadata.get("caption", caption)
                except Exception:
                    pass

        # Random caption dropping for CFG
        if uncond_p > 0 and random.random() < uncond_p:  # noqa: S311 - Pseudorandom appropriate for unconditional token dropout
            caption = ""

        # Apply white padding trimming if enabled
        if trim_white_padding:
            image = trim_white_padding_pil(image, white_thresh=white_thresh)

        # Process image
        if enable_upsample:
            # For upsampling, we need low-res and high-res versions
            high_res = image

            # Create low-res version
            low_res_size = (side_x, side_y)
            low_res = high_res.resize(low_res_size, Image.Resampling.LANCZOS)

            # Ensure high-res is the right size
            high_res_size = (side_x * upscale_factor, side_y * upscale_factor)
            if high_res.size != high_res_size:
                high_res = high_res.resize(high_res_size, Image.Resampling.LANCZOS)

            # Convert to tensors
            low_res_tensor = pil_image_to_norm_tensor(low_res)  # type: ignore[no-untyped-call]
            high_res_tensor = pil_image_to_norm_tensor(high_res)  # type: ignore[no-untyped-call]

            return caption, low_res_tensor, high_res_tensor
        # Regular training
        target_size = (side_x, side_y)

        # Apply augmentations if enabled
        if use_augmentations and augmentor is not None:
            # Apply PIL-based augmentations (includes crop, flip, color jitter)
            image = augmentor.augment_pil(image, target_size)
        else:
            # Fallback to basic processing
            # Random resize for augmentation
            if resize_ratio < 1.0:
                crop_ratio = random.uniform(resize_ratio, 1.0)  # noqa: S311 - Pseudorandom appropriate for data augmentation
                width, height = image.size
                crop_size = (int(width * crop_ratio), int(height * crop_ratio))

                # Random crop position
                left = random.randint(0, width - crop_size[0])  # noqa: S311 - Pseudorandom appropriate for data augmentation
                top = random.randint(0, height - crop_size[1])  # noqa: S311 - Pseudorandom appropriate for data augmentation
                image = image.crop((left, top, left + crop_size[0], top + crop_size[1]))

            # Resize to target size
            image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Random horizontal flip
            if random.random() < 0.5:  # noqa: S311 - Pseudorandom appropriate for data augmentation
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Convert to tensor
        image_tensor = pil_image_to_norm_tensor(image)  # type: ignore[no-untyped-call]

        # Apply tensor-based augmentations (gaussian noise)
        if use_augmentations and augmentor is not None:
            image_tensor = augmentor.augment_tensor(image_tensor)

        return caption, image_tensor

    # Apply decoding
    dataset = dataset.map(decode_sample)

    # Set epoch length for the dataset
    if epoch_samples is not None:
        samples_per_worker = epoch_samples // world_size
        # Use with_epoch to set the number of samples per epoch
        dataset = dataset.with_epoch(samples_per_worker // batch_size)  # Convert to batches

    # Create batched dataset with tokenizer
    if enable_upsample:
        dataset = dataset.batched(
            batch_size, collation_fn=lambda samples: collate_upsample(samples, tokenizer)
        )
    else:
        dataset = dataset.batched(
            batch_size, collation_fn=lambda samples: collate_base(samples, tokenizer)
        )

    # Create DataLoader
    # Note: WebDataset handles its own batching, so we set batch_size=None
    return DataLoader(
        dataset,
        batch_size=None,  # WebDataset handles batching
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=True,
    )



def collate_base(samples: list[tuple[str, th.Tensor]], tokenizer: Any | None = None) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Collate function for base model training."""
    captions, images_tuple = zip(*samples, strict=False)
    images = list(images_tuple)  # Convert to list for stacking

    # Tokenize captions
    tokens_list = []
    masks_list = []

    for caption in captions:
        if tokenizer is not None:
            if caption == "":
                tokens, mask = get_uncond_tokens_mask(tokenizer)
            else:
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
        else:
            # Fallback to dummy tokens if no tokenizer
            tokens = th.zeros(128, dtype=th.long)
            mask = th.ones(128, dtype=th.bool)

        tokens_list.append(tokens)
        masks_list.append(mask)

    # Stack tokens and masks
    tokens = th.stack(tokens_list)
    masks = th.stack(masks_list)

    # Stack images
    images_tensor = th.stack(images)

    return tokens, masks, images_tensor


def collate_upsample(samples: list[tuple[str, th.Tensor, th.Tensor]], tokenizer: Any | None = None) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Collate function for upsampling model training."""
    captions, low_res_tuple, high_res_tuple = zip(*samples, strict=False)
    low_res_images = list(low_res_tuple)
    high_res_images = list(high_res_tuple)

    # Tokenize captions
    tokens_list = []
    masks_list = []

    for caption in captions:
        if tokenizer is not None:
            if caption == "":
                tokens, mask = get_uncond_tokens_mask(tokenizer)
            else:
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
        else:
            # Fallback to dummy tokens if no tokenizer
            tokens = th.zeros(128, dtype=th.long)
            mask = th.ones(128, dtype=th.bool)

        tokens_list.append(tokens)
        masks_list.append(mask)

    # Stack tokens and masks
    tokens = th.stack(tokens_list)
    masks = th.stack(masks_list)

    # Stack images
    low_res_tensor = th.stack(low_res_images)
    high_res_tensor = th.stack(high_res_images)

    return tokens, masks, low_res_tensor, high_res_tensor


def create_distributed_wds_dataloader(
    urls: str | list[str], batch_size: int, world_size: int, rank: int, **kwargs: Any
) -> DataLoader[Any]:
    """
    Convenience function to create a distributed WebDataset dataloader.

    Args:
        urls: Path to tar files
        batch_size: Batch size per GPU
        world_size: Total number of GPUs
        rank: Current GPU rank
        **kwargs: Additional arguments for distributed_wds_loader

    Returns:
        Distributed DataLoader
    """
    return distributed_wds_loader(
        urls=urls, batch_size=batch_size, world_size=world_size, rank=rank, **kwargs
    )
