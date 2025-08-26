"""
Optimized WebDataset loader using pre-built bloom filter for fast filtering.

This loader uses a pre-computed bloom filter to efficiently filter samples,
avoiding the overhead of parsing and checking metadata for rejected samples.
"""

import io
import json
import pickle
from collections.abc import Iterator
from random import random
from typing import Any, cast

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
logger = get_logger("glide_finetune.wds_loader_optimized")


def glide_wds_loader_optimized(
    urls: str | list[str],
    bloom_filter_path: str,
    tokenizer: Any | None = None,
    # Image processing parameters
    base_x: int = 64,
    base_y: int = 64,
    enable_upsample: bool = False,
    upscale_factor: int = 4,
    trim_white_padding: bool = False,
    white_thresh: int = 245,
    resampling_method: str = "bicubic",  # Resampling method for image resizing
    # Caption parameters
    enable_text: bool = True,
    uncond_p: float = 0.2,
    caption_key: str = "txt",
    image_key: str = "jpg",
    # Dataset format
    dataset_name: str = "laion",  # laion, alamy, synthetic
    # Performance parameters
    cache_path: str | None = None,
    handler: Any = wds.handlers.warn_and_continue,
    # Optional runtime filters (applied after bloom filter)
    runtime_filters: dict[str, Any] | None = None,
) -> Iterator[tuple[th.Tensor, th.Tensor, th.Tensor]]:
    """
    Optimized WebDataset loader using bloom filter for efficient filtering.

    Args:
        urls: Path(s) to WebDataset tar files
        bloom_filter_path: Path to pre-built bloom filter pickle file
        tokenizer: Tokenizer for text encoding
        base_x, base_y: Base image dimensions
        enable_upsample: Whether to prepare upsampled images
        upscale_factor: Upsampling factor
        trim_white_padding: Whether to remove white padding
        white_thresh: Threshold for white padding detection
        enable_text: Whether to process captions
        uncond_p: Probability of using unconditional tokens
        caption_key: Key for caption in dataset
        image_key: Key for image in dataset
        dataset_name: Dataset format (laion, alamy, synthetic)
        cache_path: Optional cache directory for WebDataset
        handler: Error handler for WebDataset
        runtime_filters: Optional additional filters to apply at runtime

    Returns:
        WebDataset iterator with filtered and preprocessed samples
    """

    # Load bloom filter
    logger.info(f"Loading bloom filter from {bloom_filter_path}...")
    with open(bloom_filter_path, "rb") as f:
        bloom_filter = pickle.load(f)
    logger.info(f"Bloom filter loaded (size: ~{bloom_filter.num_bits / 8 / 1024 / 1024:.1f} MB)")

    # Statistics tracking
    stats = {
        "total_seen": 0,
        "bloom_accepted": 0,
        "runtime_rejected": 0,
        "final_accepted": 0,
    }
    
    # Set resampling method
    if resampling_method.lower() == "lanczos":
        resample = Image.Resampling.LANCZOS
    else:  # default to bicubic
        resample = Image.Resampling.BICUBIC

    # Create WebDataset
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10 if cache_path else 0,
        handler=handler,
        shardshuffle=False,  # Disable shard shuffling for consistent ordering
    )

    # Define bloom filter selection
    def bloom_filter_select(sample: dict[str, Any]) -> bool:
        """First-pass filter using bloom filter (very fast)."""
        stats["total_seen"] += 1

        # Check bloom filter
        if sample["__key__"] in bloom_filter:
            stats["bloom_accepted"] += 1
            return True
        return False

    # Apply bloom filter
    dataset = dataset.select(bloom_filter_select)

    # Optional runtime filters (e.g., for additional filtering or debugging)
    if runtime_filters:

        def apply_runtime_filters(sample: dict[str, Any]) -> bool:
            """Apply additional runtime filters if specified."""
            # Example runtime filters:
            # - Additional NSFW checking
            # - Specific caption keywords
            # - Image quality metrics

            if "min_width" in runtime_filters or "min_height" in runtime_filters:
                try:
                    metadata = json.loads(sample.get("json", b"{}"))
                    w = metadata.get("original_width", 0)
                    h = metadata.get("original_height", 0)

                    if w < runtime_filters.get("min_width", 0):
                        stats["runtime_rejected"] += 1
                        return False
                    if h < runtime_filters.get("min_height", 0):
                        stats["runtime_rejected"] += 1
                        return False
                except Exception:
                    pass

            stats["final_accepted"] += 1
            return True

        dataset = dataset.select(apply_runtime_filters)

    # Image dimensions
    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))

    def preprocess_sample(sample: dict[str, Any]) -> tuple[th.Tensor, th.Tensor, th.Tensor] | tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor] | None:
        """Preprocess accepted samples for training."""
        tokens, mask, base_tensor, upsample_tensor = None, None, None, None

        # Handle text/captions
        if not enable_text or random() < uncond_p:  # noqa: S311 - Pseudorandom appropriate for unconditional token dropout
            # Use unconditional tokens
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            # Extract caption based on dataset format
            caption = ""

            if dataset_name == "synthetic":
                # Synthetic dataset: caption in JSON metadata
                try:
                    metadata = json.loads(sample["json"])
                    caption = metadata.get("short_caption", metadata.get("long_caption", ""))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to extract caption from synthetic dataset JSON: {e}")
                    caption = ""

            elif dataset_name == "alamy":
                # Alamy dataset: caption in JSON metadata
                try:
                    metadata = json.loads(sample["json"])
                    caption = metadata.get("caption", "")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to extract caption from alamy dataset JSON: {e}")
                    caption = ""

            else:  # laion or default
                # LAION dataset: caption in separate txt field
                caption = sample.get(caption_key, b"").decode("utf-8", errors="ignore")

            if caption:
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
            else:
                logger.warning(f"Empty caption for sample in {dataset_name} dataset, using unconditional tokens")
                tokens, mask = get_uncond_tokens_mask(tokenizer)

        # Process image
        image_data = sample.get(image_key)
        if image_data is None:
            # This shouldn't happen if bloom filter is correct, but handle gracefully
            return None

        try:
            original_pil_image: Image.Image = PIL.Image.open(io.BytesIO(image_data))

            # Apply white padding removal if requested
            if trim_white_padding:
                original_pil_image = trim_white_padding_pil(
                    original_pil_image.convert("RGB"), white_thresh=white_thresh
                )
            else:
                original_pil_image = original_pil_image.convert("RGB")

            # Resize to base size
            base_pil_image = original_pil_image.resize(base_image_shape, resample=resample)
            base_tensor = pil_image_to_norm_tensor(base_pil_image)  # type: ignore[no-untyped-call]

            # Handle upsampling if needed
            if enable_upsample:
                upsample_pil_image = original_pil_image.resize(
                    upsample_image_shape, resample=resample
                )
                upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)  # type: ignore[no-untyped-call]

                return (
                    tokens.clone(),
                    mask.clone(),
                    base_tensor,
                    upsample_tensor,
                )

            return tokens.clone(), mask.clone(), base_tensor

        except Exception as e:
            logger.info(f"Error processing image: {e}")
            return None

    # Apply preprocessing
    dataset = dataset.map(preprocess_sample)

    # Filter out failed preprocessing
    dataset = dataset.select(lambda x: x is not None)

    # Periodic statistics reporting
    def report_stats(sample: Any) -> Any:
        """Report statistics periodically."""
        if stats["bloom_accepted"] % 10000 == 0 and stats["bloom_accepted"] > 0:
            accept_rate = stats["bloom_accepted"] / max(stats["total_seen"], 1)
            logger.info(
                f"Processed {stats['total_seen']:,} samples, "
                f"accepted {stats['bloom_accepted']:,} ({accept_rate:.1%})"
            )
        return sample

    return cast(Iterator[tuple[th.Tensor, th.Tensor, th.Tensor]], dataset.map(report_stats))



def create_optimized_dataloader(
    urls: str | list[str],
    bloom_filter_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    tokenizer: Any | None = None,
    **loader_kwargs: Any,
) -> Any:
    """
    Create a PyTorch DataLoader with the optimized WebDataset.

    Args:
        urls: Path(s) to WebDataset tar files
        bloom_filter_path: Path to bloom filter
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        tokenizer: Tokenizer for text encoding
        **loader_kwargs: Additional arguments for glide_wds_loader_optimized

    Returns:
        PyTorch DataLoader with optimized filtering
    """

    # Create WebDataset with bloom filter
    dataset = glide_wds_loader_optimized(
        urls=urls, bloom_filter_path=bloom_filter_path, tokenizer=tokenizer, **loader_kwargs
    )

    # Convert to PyTorch DataLoader
    # WebDataset handles batching internally, so we use batch_size=None
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,  # WebDataset handles batching
        num_workers=num_workers,
        pin_memory=True,
    )

    # Add batching
    return dataloader.batched(batch_size)



# Convenience function for testing
def test_optimized_loader(
    tar_file: str,
    bloom_filter_path: str,
    num_samples: int = 10,
) -> None:
    """Test the optimized loader by loading a few samples."""

    logger.info(f"Testing optimized loader with {tar_file}")
    logger.info(f"Using bloom filter: {bloom_filter_path}")

    # Create simple tokenizer for testing (replace with actual)
    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [0] * 128  # Dummy tokens

    dataset = glide_wds_loader_optimized(
        urls=tar_file,
        bloom_filter_path=bloom_filter_path,
        tokenizer=DummyTokenizer(),
        enable_text=True,
        enable_upsample=False,
    )

    # Load and display a few samples
    count = 0
    for sample in dataset:
        tokens, mask, image_tensor = sample[:3]
        logger.info(f"Sample {count + 1}:")
        logger.info(f"  Tokens shape: {tokens.shape}")
        logger.info(f"  Mask shape: {mask.shape}")
        logger.info(f"  Image shape: {image_tensor.shape}")
        logger.info(f"  Image range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")

        count += 1
        if count >= num_samples:
            break

    logger.info(f"\nSuccessfully loaded {count} samples")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        logger.info("Usage: python wds_loader_optimized.py <tar_file> <bloom_filter.pkl> [num_samples]")
        sys.exit(1)

    tar_file = sys.argv[1]
    bloom_path = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    test_optimized_loader(tar_file, bloom_path, num_samples)
