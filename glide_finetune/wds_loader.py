import io
import json
from random import random
import tarfile
import traceback

import PIL
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor


def handle_wds_errors(exc):
    """
    Custom error handler for WebDataset that gracefully handles corrupted data.

    Returns:
        True to skip the sample, False to re-raise the exception
    """
    # Handle tar file corruption errors
    if isinstance(exc, (tarfile.ReadError, EOFError)):
        print(f"Warning: Corrupted data encountered in WebDataset, skipping sample...")
        print(f"  Error type: {type(exc).__name__}")
        return True  # Skip this sample and continue

    # Handle PIL image errors
    if isinstance(exc, (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError)):
        print(f"Warning: Invalid image encountered, skipping sample...")
        return True  # Skip this sample

    # Handle JSON decode errors
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        print(f"Warning: Invalid JSON/text data encountered, skipping sample...")
        return True  # Skip this sample

    # Handle KeyError for missing keys in data
    if isinstance(exc, KeyError):
        print(f"Warning: Missing key in data sample: {exc}, skipping...")
        return True  # Skip this sample

    # For other exceptions, print details but re-raise
    print(f"Unhandled exception in WebDataset: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    return False  # Re-raise the exception


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
    dataset_name="laion",  # can be laion, alamy, or simple.
    upscale_factor=4,
    buffer_size=1000,  # Shuffle buffer size
    initial_prefetch=10,  # Initial prefetch size
    debug=False,  # Enable debug printing
    random_hflip=False,  # Random horizontal flip augmentation
):
    if debug:
        print("\nDEBUG: glide_wds_loader called with:")
        print(f"  - URLs: {len(urls) if isinstance(urls, list) else 'single/pattern'}")
        print(f"  - Dataset name: {dataset_name}")
        print(f"  - Image key: {image_key}")
        print(f"  - Caption key: {caption_key}")
        print(f"  - Enable text: {enable_text}")
        print(f"  - Enable upsample: {enable_upsample}")
        print(f"  - Buffer size: {buffer_size}")
        print("  - Workers will be used for parallel processing")

    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))
    # Create WebDataset with optimizations and custom error handler
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=handle_wds_errors,  # Use custom error handler for robustness
        shardshuffle=False,  # Disabled when using resampled=True to avoid warning
        resampled=True,  # Infinite iteration to avoid exhaustion
    )

    def filter_dataset_laion(item):
        # Quick checks first (before JSON decoding)
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False

        # Decode metadata once
        try:
            metadata = json.loads(item[metadata_key].decode("utf-8"))
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            return False

        # Extract all values at once
        try:
            similarity = metadata["similarity"]
            original_height = metadata["original_height"]
            original_width = metadata["original_width"]
            nsfw_rating = metadata.get("NSFW", "UNLIKELY")
        except KeyError:
            return False

        # Fast numeric checks (no unnecessary float conversion)
        if original_height < min_original_height or original_width < min_original_width:
            return False

        aspect_ratio = original_width / original_height
        if aspect_ratio < ar_lower or aspect_ratio > ar_upper:
            return False

        if (
            similarity < similarity_threshold_lower
            or similarity > similarity_threshold_upper
        ):
            return False

        if nsfw_filter and nsfw_rating in ["NSFW", "LIKELY"]:
            return False

        # Text check last (most expensive)
        if words_to_skip:
            caption = item[caption_key].decode("utf-8").lower()
            if any(word.lower() in caption for word in words_to_skip):
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

    def filter_dataset_simple(item):
        # Simple filter for datasets with just image and text pairs (no metadata)
        if enable_text and caption_key not in item:
            if debug:
                print(
                    f"DEBUG: Item missing caption key '{caption_key}'. Keys: {list(item.keys())}"
                )
            return False
        if enable_image and image_key not in item:
            if debug:
                print(
                    f"DEBUG: Item missing image key '{image_key}'. Keys: {list(item.keys())}"
                )
            return False
        return True
    
    def filter_dataset_synthetic(item):
        # Filter for synthetic DALL-E 3 dataset with JSON metadata
        if enable_image and image_key not in item:
            if debug:
                print(f"DEBUG: Item missing image key '{image_key}'. Keys: {list(item.keys())}")
            return False
        if enable_text and "json" not in item:
            if debug:
                print(f"DEBUG: Item missing json key. Keys: {list(item.keys())}")
            return False
        return True

    if debug:
        print(f"DEBUG: Using dataset filter for '{dataset_name}'")
        print(
            f"DEBUG: Looking for image_key='{image_key}', caption_key='{caption_key}'"
        )

    if dataset_name == "laion":
        filtered_dataset = dataset.select(filter_dataset_laion)
    elif dataset_name == "alamy":
        filtered_dataset = dataset.select(filter_dataset_alamy)
    elif dataset_name == "simple":
        filtered_dataset = dataset.select(filter_dataset_simple)
    elif dataset_name == "synthetic":
        filtered_dataset = dataset.select(filter_dataset_synthetic)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of 'laion', 'alamy', 'simple', or 'synthetic'."
        )

    # Add a counter to track processed items (only if debugging)
    processed_count = [0] if debug else None

    def preprocess_dataset(item):
        if debug and processed_count is not None:
            processed_count[0] += 1
            if processed_count[0] <= 3:  # Log first 3 items for debugging
                print(f"\nDEBUG: Processing item {processed_count[0]}")
                print(f"  Keys in item: {list(item.keys())}")

        tokens, mask, base_tensor, upsample_tensor = None, None, None, None

        # 20%, the empty token is used to represent the unconditional token.
        # This lets classifier-free guidance work after training.
        if not enable_text or random() < uncond_p:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            # Handle synthetic dataset with JSON metadata
            if dataset_name == "synthetic":
                json_data = json.loads(item["json"].decode("utf-8"))
                # Use short_caption for training (or long_caption if preferred)
                caption = json_data.get("short_caption", json_data.get("long_caption", ""))
            else:
                caption = item[caption_key].decode("utf-8")
            tokens, mask = get_tokens_and_mask(tokenizer, caption)

        image_data = item[image_key]
        original_pil_image = PIL.Image.open(io.BytesIO(image_data))

        # Apply random horizontal flip if enabled
        if random_hflip and random() < 0.5:
            original_pil_image = original_pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        base_pil_image = original_pil_image.resize(
            base_image_shape, resample=PIL.Image.BICUBIC
        ).convert("RGB")
        base_tensor = pil_image_to_norm_tensor(base_pil_image)

        # The upsample model needs both the base and the upsample images e.g. 64x64 and 256x256.
        if enable_upsample:
            upsample_pil_image = original_pil_image.resize(
                upsample_image_shape
            ).convert("RGB")
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
            return (
                tokens,  # Already a tensor from get_tokens_and_mask
                mask,  # Already a tensor from get_tokens_and_mask
                base_tensor,
                upsample_tensor,
            )
        return tokens, mask, base_tensor  # Already tensors

    # Apply transformations with optimizations
    transformed_dataset = (
        filtered_dataset.shuffle(buffer_size).map(  # Add shuffling with buffer
            preprocess_dataset, handler=handle_wds_errors  # Use custom error handler
        )
        # Note: batched() creates an extra dimension, skip it for now
        # The DataLoader will handle batching
    )

    if debug:
        print("\nDEBUG: WebDataset loader setup complete")
        print("  Returning dataset pipeline with filters and preprocessing")
        print(f"  Pipeline: WebDataset -> Filter -> Shuffle({buffer_size}) -> Map")

    return transformed_dataset
