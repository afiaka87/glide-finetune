import io
import json
from pathlib import Path
from random import random
import tarfile
import traceback

import PIL
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor


_IMAGE_KEYS = ("jpg", "jpeg", "png", "webp", "bmp", "tiff")


def load_captions_jsonl(jsonl_path: str | Path) -> dict[str, str]:
    """Load a JSONL captions file into a dict mapping sample key to caption.

    Streams the file line-by-line to handle multi-GB files without loading the
    entire file into memory at once.
    """
    captions: dict[str, str] = {}
    jsonl_path = Path(jsonl_path)
    print(f"Loading captions from {jsonl_path} ...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj["key"]
                caption = obj["caption"]
                captions[key] = caption
            except (json.JSONDecodeError, KeyError):
                continue
            if (i + 1) % 1_000_000 == 0:
                print(f"  ... loaded {i + 1:,} captions")
    print(f"Loaded {len(captions):,} captions from {jsonl_path.name}")
    return captions


def _find_image_key(item):
    """Return the first matching image key in the item, or None."""
    for k in _IMAGE_KEYS:
        if k in item:
            return k
    return None


def handle_wds_errors(exc):
    """
    Custom error handler for WebDataset that gracefully handles corrupted data.

    Returns:
        True to skip the sample, False to re-raise the exception
    """
    # Handle tar file corruption errors
    if isinstance(exc, (tarfile.ReadError, EOFError)):
        print("Warning: Corrupted data encountered in WebDataset, skipping sample...")
        print(f"  Error type: {type(exc).__name__}")
        return True  # Skip this sample and continue

    # Handle PIL image errors
    if isinstance(exc, (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError)):
        print("Warning: Invalid image encountered, skipping sample...")
        return True  # Skip this sample

    # Handle JSON decode errors
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        print("Warning: Invalid JSON/text data encountered, skipping sample...")
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
    similarity_threshold_lower=0.0,
    similarity_threshold_upper=1.0,
    words_to_skip=[],
    dataset_name="laion",  # can be laion, alamy, simple, synthetic, datacomp-synthetic, datacomp-real, or datacomp-clip.
    upscale_factor=4,
    buffer_size=1000,  # Shuffle buffer size
    initial_prefetch=10,  # Initial prefetch size
    debug=False,  # Enable debug printing
    random_hflip=False,  # Random horizontal flip augmentation
    captions_jsonl_path=None,  # Path to external JSONL captions (required for datacomp-synthetic and datacomp-clip)
    latent_mode=False,  # Latent diffusion mode: resize to 256x256, return caption strings
    clip_threshold=0.0,  # Minimum CLIP score for datacomp-clip (max of orig/gen must meet this)
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
        if enable_image and _find_image_key(item) is None:
            if debug:
                print(f"DEBUG: Item missing any image key. Keys: {list(item.keys())}")
            return False
        return True

    def filter_dataset_synthetic(item):
        # Filter for synthetic DALL-E 3 dataset with JSON metadata
        if enable_image and image_key not in item:
            if debug:
                print(
                    f"DEBUG: Item missing image key '{image_key}'. Keys: {list(item.keys())}"
                )
            return False
        if enable_text and "json" not in item:
            if debug:
                print(f"DEBUG: Item missing json key. Keys: {list(item.keys())}")
            return False
        return True

    # Load external captions for datacomp dataset
    _captions_map: dict[str, str] | None = None
    if dataset_name == "datacomp-synthetic":
        if captions_jsonl_path is None:
            raise ValueError(
                "captions_jsonl_path is required for dataset_name='datacomp-synthetic'"
            )
        _captions_map = load_captions_jsonl(captions_jsonl_path)

    # datacomp-clip: load JSONL caption index keyed by sample key (needs full entries for CLIP scores)
    _caption_index: dict[str, dict] | None = None
    if dataset_name == "datacomp-clip":
        if captions_jsonl_path is None:
            raise ValueError("captions_jsonl_path is required for dataset_name='datacomp-clip'")
        print(f"Loading caption index from {captions_jsonl_path} ...")
        _caption_index = {}
        with open(captions_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                if not key:
                    continue
                _caption_index[key] = obj
        print(f"  Loaded {len(_caption_index):,} entries")
        if clip_threshold > 0.0:
            eligible = sum(
                1 for e in _caption_index.values()
                if max(
                    e.get("original_caption_clip_score", 0.0),
                    e.get("generated_caption_clip_score", 0.0),
                ) >= clip_threshold
            )
            print(f"  CLIP threshold {clip_threshold}: {eligible:,} / {len(_caption_index):,} entries eligible ({100 * eligible / len(_caption_index):.1f}%)")

    def filter_dataset_datacomp_synthetic(item):
        # Require an image
        if enable_image and _find_image_key(item) is None:
            if debug:
                print(f"DEBUG: Item missing any image key. Keys: {list(item.keys())}")
            return False
        # Require that the sample key has a caption in the external JSONL
        if enable_text:
            sample_key = item.get("__key__")
            if sample_key is None or (
                _captions_map is not None and sample_key not in _captions_map
            ):
                if debug:
                    print(f"DEBUG: No external caption for key '{sample_key}'")
                return False
        return True

    def filter_dataset_datacomp_real(item):
        # Require an image and a txt caption in the tar
        if enable_image and _find_image_key(item) is None:
            if debug:
                print(f"DEBUG: Item missing any image key. Keys: {list(item.keys())}")
            return False
        if enable_text and "txt" not in item:
            if debug:
                print(f"DEBUG: Item missing txt key. Keys: {list(item.keys())}")
            return False
        return True

    def filter_dataset_datacomp_clip(item):
        if enable_image and _find_image_key(item) is None:
            return False
        # Need __key__ to look up caption from JSONL
        sample_key = item.get("__key__", "")
        if not sample_key or (_caption_index is not None and sample_key not in _caption_index):
            return False
        # Skip samples missing either CLIP score
        if _caption_index is not None:
            entry = _caption_index[sample_key]
            if "original_caption_clip_score" not in entry or "generated_caption_clip_score" not in entry:
                return False
            # Apply CLIP score threshold: best-of-two must meet threshold
            if clip_threshold > 0.0:
                best_score = max(
                    entry["original_caption_clip_score"],
                    entry["generated_caption_clip_score"],
                )
                if best_score < clip_threshold:
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
    elif dataset_name == "datacomp-synthetic":
        filtered_dataset = dataset.select(filter_dataset_datacomp_synthetic)
    elif dataset_name == "datacomp-real":
        filtered_dataset = dataset.select(filter_dataset_datacomp_real)
    elif dataset_name == "datacomp-clip":
        filtered_dataset = dataset.select(filter_dataset_datacomp_clip)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of 'laion', 'alamy', 'simple', 'synthetic', 'datacomp-synthetic', 'datacomp-real', or 'datacomp-clip'."
        )

    # Add a counter to track processed items (only if debugging)
    processed_count = [0] if debug else None

    # Caption selection stats for datacomp-clip
    clip_stats = {"generated_wins": 0, "original_wins": 0, "total": 0}

    def _extract_caption(item):
        """Extract caption string from an item based on dataset_name."""
        if dataset_name == "synthetic":
            json_data = json.loads(item["json"].decode("utf-8"))
            return json_data.get("short_caption", json_data.get("long_caption", ""))
        elif dataset_name == "datacomp-synthetic" and _captions_map is not None:
            return _captions_map[item["__key__"]]
        elif dataset_name == "datacomp-clip" and _caption_index is not None:
            # Pick the caption with the higher CLIP score
            entry = _caption_index[item["__key__"]]
            orig_score = entry.get("original_caption_clip_score", 0.0)
            gen_score = entry.get("generated_caption_clip_score", 0.0)
            clip_stats["total"] += 1
            if gen_score >= orig_score:
                clip_stats["generated_wins"] += 1
                caption = entry.get("caption", "")
            else:
                clip_stats["original_wins"] += 1
                caption = entry.get("original_caption", "")
            if clip_stats["total"] % 1000 == 0:
                t = clip_stats["total"]
                gw = clip_stats["generated_wins"]
                ow = clip_stats["original_wins"]
                print(f"[datacomp-clip] {t} samples: generated won {gw} ({100*gw/t:.1f}%), original won {ow} ({100*ow/t:.1f}%)")
            elif clip_stats["total"] <= 5:
                winner = "generated" if gen_score >= orig_score else "original"
                print(f"[datacomp-clip] {item['__key__']}: using {winner} caption (orig={orig_score:.4f}, gen={gen_score:.4f})")
            return caption
        else:
            return item[caption_key].decode("utf-8")

    def preprocess_dataset(item):
        if debug and processed_count is not None:
            processed_count[0] += 1
            if processed_count[0] <= 3:  # Log first 3 items for debugging
                print(f"\nDEBUG: Processing item {processed_count[0]}")
                print(f"  Keys in item: {list(item.keys())}")

        tokens, mask, base_tensor, upsample_tensor = None, None, None, None

        # Extract caption text (needed for both tokenization and latent CLIP)
        is_uncond = not enable_text or random() < uncond_p
        caption_text = ""
        if not is_uncond:
            caption_text = _extract_caption(item)

        # Tokenize for the GLIDE text transformer
        if is_uncond:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            tokens, mask = get_tokens_and_mask(tokenizer, caption_text)

        image_data = item[_find_image_key(item) or image_key]
        original_pil_image = PIL.Image.open(io.BytesIO(image_data))

        # Apply random horizontal flip if enabled
        if random_hflip and random() < 0.5:
            original_pil_image = original_pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Latent mode: resize to 256x256 for VAE encoding, return caption string for CLIP
        if latent_mode:
            latent_pil = original_pil_image.resize(
                (256, 256), resample=PIL.Image.BICUBIC
            ).convert("RGB")
            latent_tensor = pil_image_to_norm_tensor(latent_pil)
            return tokens, mask, latent_tensor, caption_text

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
            preprocess_dataset,
            handler=handle_wds_errors,  # Use custom error handler
        )
        # Note: batched() creates an extra dimension, skip it for now
        # The DataLoader will handle batching
    )

    if debug:
        print("\nDEBUG: WebDataset loader setup complete")
        print("  Returning dataset pipeline with filters and preprocessing")
        print(f"  Pipeline: WebDataset -> Filter -> Shuffle({buffer_size}) -> Map")

    return {
        "dataset": transformed_dataset,
        "clip_caption_stats": clip_stats if dataset_name == "datacomp-clip" else None,
    }


def latent_collate_fn(batch):
    """Custom collate function for latent mode batches.

    Each sample is (tokens_tensor, mask_tensor, image_tensor, caption_string).
    Default collation can't handle the string element, so we stack tensors
    and collect strings into a list.
    """
    import torch as th

    tokens = th.stack([b[0] for b in batch])
    masks = th.stack([b[1] for b in batch])
    images = th.stack([b[2] for b in batch])
    captions = [b[3] for b in batch]
    return tokens, masks, images, captions
