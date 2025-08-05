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
from pathlib import Path
from typing import Any, Optional

import PIL.Image
import torch as th
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.image_processing import (
    random_center_crop,
    trim_white_padding_pil,
)
from glide_finetune.train_util import pil_image_to_norm_tensor


def webdataset_probe(loader, n=2):
    """Probe a WebDataset loader to debug its output format.
    
    Args:
        loader: The WebDataset loader to probe
        n: Number of samples to print (default: 2)
    """
    print("=== WebDataset Probe ===")
    for i, sample in enumerate(loader):
        if i >= n:
            break
        
        print(f"\n--- Sample {i} ---")
        print(f"Type: {type(sample)}")
        
        if isinstance(sample, dict):
            # Dictionary format (raw WebDataset)
            print(f"Keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, th.Tensor):
                    print(f"  {key}: Tensor {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} len={len(value)}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        elif isinstance(sample, (list, tuple)):
            # Tuple format (after to_tuple)
            print(f"Length: {len(sample)}")
            for j, elem in enumerate(sample):
                if elem is None:
                    print(f"  [{j}]: None")
                elif isinstance(elem, th.Tensor):
                    print(f"  [{j}]: Tensor {elem.shape} dtype={elem.dtype}")
                elif isinstance(elem, str):
                    preview = elem[:60] + "..." if len(elem) > 60 else elem
                    print(f"  [{j}]: str '{preview}'")
                elif isinstance(elem, PIL.Image.Image):
                    print(f"  [{j}]: PIL.Image {elem.size} mode={elem.mode}")
                else:
                    print(f"  [{j}]: {type(elem).__name__}")
        else:
            print(f"Unexpected format: {sample}")
    
    print("\n=== End Probe ===")

# -----------------------------------------------------------------------------#
# ------------------------  CLIP Cache Helper Functions  ----------------------#
# -----------------------------------------------------------------------------#


def sanitize_model_name(model_name: str) -> str:
    """Convert CLIP model name to filesystem-safe string."""
    return model_name.replace("/", "-").replace("@", "-")


def load_clip_embedding_from_cache(
    tar_name: str,
    sample_key: str,
    clip_cache_dir: str,
    clip_model_name: str,
) -> Optional[th.Tensor]:
    """Load CLIP embedding from cache for a specific sample."""
    if not clip_cache_dir:
        return None

    # Construct cache file path
    cache_path = Path(clip_cache_dir)
    model_dir = cache_path / sanitize_model_name(clip_model_name)
    cache_file = model_dir / "embeddings" / f"{tar_name}.pt"

    if not cache_file.exists():
        return None

    try:
        # Load cache file (PyTorch 2.6+ requires weights_only=False for numpy arrays)
        cache_data = th.load(cache_file, map_location="cpu", weights_only=False)

        # Verify cache data is a dictionary
        if not isinstance(cache_data, dict):
            return None

        # Verify metadata matches
        metadata = cache_data.get("metadata", {})
        if metadata.get("clip_model") != clip_model_name:
            return None

        # Get embeddings dictionary
        embeddings = cache_data.get("embeddings", {})
        if not isinstance(embeddings, dict):
            return None
            
        if sample_key not in embeddings:
            return None

        # Extract embedding
        embedding_data = embeddings[sample_key]
        
        # Handle both dict format (new) and direct tensor format (legacy)
        if isinstance(embedding_data, dict):
            embedding = embedding_data.get("embedding")
        elif isinstance(embedding_data, th.Tensor):
            embedding = embedding_data
        else:
            return None

        if embedding is None:
            return None

        return embedding

    except Exception:
        return None


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
    use_clip_cache: bool = False,
    clip_cache_dir: str | None = None,
    clip_model_name: str = "ViT-L/14",
) -> Any:
    words_to_skip = words_to_skip or []

    # Load all CLIP cache metadata if using cache
    if use_clip_cache and clip_cache_dir:
        cache_path_obj = Path(clip_cache_dir)
        
        # Check if cache directory exists
        if not cache_path_obj.exists():
            print(
                f"Warning: CLIP cache directory '{clip_cache_dir}' does not exist. "
                f"Please run the precompute scripts to generate the cache first.\n"
                f"Example: uv run python scripts/precompute_clip_webdataset_embeddings.py "
                f"--tar_urls '/path/to/data/*.tar' --cache_dir '{clip_cache_dir}'"
            )
            use_clip_cache = False
        else:
            model_dir = cache_path_obj / sanitize_model_name(clip_model_name)
            embeddings_dir = model_dir / "embeddings"
            metadata_file = model_dir / "tar_metadata.json"

            # Check if model-specific directory exists
            if not model_dir.exists():
                print(
                    f"Warning: CLIP cache for model '{clip_model_name}' not found at '{model_dir}'. "
                    f"Available models in cache: "
                    f"{[d.name for d in cache_path_obj.iterdir() if d.is_dir()]}\n"
                    f"Please run precompute scripts with --clip_model_name '{clip_model_name}'"
                )
                use_clip_cache = False
            elif not embeddings_dir.exists():
                print(
                    f"Warning: CLIP embeddings directory not found at '{embeddings_dir}'. "
                    f"Cache structure may be incomplete. Please regenerate the cache."
                )
                use_clip_cache = False
            elif metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        tar_metadata = json.load(f)
                        
                        # Check if any tar files are cached
                        if not tar_metadata:
                            print(
                                f"Warning: CLIP cache metadata is empty. No tar files have been processed. "
                                f"Please run the precompute scripts."
                            )
                        else:
                            print(
                                f"CLIP cache loaded: {len(tar_metadata)} tar files cached for {clip_model_name}"
                            )
                except Exception as e:
                    print(f"Warning: Could not load CLIP cache metadata: {e}")
            else:
                print(
                    f"Warning: CLIP cache metadata file not found at '{metadata_file}'. "
                    f"Cache may be incomplete."
                )

    # ------------------------------------------------------------------#
    #  dataset creation + optional filtering
    # ------------------------------------------------------------------#
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.warn_and_continue,  # Skip duplicates instead of crashing
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
        if (
            similarity < similarity_threshold_lower
            or similarity > similarity_threshold_upper
        ):
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
        # Extract tar file name from __url__ field
        tar_url = item.get("__url__", "")
        tar_name = Path(tar_url).name if tar_url else None

        # Extract sample key
        sample_key = item.get("__key__", None)

        # ---------- text conditioning ----------
        if not enable_text or random.random() < uncond_p:
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            # Handle both direct text (txt) and JSON-embedded captions
            if caption_key == "json" or caption_key.endswith(".json"):
                # Extract caption from JSON
                json_data = json.loads(item.get(caption_key, b"{}").decode("utf-8"))
                # Prefer long_caption for more detailed descriptions, fall back to short_caption
                caption = json_data.get("long_caption", "") or \
                         json_data.get("short_caption", "") or \
                         json_data.get("caption", "") or \
                         json_data.get("text", "") or \
                         json_data.get("prompt", "")
            else:
                # Direct text file
                caption = item.get(caption_key, b"").decode("utf-8")
            
            if caption:
                tokens, mask = get_tokens_and_mask(tokenizer, caption)
            else:
                tokens, mask = get_uncond_tokens_mask(tokenizer)

        # ---------- image processing ----------
        # Handle multiple possible image keys (jpg, jpeg, png)
        img_bytes = None
        for key in [image_key, "jpg", "jpeg", "png"]:
            if key in item:
                img_bytes = item[key]
                break
        
        if img_bytes is None:
            raise ValueError(f"No image found in item. Available keys: {list(item.keys())}")
            
        pil = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 1. remove white borders
        pil = trim_white_padding_pil(pil, white_thresh=245)

        # 2. light random-centre crop (only if large enough)
        if pil.width > base_x and pil.height > base_y:
            pil = random_center_crop(pil, min_scale=0.8, jitter_frac=0.1)

        # 3. resize to target sizes
        base_pil = pil.resize(base_size, resample=PIL.Image.BICUBIC)
        base_tensor = pil_image_to_norm_tensor(base_pil)  # (3,64,64) float32

        # ---------- CLIP embedding loading ----------
        clip_embedding = None
        if use_clip_cache and tar_name and sample_key and clip_cache_dir:
            clip_embedding = load_clip_embedding_from_cache(
                tar_name, sample_key, clip_cache_dir, clip_model_name
            )

        if enable_upsample:
            up_pil = pil.resize(up_size, resample=PIL.Image.BICUBIC)
            up_tensor = pil_image_to_norm_tensor(up_pil)
            if use_clip_cache:
                return (
                    tokens.detach().clone() if isinstance(tokens, th.Tensor) else th.tensor(tokens),
                    mask.detach().clone() if isinstance(mask, th.Tensor) else th.tensor(mask, dtype=th.bool),
                    base_tensor,
                    up_tensor,
                    clip_embedding,
                )
            return (
                tokens.detach().clone() if isinstance(tokens, th.Tensor) else th.tensor(tokens),
                mask.detach().clone() if isinstance(mask, th.Tensor) else th.tensor(mask, dtype=th.bool),
                base_tensor,
                up_tensor,
            )

        if use_clip_cache:
            return (
                tokens.detach().clone() if isinstance(tokens, th.Tensor) else th.tensor(tokens),
                mask.detach().clone() if isinstance(mask, th.Tensor) else th.tensor(mask, dtype=th.bool),
                base_tensor,
                clip_embedding,
            )
        return (
            tokens.detach().clone() if isinstance(tokens, th.Tensor) else th.tensor(tokens),
            mask.detach().clone() if isinstance(mask, th.Tensor) else th.tensor(mask, dtype=th.bool),
            base_tensor
        )

    # warn_and_continue: skip corrupt samples, keep training rolling
    dataset = dataset.map(_preprocess, handler=wds.handlers.warn_and_continue)
    return dataset
