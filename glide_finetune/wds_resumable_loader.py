"""
Resumable WebDataset loader that can efficiently skip to a specific position.

This loader can calculate which tar file and position to start from,
avoiding the need to iterate through all previous tar files.
"""

import io
import json
import glob
from pathlib import Path
from typing import List, Union, Optional, Tuple

import PIL
import torch as th
import webdataset as wds

from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor
from glide_finetune.image_processing import trim_white_padding_pil


def calculate_tar_position(
    tar_files: List[str], 
    global_step: int, 
    batch_size: int, 
    gradient_accumulation_steps: int,
    samples_per_tar: int = 10000
) -> Tuple[List[str], int]:
    """
    Calculate which tar file to start from and how many samples to skip.
    
    Args:
        tar_files: List of tar file paths
        global_step: Current global training step
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        samples_per_tar: Estimated samples per tar file
        
    Returns:
        Tuple of (list of remaining tar files, samples to skip in first tar)
    """
    # Calculate total samples to skip
    total_samples_to_skip = global_step * batch_size * gradient_accumulation_steps
    
    # Calculate which tar file to start from
    tar_idx = total_samples_to_skip // samples_per_tar
    samples_to_skip_in_tar = total_samples_to_skip % samples_per_tar
    
    # If we've gone through all tars, wrap around
    if tar_idx >= len(tar_files):
        tar_idx = tar_idx % len(tar_files)
        
    # Get the tar files starting from the calculated position
    remaining_tars = tar_files[tar_idx:] + tar_files[:tar_idx]
    
    print(f"📊 Resume position calculated:")
    print(f"   Total samples to skip: {total_samples_to_skip:,}")
    print(f"   Starting from tar #{tar_idx + 1}/{len(tar_files)}: {Path(remaining_tars[0]).name}")
    print(f"   Skipping {samples_to_skip_in_tar:,} samples within that tar")
    
    return remaining_tars, samples_to_skip_in_tar


def glide_wds_resumable_loader(
    urls: Union[str, List[str]],
    resume_from_step: int = 0,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    samples_per_tar: int = 10000,
    # Standard WebDataset parameters
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
    dataset_name="laion",
    upscale_factor=4,
    trim_white_padding=False,
    white_thresh=245,
):
    """
    Resumable WebDataset loader that efficiently skips to the correct position.
    
    This avoids the performance penalty of iterating through all tar files
    by calculating which tar to start from and only skipping within that tar.
    """
    
    # Expand glob patterns if needed
    if isinstance(urls, str):
        if '*' in urls or '?' in urls or '[' in urls:
            tar_files = sorted(glob.glob(urls))
        else:
            tar_files = [urls]
    else:
        tar_files = urls
    
    # Calculate starting position if resuming
    skip_in_first_tar = 0
    if resume_from_step > 0:
        tar_files, skip_in_first_tar = calculate_tar_position(
            tar_files,
            resume_from_step,
            batch_size,
            gradient_accumulation_steps,
            samples_per_tar
        )
    
    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))
    
    # Custom handler that warns about duplicates but continues
    def handle_duplicates(exn):
        """Handle duplicate keys by logging and continuing."""
        if "duplicate" in str(exn):
            print(f"Warning: Skipping duplicate file in tar: {exn}")
            return True
        return False
    
    # Create WebDataset with the adjusted tar list
    dataset = wds.WebDataset(
        tar_files,
        cache_dir=cache_path,
        cache_size=10**10 if cache_path else 0,
        handler=handle_duplicates,
        shardshuffle=False,  # Keep order for resumption
    )
    
    # Define filter functions (same as original)
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
        if similarity < similarity_threshold_lower or similarity > similarity_threshold_upper:
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
        return True
    
    def filter_dataset_synthetic(item):
        if enable_image and image_key not in item:
            return False
        return True
    
    # Select filter based on dataset
    if dataset_name == "laion":
        filter_fn = filter_dataset_laion
    elif dataset_name == "alamy":
        filter_fn = filter_dataset_alamy
    elif dataset_name == "synthetic":
        filter_fn = filter_dataset_synthetic
    else:
        filter_fn = lambda x: True
    
    filtered_dataset = dataset.select(filter_fn)
    
    # Preprocessing function
    def preprocess_dataset(item):
        # Get caption
        if dataset_name == "synthetic":
            caption_types = ["caption_blip", "caption_cogvlm", "caption_gpt4", "caption_llava", "txt"]
            caption = None
            for cap_type in caption_types:
                if cap_type in item:
                    caption = item[cap_type].decode("utf-8") if isinstance(item[cap_type], bytes) else item[cap_type]
                    break
            if caption is None:
                caption = ""
        else:
            caption = item[caption_key].decode("utf-8") if enable_text else ""
        
        # Process image
        original_pil_image = PIL.Image.open(io.BytesIO(item[image_key]))
        
        # Apply white padding removal if requested
        if trim_white_padding:
            original_pil_image = trim_white_padding_pil(original_pil_image, white_thresh=white_thresh)
        
        # Get tokens
        if tokenizer is not None:
            tokens, mask = get_tokens_and_mask(tokenizer, caption)
        else:
            tokens = th.zeros(1, 128)
            mask = th.zeros(1, 128).bool()
        
        # Process image
        base_pil_image = original_pil_image.resize(base_image_shape, resample=PIL.Image.BICUBIC)
        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        
        if enable_upsample:
            upsample_pil_image = original_pil_image.resize(upsample_image_shape)
            upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
            return tokens.clone(), mask.clone(), base_tensor, upsample_tensor
        
        return tokens.clone(), mask.clone(), base_tensor
    
    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.reraise_exception
    )
    
    # For WebDataset with PyTorch DataLoader, we need to return the dataset as-is
    # and handle skipping in a different way. Store the skip count as an attribute.
    if skip_in_first_tar > 0:
        print(f"ℹ️  Will skip {skip_in_first_tar} samples in first tar during iteration")
        # We'll handle the skipping in the training loop instead
        transformed_dataset._skip_count = skip_in_first_tar
    
    return transformed_dataset