"""Model loading and setup utilities."""

import glob
from pathlib import Path
from typing import Any

import torch as th
from torch import nn
from torch.utils.data import DataLoader

from glide_finetune.loaders.loader import TextImageDataset
from glide_finetune.training_types import ModelConfig, TrainConfig, TrainingConfig
from glide_finetune.utils.freeze_utils import apply_freeze_policy
from glide_finetune.utils.glide_util import load_model
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.randomize_utils import randomize_diffusion, randomize_transformer
from glide_finetune.utils.freeze_utils import build_optimizer_params
from glide_finetune.loaders.wds_loader import glide_wds_loader
from glide_finetune.loaders.wds_loader_distributed import create_distributed_wds_dataloader
from glide_finetune.loaders.wds_loader_optimized import create_optimized_dataloader

logger = get_logger(__name__)


def apply_model_modifications(model: nn.Module, config: ModelConfig) -> None:
    """Apply freeze and randomization policies to the model.

    Args:
        model: The model to modify.
        config: Model configuration with freeze/randomize settings.
    """
    # Apply freeze policy if needed
    if config.freeze_transformer or config.freeze_diffusion:
        freeze_summary = apply_freeze_policy(
            model,
            freeze_transformer=config.freeze_transformer,
            freeze_diffusion=config.freeze_diffusion,
        )
        logger.info(f"\n{freeze_summary}\n")

    # Apply randomization if needed
    if config.randomize_transformer:
        logger.info("\nRandomizing transformer weights...")
        summary = randomize_transformer(model, init_std=config.randomize_init_std)
        logger.info(f"Randomized {summary.selected_params:,} parameters\n")
    elif config.randomize_diffusion:
        logger.info("\nRandomizing diffusion weights...")
        summary = randomize_diffusion(model, init_std=config.randomize_init_std)
        logger.info(f"Randomized {summary.selected_params:,} parameters\n")


def load_glide_model(
    config: ModelConfig, use_fp16: bool = False, device: th.device | None = None,
    accelerator: Any = None
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load GLIDE model with optional checkpoint resumption.

    Args:
        config: Model configuration.
        use_fp16: Whether to use FP16 (handled separately now).
        device: Device to load model on.
        accelerator: Optional Accelerator instance for distributed downloading.

    Returns:
        Tuple of (model, diffusion, options).
    """
    # Determine model path
    model_path = config.model_path or config.resume_ckpt
    if model_path and not Path(model_path).exists():
        logger.warning(f"Warning: Model path {model_path} not found, using base model")
        model_path = None

    # Determine model type
    model_type = "upsample" if config.train_upsample else "base"

    logger.info(f"Loading {model_type} model...")
    if model_path:
        logger.info(f"  From: {model_path}")
    else:
        logger.info("  Using OpenAI base model")

    # Load model
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=model_path or "",
        use_fp16=False,  # FP16 conversion handled separately
        freeze_transformer=config.freeze_transformer,
        freeze_diffusion=config.freeze_diffusion,
        activation_checkpointing=config.activation_checkpointing,
        model_type=model_type,
        accelerator=accelerator,
    )

    # Move to device if specified
    if device is not None:
        glide_model = glide_model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in glide_model.parameters())
    trainable_params = sum(p.numel() for p in glide_model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if config.freeze_transformer or config.freeze_diffusion:
        frozen_params = total_params - trainable_params
        logger.info(f"Frozen parameters: {frozen_params:,}")

    return glide_model, glide_diffusion, glide_options


def create_optimizer(
    model: nn.Module, config: TrainingConfig, use_8bit: bool = False
) -> th.optim.Optimizer:
    """Create optimizer with proper parameter groups.

    Args:
        model: Model to optimize.
        config: Training configuration.
        use_8bit: Whether to use 8-bit optimizer.

    Returns:
        Configured optimizer.
    """
    # Build parameter groups with proper frozen parameter exclusion
    param_groups = build_optimizer_params(model, weight_decay=config.adam_weight_decay)

    if not param_groups:
        msg = "No trainable parameters found!"
        raise ValueError(msg)

    # Create optimizer
    if use_8bit:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning("Warning: bitsandbytes not available, falling back to standard AdamW")
            optimizer = th.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
    else:
        optimizer = th.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )

    return optimizer


def create_dataloader(
    config: TrainConfig, model: nn.Module, distributed: bool = False
) -> DataLoader[Any]:
    """Create unified data loader for all training modes.

    Args:
        config: Complete training configuration.
        model: Model (needed for tokenizer).
        distributed: Whether to create distributed loader.

    Returns:
        Configured DataLoader.
    """
    if config.data.use_webdataset:
        return create_webdataset_loader(config, model, distributed)
    return create_local_dataset_loader(config, model, distributed)


def create_local_dataset_loader(
    config: TrainConfig, model: nn.Module, distributed: bool = False
) -> DataLoader[Any]:
    """Create DataLoader for local TextImageDataset.

    Args:
        config: Complete training configuration.
        model: Model (needed for tokenizer).
        distributed: Whether to create distributed loader.

    Returns:
        DataLoader for local dataset.
    """
    # Create dataset
    dataset = TextImageDataset(
        folder=config.data.data_dir,
        side_x=config.data.side_x,
        side_y=config.data.side_y,
        resize_ratio=config.data.resize_ratio,
        uncond_p=config.data.uncond_p,
        shuffle=True,
        tokenizer=model.tokenizer,
        text_ctx_len=128,  # Default context length
        use_captions=config.data.use_captions,
        enable_glide_upsample=config.model.train_upsample,
        upscale_factor=config.model.upscale_factor,
        trim_white_padding=config.data.trim_white_padding,
        white_thresh=config.data.white_thresh,
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=not distributed,  # Distributed sampler handles shuffling
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=distributed,  # Important for distributed training
    )

    logger.info(f"Local dataset: {len(dataset):,} images")
    return loader


def create_webdataset_loader(
    config: TrainConfig, model: nn.Module, distributed: bool = False
) -> DataLoader[Any]:
    """Create DataLoader for WebDataset (tar files).

    Args:
        config: Complete training configuration.
        model: Model (needed for tokenizer).
        distributed: Whether to create distributed loader.

    Returns:
        DataLoader for WebDataset.
    """
    # Expand glob patterns for WebDataset
    if "*" in config.data.data_dir or "?" in config.data.data_dir or "[" in config.data.data_dir:
        tar_files = sorted(glob.glob(config.data.data_dir))
        if not tar_files:
            msg = f"No files found matching pattern: {config.data.data_dir}"
            raise ValueError(msg)
        logger.info(f"Found {len(tar_files)} tar files matching pattern: {config.data.data_dir}")
        urls = tar_files
    else:
        # Single file or URL
        urls = config.data.data_dir

    # Choose appropriate WebDataset loader
    if config.data.use_optimized_loader and config.data.bloom_filter_path:
        dataloader = create_optimized_webdataset_loader(config, model, urls, distributed)
    elif distributed:
        dataloader = create_distributed_webdataset_loader(config, model, urls)
    else:
        dataloader = create_standard_webdataset_loader(config, model, urls)

    return dataloader


def create_standard_webdataset_loader(
    config: TrainConfig, model: nn.Module, urls: str | list[str]
) -> DataLoader[Any]:
    """Create standard WebDataset loader.

    Args:
        config: Complete training configuration.
        model: Model (needed for tokenizer).
        urls: URL(s) to tar files.

    Returns:
        DataLoader for standard WebDataset.
    """
    logger.info("Using standard WebDataset loader")

    dataset = glide_wds_loader(
        urls=urls,
        caption_key=config.data.caption_key,
        image_key=config.data.image_key,
        enable_image=True,
        enable_text=config.data.use_captions,
        enable_upsample=config.model.train_upsample,
        tokenizer=model.tokenizer,
        ar_lower=0.5,
        ar_upper=2.0,
        min_original_height=config.data.side_x * config.model.upscale_factor,
        min_original_width=config.data.side_y * config.model.upscale_factor,
        base_x=config.data.side_x,
        base_y=config.data.side_y,
        uncond_p=config.data.uncond_p,
        upscale_factor=config.model.upscale_factor,
        nsfw_filter=True,
        similarity_threshold_upper=0.0,
        similarity_threshold_lower=0.5,
        dataset_name=config.data.wds_dataset_name,
        trim_white_padding=config.data.trim_white_padding,
        white_thresh=config.data.white_thresh,
    )

    # WebDataset handles its own batching
    return dataset.batched(config.data.batch_size)


def create_optimized_webdataset_loader(
    config: TrainConfig, model: nn.Module, urls: str | list[str], distributed: bool
) -> DataLoader[Any]:
    """Create optimized WebDataset loader with bloom filter.

    Args:
        config: Complete training configuration.
        model: Model (needed for tokenizer).
        urls: URL(s) to tar files.
        distributed: Whether to create distributed loader.

    Returns:
        DataLoader for optimized WebDataset.
    """
    logger.info(f"Using optimized WebDataset loader with bloom filter: {config.data.bloom_filter_path}")

    return create_optimized_dataloader(
        urls=urls,
        bloom_filter_path=config.data.bloom_filter_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        tokenizer=model.tokenizer,
        base_x=config.data.side_x,
        base_y=config.data.side_y,
        enable_upsample=config.model.train_upsample,
        upscale_factor=config.model.upscale_factor,
        trim_white_padding=config.data.trim_white_padding,
        white_thresh=config.data.white_thresh,
        enable_text=config.data.use_captions,
        uncond_p=config.data.uncond_p,
        caption_key=config.data.caption_key,
        image_key=config.data.image_key,
        dataset_name=config.data.wds_dataset_name,
    )


def create_distributed_webdataset_loader(
    config: TrainConfig, model: nn.Module, urls: str | list[str]
) -> DataLoader[Any]:
    """Create distributed WebDataset loader for multi-GPU training.

    Args:
        config: Complete training configuration.
        model: Model (needed for tokenizer).
        urls: URL(s) to tar files.

    Returns:
        DataLoader for distributed WebDataset.
    """
    logger.info("Using distributed WebDataset loader")

    # Get distributed info from environment variables
    import os
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    return create_distributed_wds_dataloader(
        urls=urls,
        batch_size=config.data.batch_size,
        side_x=config.data.side_x,
        side_y=config.data.side_y,
        resize_ratio=config.data.resize_ratio,
        uncond_p=config.data.uncond_p,
        image_key=config.data.image_key,
        caption_key=config.data.caption_key,
        enable_metadata=True,
        enable_upsample=config.model.train_upsample,
        upscale_factor=config.model.upscale_factor,
        wds_dataset_name=config.data.wds_dataset_name,
        world_size=world_size,
        rank=rank,
        num_workers=config.data.num_workers,
        epoch_samples=config.data.epoch_samples,
        tokenizer=model.tokenizer,
        trim_white_padding=config.data.trim_white_padding,
        white_thresh=config.data.white_thresh,
        use_augmentations=config.data.use_augmentations,
    )
