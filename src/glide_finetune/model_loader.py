"""
Unified model loading module for GLIDE finetuning.

This module consolidates all model loading logic with proper type safety,
checkpoint resumption support, and various model configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, Union, cast

import torch as th
from torch import nn

from glide_finetune.utils.freeze_utils import FreezeSummary, apply_freeze_policy
from glide_finetune.utils.layer_utils import LayerSelectionSummary
from glide_finetune.utils.logging_utils import get_logger
from glide_finetune.utils.randomize_utils import (
    randomize_diffusion,
    randomize_transformer,
)
from glide_text2im.download import load_checkpoint as load_openai_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

logger = get_logger(__name__)

# Type definitions
ModelType = Literal["base", "upsample", "base-inpaint", "upsample-inpaint"]
DeviceType = Union[str, th.device]


class ModelOptions(TypedDict):
    """Type definition for model options."""

    image_size: int
    num_channels: int
    num_res_blocks: int
    attention_resolutions: str
    dropout: float
    learn_sigma: bool
    sigma_small: bool
    num_classes: int
    classifier_use_scale_shift_norm: bool
    diffusion_steps: int
    noise_schedule: str
    timestep_respacing: str
    use_kl: bool
    predict_xstart: bool
    rescale_timesteps: bool
    rescale_learned_sigmas: bool
    use_fp16: bool
    text_ctx: int
    xf_width: int
    xf_layers: int
    xf_heads: int
    xf_final_ln: bool
    xf_padding: bool
    cache_text_emb: bool
    model_channels: int | None
    inpaint: bool | None
    super_res: bool | None


@dataclass
class ModelLoadConfig:
    """Configuration for model loading."""

    model_type: ModelType = "base"
    checkpoint_path: str | None = None
    use_fp16: bool = False
    freeze_transformer: bool = False
    freeze_diffusion: bool = False
    randomize_transformer: bool = False
    randomize_diffusion: bool = False
    randomize_init_std: float = 0.02
    activation_checkpointing: bool = False
    device: DeviceType | None = None
    use_sdpa: bool = True  # Scaled dot-product attention optimization
    use_openai_checkpoint: bool = True  # Use OpenAI base model if no checkpoint provided


class ModelInfo:
    """Container for model information."""

    def __init__(
        self,
        model: nn.Module,
        diffusion: Any,
        options: ModelOptions,
        tokenizer: Any,  # Encoder from glide_text2im
        model_type: ModelType,
        freeze_summary: FreezeSummary | None = None,
        randomization_summary: LayerSelectionSummary | None = None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.options = options
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.freeze_summary = freeze_summary
        self.randomization_summary = randomization_summary

        # Calculate parameter counts
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.frozen_params = self.total_params - self.trainable_params


class UnifiedModelLoader:
    """Unified model loader with comprehensive support for all model types and configurations."""

    @staticmethod
    def load_model(
        config: ModelLoadConfig,
        verbose: bool = True,
    ) -> ModelInfo:
        """
        Load a GLIDE model with the specified configuration.
        
        Args:
            config: Model loading configuration
            verbose: Whether to print loading information
            
        Returns:
            ModelInfo container with model, diffusion, options, and metadata
        """
        # Get model options based on type
        options = UnifiedModelLoader._get_model_options(config.model_type, config.use_fp16)

        # Create model and diffusion
        if verbose:
            logger.info(f"Creating {config.model_type} model...")
        model, diffusion = create_model_and_diffusion(**options)

        # Enable activation checkpointing if requested
        if config.activation_checkpointing:
            model.use_checkpoint = True
            if verbose:
                logger.info("Activation checkpointing enabled")

        # Load checkpoint
        checkpoint_loaded = UnifiedModelLoader._load_checkpoint(
            model, config.checkpoint_path, config.model_type, config.use_openai_checkpoint, verbose
        )

        # Convert to FP16 if requested
        if config.use_fp16:
            model.convert_to_fp16()
            if verbose:
                logger.info("Model converted to FP16")

        # Apply SDPA optimization if requested and not using old checkpoint
        if config.use_sdpa and checkpoint_loaded != "old_checkpoint":
            UnifiedModelLoader._apply_sdpa_optimization(model, verbose)

        # Move to device if specified
        if config.device is not None:
            model = model.to(config.device)
            if verbose:
                logger.info(f"Model moved to {config.device}")

        # Apply freezing if requested
        freeze_summary = None
        if config.freeze_transformer or config.freeze_diffusion:
            freeze_summary = apply_freeze_policy(
                model,
                freeze_transformer=config.freeze_transformer,
                freeze_diffusion=config.freeze_diffusion,
            )
            if verbose:
                logger.info(f"\n{freeze_summary}\n")

        # Apply randomization if requested
        randomization_summary = None
        if config.randomize_transformer:
            if verbose:
                logger.info("Randomizing transformer weights...")
            randomization_summary = randomize_transformer(model, init_std=config.randomize_init_std)
            if verbose:
                logger.info(f"Randomized {randomization_summary.selected_params:,} parameters")
        elif config.randomize_diffusion:
            if verbose:
                logger.info("Randomizing diffusion weights...")
            randomization_summary = randomize_diffusion(model, init_std=config.randomize_init_std)
            if verbose:
                logger.info(f"Randomized {randomization_summary.selected_params:,} parameters")

        # Get tokenizer
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = Encoder()

        # Create ModelInfo
        model_info = ModelInfo(
            model=model,
            diffusion=diffusion,
            options=cast("ModelOptions", options),
            tokenizer=tokenizer,
            model_type=config.model_type,
            freeze_summary=freeze_summary,
            randomization_summary=randomization_summary,
        )

        # Print summary if verbose
        if verbose:
            UnifiedModelLoader._print_model_summary(model_info)

        return model_info

    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: str,
        model_type: ModelType | None = None,
        device: DeviceType | None = None,
        verbose: bool = True,
    ) -> ModelInfo:
        """
        Load a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_type: Override model type (if None, detect from checkpoint)
            device: Device to load model on
            verbose: Whether to print loading information
            
        Returns:
            ModelInfo container
        """
        if not Path(checkpoint_path).exists():
            msg = f"Checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(msg)

        # Load checkpoint to detect model type if not specified
        if model_type is None:
            checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_type" in checkpoint:
                model_type = checkpoint["model_type"]
            else:
                # Try to infer from model structure
                model_type = UnifiedModelLoader._infer_model_type(checkpoint)
                if verbose:
                    logger.info(f"Inferred model type: {model_type}")

        # Create config and load
        config = ModelLoadConfig(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device=device,
            use_openai_checkpoint=False,
        )

        return UnifiedModelLoader.load_model(config, verbose=verbose)

    @staticmethod
    def load_for_training(
        model_type: ModelType = "base",
        checkpoint_path: str | None = None,
        resume_checkpoint: str | None = None,
        freeze_transformer: bool = False,
        freeze_diffusion: bool = False,
        randomize_transformer: bool = False,
        randomize_diffusion: bool = False,
        use_fp16: bool = False,
        activation_checkpointing: bool = False,
        device: DeviceType | None = None,
        verbose: bool = True,
    ) -> tuple[ModelInfo, dict[str, Any] | None]:
        """
        Load a model for training with optional checkpoint resumption.
        
        Args:
            model_type: Type of model to load
            checkpoint_path: Path to base model checkpoint
            resume_checkpoint: Path to training checkpoint to resume from
            freeze_transformer: Whether to freeze transformer
            freeze_diffusion: Whether to freeze diffusion
            randomize_transformer: Whether to randomize transformer
            randomize_diffusion: Whether to randomize diffusion
            use_fp16: Whether to use FP16
            activation_checkpointing: Whether to use activation checkpointing
            device: Device to load on
            verbose: Whether to print information
            
        Returns:
            Tuple of (ModelInfo, training_state_dict or None)
        """
        # Handle checkpoint resumption
        training_state = None
        if resume_checkpoint and Path(resume_checkpoint).exists():
            if verbose:
                logger.info(f"Loading training checkpoint: {resume_checkpoint}")

            checkpoint = th.load(resume_checkpoint, map_location="cpu", weights_only=False)

            # Extract model state and training state
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    # It's a training checkpoint
                    checkpoint_path = resume_checkpoint  # Use the full checkpoint
                    training_state = {
                        k: v for k, v in checkpoint.items()
                        if k != "model_state_dict"
                    }
                    if verbose:
                        step = checkpoint.get("step", 0)
                        logger.info(f"Resuming from step {step}")
                else:
                    # It's just a model checkpoint
                    checkpoint_path = resume_checkpoint

        # Create config
        config = ModelLoadConfig(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            randomize_transformer=randomize_transformer,
            randomize_diffusion=randomize_diffusion,
            randomize_init_std=0.02,
            use_fp16=use_fp16,
            activation_checkpointing=activation_checkpointing,
            device=device,
        )

        # Load model
        model_info = UnifiedModelLoader.load_model(config, verbose=verbose)

        return model_info, training_state

    @staticmethod
    def load_for_inference(
        model_type: ModelType = "base",
        checkpoint_path: str | None = None,
        device: DeviceType = "cuda",
        use_fp16: bool = True,
        verbose: bool = True,
    ) -> ModelInfo:
        """
        Load a model optimized for inference.
        
        Args:
            model_type: Type of model to load
            checkpoint_path: Path to checkpoint
            device: Device to load on
            use_fp16: Whether to use FP16 for faster inference
            verbose: Whether to print information
            
        Returns:
            ModelInfo container
        """
        config = ModelLoadConfig(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            use_fp16=use_fp16,
            device=device,
            freeze_transformer=True,  # Everything frozen for inference
            freeze_diffusion=True,
            activation_checkpointing=False,  # No need for memory saving in inference
        )

        model_info = UnifiedModelLoader.load_model(config, verbose=verbose)

        # Set to eval mode
        model_info.model.eval()

        return model_info

    @staticmethod
    def _get_model_options(model_type: ModelType, use_fp16: bool) -> dict[str, Any]:
        """Get model options for the specified model type."""
        options: dict[str, Any]
        if model_type in ["base", "base-inpaint"]:
            options = model_and_diffusion_defaults()
        elif model_type in ["upsample", "upsample-inpaint"]:
            options = model_and_diffusion_defaults_upsampler()
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        if "inpaint" in model_type:
            options["inpaint"] = True

        options["use_fp16"] = use_fp16

        return options

    @staticmethod
    def _load_checkpoint(
        model: nn.Module,
        checkpoint_path: str | None,
        model_type: ModelType,
        use_openai: bool,
        verbose: bool,
    ) -> str:
        """
        Load checkpoint into model.
        
        Returns:
            "user_checkpoint", "openai_checkpoint", "old_checkpoint", or "none"
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            if verbose:
                logger.info(f"Loading checkpoint: {checkpoint_path}")

            checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    # Training checkpoint format
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    # Alternative format
                    state_dict = checkpoint["state_dict"]
                else:
                    # Assume it's a raw state dict
                    state_dict = checkpoint
            else:
                # Raw state dict
                state_dict = checkpoint

            # Check for architecture compatibility
            is_old_checkpoint = UnifiedModelLoader._check_old_checkpoint(state_dict)
            if is_old_checkpoint and verbose:
                logger.warning(
                    "Detected old checkpoint format (pre-SDPA). "
                    "Consider converting with convert_checkpoint.py for better performance."
                )

            model.load_state_dict(state_dict)
            return "old_checkpoint" if is_old_checkpoint else "user_checkpoint"

        if use_openai:
            if verbose:
                logger.info(f"Loading OpenAI {model_type} checkpoint")
            model.load_state_dict(load_openai_checkpoint(model_type, "cpu"))
            return "openai_checkpoint"

        if verbose:
            logger.info("No checkpoint loaded, using random initialization")
        return "none"

    @staticmethod
    def _check_old_checkpoint(state_dict: dict[str, Any]) -> bool:
        """Check if checkpoint is from old architecture (pre-SDPA)."""
        # Check for telltale signs of old architecture
        # Look for attention-related keys that might indicate old format
        return any("attention" in key.lower() and "qkv" in key.lower() for key in state_dict)

    @staticmethod
    def _apply_sdpa_optimization(model: nn.Module, verbose: bool) -> None:
        """Apply scaled dot-product attention optimization."""
        # SDPA optimization would go here if available
        # For now, just log that we're using standard attention
        if verbose:
            logger.info("Using standard attention implementation")

    @staticmethod
    def _infer_model_type(checkpoint: dict[str, Any]) -> ModelType:
        """Infer model type from checkpoint structure."""
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Check for upsampling-specific keys
        for key in state_dict:
            if "upsample" in key.lower() or "super_res" in key:
                return "upsample"
            if "inpaint" in key.lower():
                if "upsample" in key.lower():
                    return "upsample-inpaint"
                return "base-inpaint"

        return "base"

    @staticmethod
    def _print_model_summary(model_info: ModelInfo) -> None:
        """Print model summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Model Type: {model_info.model_type}")
        logger.info(f"Total Parameters: {model_info.total_params:,}")
        logger.info(f"Trainable Parameters: {model_info.trainable_params:,}")
        logger.info(f"Frozen Parameters: {model_info.frozen_params:,}")

        if model_info.freeze_summary:
            logger.info(f"Trainable Params: {model_info.freeze_summary.trainable_params:,}/{model_info.total_params:,}")

        if model_info.randomization_summary:
            logger.info(f"Randomized Parameters: {model_info.randomization_summary.selected_params:,}")

        logger.info(f"{'='*60}\n")


# Convenience functions for backward compatibility
def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """
    Backward-compatible load_model function.
    
    This maintains the old API while using the new unified loader.
    """
    config = ModelLoadConfig(
        model_type=cast("ModelType", model_type),
        checkpoint_path=glide_path if glide_path else None,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
    )

    model_info = UnifiedModelLoader.load_model(config, verbose=False)

    return model_info.model, model_info.diffusion, dict(model_info.options)


def load_checkpoint_for_training(
    checkpoint_path: str,
    device: DeviceType | None = None,
) -> tuple[nn.Module, Any, dict[str, Any], dict[str, Any]]:
    """
    Load a checkpoint for resuming training.
    
    Returns:
        Tuple of (model, diffusion, options, training_state)
    """
    model_info, training_state = UnifiedModelLoader.load_for_training(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return (
        model_info.model,
        model_info.diffusion,
        dict(model_info.options),
        training_state or {},
    )


def create_diffusion_for_sampling(
    options: dict[str, Any],
    num_steps: int = 100,
) -> Any:
    """
    Create a diffusion instance configured for sampling.
    
    Args:
        options: Model options dictionary
        num_steps: Number of sampling steps
        
    Returns:
        Configured diffusion instance
    """
    return create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=str(num_steps),
    )


# Export main components
__all__ = [
    "ModelInfo",
    "ModelLoadConfig",
    "ModelType",
    "UnifiedModelLoader",
    "create_diffusion_for_sampling",
    "load_checkpoint_for_training",
    "load_model",
]
