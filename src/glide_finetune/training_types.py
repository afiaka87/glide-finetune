"""Type definitions for the training pipeline.

This module provides proper type hints for the training pipeline,
replacing generic Any types with specific type definitions.
"""

from typing import Any, Protocol, Union

import torch as th
from pydantic import BaseModel, Field, validator
from torch import nn
from torch.utils.data import DataLoader


class DataConfig(BaseModel):
    """Data pipeline configuration with validation."""
    data_dir: str = Field(..., description="Directory containing training data")
    use_webdataset: bool = Field(False, description="Use WebDataset format")
    use_optimized_loader: bool = Field(False, description="Use optimized loader with bloom filter")
    wds_dataset_name: str = Field("laion", description="WebDataset name: laion, alamy, or synthetic")
    image_key: str = Field("jpg", description="Key for images in dataset")
    caption_key: str = Field("txt", description="Key for captions in dataset")
    bloom_filter_path: str | None = Field(None, description="Path to bloom filter for optimized loader")
    side_x: int = Field(64, gt=0, description="Image width")
    side_y: int = Field(64, gt=0, description="Image height")
    resize_ratio: float = Field(1.0, gt=0.0, le=2.0, description="Random resize ratio")
    uncond_p: float = Field(0.2, ge=0.0, le=1.0, description="Unconditional probability for CFG")
    use_captions: bool = Field(True, description="Use text captions")
    trim_white_padding: bool = Field(False, description="Remove white padding from images")
    white_thresh: int = Field(245, ge=0, le=255, description="Threshold for white padding detection")
    use_augmentations: bool = Field(True, description="Apply data augmentations")
    batch_size: int = Field(1, gt=0, description="Batch size per GPU")
    num_workers: int = Field(4, ge=0, description="Number of data loader workers")
    epoch_samples: int | None = Field(None, gt=0, description="Number of samples per epoch")

    @validator("wds_dataset_name")
    @classmethod
    def validate_dataset_name(cls, v):
        valid_names = ["laion", "alamy", "synthetic"]
        if v not in valid_names:
            msg = f"wds_dataset_name must be one of {valid_names}"
            raise ValueError(msg)
        return v

    @validator("bloom_filter_path")
    @classmethod
    def validate_bloom_filter(cls, v, values):
        if values.get("use_optimized_loader") and not v:
            msg = "bloom_filter_path required when use_optimized_loader is True"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class ModelConfig(BaseModel):
    """Model configuration with validation."""
    model_path: str | None = Field(None, description="Path to pretrained model")
    resume_ckpt: str = Field("", description="Path to checkpoint to resume from")
    train_upsample: bool = Field(False, description="Train upsampler instead of base model")
    freeze_transformer: bool = Field(False, description="Freeze transformer weights")
    freeze_diffusion: bool = Field(False, description="Freeze diffusion weights")
    randomize_transformer: bool = Field(False, description="Randomize transformer weights")
    randomize_diffusion: bool = Field(False, description="Randomize diffusion weights")
    randomize_init_std: float | None = Field(None, gt=0.0, description="Std dev for random init")
    activation_checkpointing: bool = Field(False, description="Use gradient checkpointing")
    upscale_factor: int = Field(4, gt=0, description="Upsampling factor")
    image_to_upsample: str = Field("low_res_face.png", description="Test image for upsampling")

    @validator("freeze_transformer")
    @classmethod
    def validate_freeze_transformer(cls, v, values):
        if v and values.get("freeze_diffusion"):
            msg = "Cannot freeze both transformer and diffusion"
            raise ValueError(msg)
        if v and values.get("randomize_transformer"):
            msg = "Cannot both freeze and randomize transformer"
            raise ValueError(msg)
        return v

    @validator("freeze_diffusion")
    @classmethod
    def validate_freeze_diffusion(cls, v, values):
        if v and values.get("randomize_diffusion"):
            msg = "Cannot both freeze and randomize diffusion"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    learning_rate: float = Field(1e-5, gt=0.0, description="Learning rate")
    batch_size: int = Field(1, gt=0, description="Batch size (same as DataConfig.batch_size)")
    num_epochs: int = Field(10, gt=0, description="Number of training epochs")
    gradient_accumulation_steps: int = Field(1, gt=0, description="Gradient accumulation steps")
    grad_clip: float = Field(1.0, gt=0.0, description="Gradient clipping value")
    warmup_steps: int = Field(0, ge=0, description="Number of warmup steps")
    warmup_start_lr: float = Field(0.0, ge=0.0, description="Starting learning rate for warmup")
    weight_decay: float = Field(0.0, ge=0.0, description="Weight decay for AdamW")
    use_8bit_adam: bool = Field(False, description="Use 8-bit Adam optimizer")
    adam_beta1: float = Field(0.9, ge=0.0, lt=1.0, description="Adam beta1")
    adam_beta2: float = Field(0.999, ge=0.0, lt=1.0, description="Adam beta2")
    adam_epsilon: float = Field(1e-8, gt=0.0, description="Adam epsilon")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    adam_weight_decay: float = Field(0.01, ge=0.0, description="AdamW weight decay")

    @validator("warmup_start_lr")
    @classmethod
    def validate_warmup_lr(cls, v, values):
        lr = values.get("learning_rate", 1e-5)
        if v > lr:
            msg = f"warmup_start_lr ({v}) cannot be greater than learning_rate ({lr})"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class FP16Config(BaseModel):
    """FP16 training configuration with validation."""
    use_fp16: bool = Field(False, description="Use FP16 mixed precision training")
    fp16_mode: str = Field("auto", description="FP16 mode: auto, conservative, or aggressive")
    fp16_loss_scale: float = Field(256.0, gt=0.0, description="Initial FP16 loss scale")
    fp16_scale_growth_interval: int = Field(100, gt=0, description="Steps between scale growth attempts")
    fp16_scale_growth_factor: float = Field(2.0, gt=1.0, description="Factor to grow loss scale")
    fp16_scale_shrink_factor: float = Field(0.5, gt=0.0, lt=1.0, description="Factor to shrink loss scale")
    fp16_scale_window: int = Field(1000, gt=0, description="Window for tracking scale stability")
    bf16: bool = Field(False, description="Use BF16 instead of FP16")

    @validator("fp16_mode")
    @classmethod
    def validate_fp16_mode(cls, v):
        valid_modes = ["auto", "conservative", "aggressive"]
        if v not in valid_modes:
            msg = f"fp16_mode must be one of {valid_modes}"
            raise ValueError(msg)
        return v

    @validator("bf16")
    @classmethod
    def validate_bf16(cls, v, values):
        if v and values.get("use_fp16"):
            msg = "Cannot use both FP16 and BF16"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class LoggingConfig(BaseModel):
    """Logging configuration with validation."""
    use_wandb: bool = Field(False, description="Use Weights & Biases logging")
    wandb_project: str = Field("glide-finetune", description="W&B project name")
    wandb_run_name: str | None = Field(None, description="W&B run name")
    wandb_tags: list[str] | None = Field(None, description="W&B tags")
    log_frequency: int = Field(100, gt=0, description="Steps between metric logs")
    sample_frequency: int = Field(1000, gt=0, description="Steps between sample generation")
    save_frequency: int = Field(5000, gt=0, description="Steps between checkpoint saves")

    class Config:
        validate_assignment = True


class SamplingConfig(BaseModel):
    """Sampling configuration with validation."""
    test_prompt: str = Field("a painting of a dog", description="Default test prompt")
    test_batch_size: int = Field(1, gt=0, description="Batch size for sampling")
    test_guidance_scale: float = Field(3.0, ge=0.0, description="Classifier-free guidance scale")
    eval_prompt_file: str | None = Field(None, description="File with evaluation prompts")
    sampler: str = Field("p", description="Sampler: p (PLMS), ddim, euler, dpm++")
    num_steps: int | None = Field(None, gt=0, description="Number of sampling steps")
    timestep_respacing: int = Field(100, gt=0, description="Timestep respacing for sampling")
    eta: float = Field(0.0, ge=0.0, le=1.0, description="DDIM eta parameter")
    use_swinir: bool = Field(False, description="Use SwinIR for super-resolution")
    swinir_model_type: str = Field("swinr_real_2x", description="SwinIR model type")

    @validator("sampler")
    @classmethod
    def validate_sampler(cls, v):
        valid_samplers = ["p", "plms", "ddim", "euler", "dpm++"]
        if v not in valid_samplers:
            msg = f"sampler must be one of {valid_samplers}"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class MultiGPUConfig(BaseModel):
    """Multi-GPU/distributed training configuration."""
    use_distributed: bool = Field(False, description="Use distributed training")
    world_size: int = Field(1, gt=0, description="Number of GPUs/processes")
    rank: int = Field(0, ge=0, description="Current process rank")
    local_rank: int = Field(0, ge=0, description="Local GPU rank")
    backend: str = Field("nccl", description="Distributed backend: nccl, gloo, or mpi")
    init_method: str = Field("env://", description="Distributed init method")

    @validator("backend")
    @classmethod
    def validate_backend(cls, v):
        valid_backends = ["nccl", "gloo", "mpi"]
        if v not in valid_backends:
            msg = f"backend must be one of {valid_backends}"
            raise ValueError(msg)
        return v

    @validator("rank")
    @classmethod
    def validate_rank(cls, v, values):
        world_size = values.get("world_size", 1)
        if v >= world_size:
            msg = f"rank ({v}) must be less than world_size ({world_size})"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class CheckpointConfig(BaseModel):
    """Checkpoint configuration with validation."""
    save_directory: str = Field("./CKPT", description="Directory to save checkpoints")
    resume_ckpt: str | None = Field(None, description="Checkpoint path to resume from")
    resume_step: int | None = Field(None, gt=0, description="Step to resume from")
    resume_tar: str | None = Field(None, description="WebDataset tar to resume from")
    skip_optimizer_resume: bool = Field(False, description="Skip loading optimizer state")

    @validator("save_directory")
    @classmethod
    def validate_save_directory(cls, v):
        # Ensure directory path is valid
        if not v:
            msg = "save_directory cannot be empty"
            raise ValueError(msg)
        return v

    class Config:
        validate_assignment = True


class TrainConfig(BaseModel):
    """Complete training configuration with nested validation."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    fp16: FP16Config
    multi_gpu: MultiGPUConfig = Field(default_factory=MultiGPUConfig)
    logging: LoggingConfig
    sampling: SamplingConfig
    checkpoint: CheckpointConfig

    @validator("training")
    @classmethod
    def validate_batch_size_consistency(cls, v, values):
        if "data" in values:
            data_config = values["data"]
            if v.batch_size != data_config.batch_size:
                # Sync batch sizes
                v.batch_size = data_config.batch_size
        return v

    @validator("checkpoint")
    @classmethod
    def validate_checkpoint_consistency(cls, v, values):
        if "model" in values:
            model_config = values["model"]
            # If model has resume_ckpt, sync with checkpoint config
            if model_config.resume_ckpt and not v.resume_ckpt:
                v.resume_ckpt = model_config.resume_ckpt
        return v

    class Config:
        validate_assignment = True


# Type aliases for clarity
TensorBatch = Union[
    tuple[th.Tensor, th.Tensor, th.Tensor],  # tokens, mask, image
    tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],  # tokens, mask, low_res, high_res
]

ModelOutput = dict[str, th.Tensor]
TrainingMetrics = dict[str, float]
CheckpointDict = dict[str, Any]


class DiffusionProcess(Protocol):
    """Protocol for diffusion process interface."""

    def training_losses(
        self,
        model: nn.Module,
        x_start: th.Tensor,
        t: th.Tensor,
        model_kwargs: dict[str, th.Tensor] | None = None,
        noise: th.Tensor | None = None,
    ) -> dict[str, th.Tensor]:
        """Calculate training losses."""
        ...

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        noise: th.Tensor | None = None,
        clip_denoised: bool = True,
        denoised_fn: Any | None = None,
        cond_fn: Any | None = None,
        model_kwargs: dict[str, th.Tensor] | None = None,
        device: th.device | None = None,
        progress: bool = False,
    ) -> th.Tensor:
        """Sample from the model."""
        ...


class CheckpointManager(Protocol):
    """Protocol for checkpoint manager interface."""

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: th.optim.Optimizer,
        epoch: int,
        step: int,
        is_interrupted: bool = False,
    ) -> str:
        """Save a checkpoint."""
        ...

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: th.optim.Optimizer | None = None,
    ) -> tuple[int, int]:
        """Load a checkpoint."""
        ...

    def should_save(self, step: int) -> bool:
        """Check if should save at this step."""
        ...

    def cleanup_interrupted_files(self) -> None:
        """Clean up interrupted checkpoint files."""
        ...


class TrainingStrategy(Protocol):
    """Protocol defining the interface for training strategies."""

    def setup_model(self, config: TrainConfig) -> tuple[nn.Module, DiffusionProcess, dict[str, Any]]:
        """Setup model for training."""
        ...

    def setup_optimizer(self, model: nn.Module, config: TrainConfig) -> th.optim.Optimizer:
        """Setup optimizer for training."""
        ...

    def setup_dataloader(self, config: TrainConfig, model: nn.Module) -> DataLoader[Any]:
        """Setup data loader for training."""
        ...

    def setup_checkpoint_manager(self, config: TrainConfig) -> CheckpointManager:
        """Setup checkpoint manager."""
        ...

    def training_step(
        self,
        model: nn.Module,
        diffusion: DiffusionProcess,
        batch: TensorBatch,
        optimizer: th.optim.Optimizer,
        config: TrainConfig
    ) -> TrainingMetrics:
        """Execute a single training step."""
        ...
