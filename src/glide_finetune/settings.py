"""Settings and configuration for GLIDE finetune with comprehensive validation."""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingMode(str, Enum):
    """Training mode enumeration."""

    SINGLE_GPU = "single_gpu"
    FP16 = "fp16"
    MULTI_GPU = "multi_gpu"


class FP16Mode(str, Enum):
    """FP16 precision mode enumeration."""

    AUTO = "auto"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class SamplerType(str, Enum):
    """Sampler type enumeration."""

    PLMS = "plms"
    DDIM = "ddim"
    EULER = "euler"
    DPM_PLUS_PLUS = "dpm++"


class DatasetSettings(BaseModel):
    """Dataset configuration settings with validation."""

    data_dir: str = Field(default="", description="Path to training data directory or tar pattern")
    use_webdataset: bool = Field(default=False, description="Use WebDataset loader")
    wds_dataset_name: str = Field(default="laion", description="WebDataset name")
    wds_image_key: str = Field(default="jpg", description="WebDataset image key")
    wds_caption_key: str = Field(default="txt", description="WebDataset caption key")
    wds_cache_dir: Path | None = Field(default=None, description="WebDataset cache directory")
    dataset_prefetch_size: int = Field(default=10, ge=1, le=100, description="Prefetch buffer size")
    num_workers: int = Field(default=4, ge=0, le=32, description="Number of data loading workers")
    epoch_samples: int | None = Field(default=None, gt=0, description="Samples per epoch")
    use_optimized_loader: bool = Field(default=False, description="Use optimized WebDataset loader")
    bloom_filter_path: Path | None = Field(default=None, description="Path to bloom filter")
    side_x: int = Field(
        default=64, ge=8, le=1024, description="Image width (must be divisible by 8)"
    )
    side_y: int = Field(
        default=64, ge=8, le=1024, description="Image height (must be divisible by 8)"
    )
    resize_ratio: float = Field(default=0.75, gt=0.0, le=1.0, description="Random resize ratio")

    @field_validator("side_x", "side_y")
    @classmethod
    def validate_image_dimensions(cls, v: int) -> int:
        """Validate image dimensions are divisible by 8."""
        if v % 8 != 0:
            msg = f"Image dimension {v} must be divisible by 8"
            raise ValueError(msg)
        return v

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: str) -> str:
        """Validate data directory or WebDataset pattern."""
        if not v:
            # Allow empty string for initial creation
            return v
        # Check if it's a WebDataset pattern or a directory
        if "{" in v and "}" in v:  # WebDataset pattern
            return v
        path = Path(v)
        if not path.exists() and not v.startswith("s3://") and not v.startswith("gs://"):
            msg = f"Data directory does not exist: {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_optimized_loader(self) -> DatasetSettings:
        """Validate optimized loader configuration."""
        if self.use_optimized_loader and not self.use_webdataset:
            msg = "Optimized loader requires WebDataset"
            raise ValueError(msg)
        if self.use_optimized_loader and not self.bloom_filter_path:
            # This is a warning, not an error
            warnings.warn("Optimized loader without bloom filter will fall back to standard loader", stacklevel=2)
        return self


class ModelSettings(BaseModel):
    """Model configuration settings with validation."""

    model_path: Path | None = Field(default=None, description="Path to pretrained model")
    resume_ckpt: Path | None = Field(
        default=None, description="Path to checkpoint to resume from"
    )
    train_upsample: bool = Field(default=False, description="Train upsampler instead of base")
    freeze_transformer: bool = Field(default=False, description="Freeze transformer")
    freeze_diffusion: bool = Field(default=False, description="Freeze diffusion UNet")
    randomize_transformer: bool = Field(default=False, description="Randomize transformer weights")
    randomize_diffusion: bool = Field(default=False, description="Randomize diffusion weights")
    randomize_init_std: float | None = Field(
        default=None, gt=0.0, le=1.0, description="Randomization std"
    )
    activation_checkpointing: bool = Field(
        default=False, description="Enable gradient checkpointing"
    )
    upscale_factor: int = Field(default=4, ge=2, le=8, description="Upscaling factor")
    image_to_upsample: str = Field(default="low_res_face.png", description="Test image")
    use_sdpa: bool = Field(default=True, description="Use scaled dot-product attention")

    @field_validator("model_path", "resume_ckpt")
    @classmethod
    def validate_paths(cls, v: Path | None) -> Path | None:
        """Validate model and checkpoint paths."""
        if v is not None and not v.exists():
            msg = f"Path does not exist: {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_freeze_and_randomize(self) -> ModelSettings:
        """Validate freeze and randomization settings."""
        if self.freeze_transformer and self.freeze_diffusion:
            msg = "Cannot freeze both transformer and diffusion (nothing to train)"
            raise ValueError(msg)

        if self.randomize_transformer and self.randomize_diffusion:
            msg = "Cannot randomize both transformer and diffusion"
            raise ValueError(msg)

        if self.randomize_transformer and self.freeze_transformer:
            msg = "Cannot randomize and freeze transformer simultaneously"
            raise ValueError(msg)

        if self.randomize_diffusion and self.freeze_diffusion:
            msg = "Cannot randomize and freeze diffusion simultaneously"
            raise ValueError(msg)

        if (
            self.randomize_transformer or self.randomize_diffusion
        ) and self.randomize_init_std is None:
            # Set default std if randomization is enabled
            self.randomize_init_std = 0.02

        return self


class TrainingSettings(BaseModel):
    """Training configuration settings with validation."""

    learning_rate: float = Field(default=1e-5, gt=0.0, le=1.0, description="Learning rate")
    adam_weight_decay: float = Field(default=0.0, ge=0.0, le=1.0, description="AdamW weight decay")
    adam_eps: float = Field(default=1e-8, gt=0.0, description="Adam epsilon")
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0, description="Adam beta1")
    adam_beta2: float = Field(default=0.999, gt=0.0, lt=1.0, description="Adam beta2")
    grad_clip: float = Field(
        default=1.0, ge=0.0, description="Gradient clipping value (0 to disable)"
    )
    num_epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    warmup_steps: int = Field(default=0, ge=0, description="Number of warmup steps")
    warmup_start_lr: float = Field(default=1e-7, gt=0.0, le=1.0, description="Warmup starting LR")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation")
    batch_size: int = Field(default=1, ge=1, le=1024, description="Batch size per GPU")
    microbatch_size: int = Field(default=1, ge=1, description="Microbatch size")
    uncond_p: float = Field(default=0.2, ge=0.0, le=1.0, description="Unconditional probability")
    use_accelerate: bool = Field(default=False, description="Use HuggingFace Accelerate")

    @model_validator(mode="after")
    def validate_training_config(self) -> TrainingSettings:
        """Validate training configuration."""
        if self.microbatch_size > self.batch_size:
            msg = "Microbatch size cannot exceed batch size"
            raise ValueError(msg)

        if self.warmup_steps > 0 and self.warmup_start_lr >= self.learning_rate:
            msg = "Warmup starting LR must be less than target LR"
            raise ValueError(msg)

        if self.gradient_accumulation_steps < 1:
            msg = "Gradient accumulation steps must be at least 1"
            raise ValueError(msg)

        return self


class FP16Settings(BaseModel):
    """FP16 mixed precision settings with validation."""

    use_fp16: bool = Field(default=False, description="Enable FP16 training")
    fp16_mode: FP16Mode = Field(default=FP16Mode.AUTO, description="FP16 conversion mode")
    fp16_loss_scale: float = Field(default=256.0, gt=0.0, description="Initial loss scale")
    fp16_scale_window: int = Field(default=2000, ge=100, description="Loss scale window")
    fp16_min_loss_scale: float = Field(default=1.0, gt=0.0, description="Minimum loss scale")
    fp16_max_loss_scale: float = Field(default=2**20, gt=1.0, description="Maximum loss scale")
    fp16_scale_growth_factor: float = Field(default=2.0, gt=1.0, description="Scale growth factor")
    fp16_scale_growth_interval: int = Field(default=2000, ge=1, description="Scale growth interval")
    fp16_scale_tolerance: int = Field(
        default=0, ge=0, description="NaN/Inf tolerance before scale reduction"
    )

    @model_validator(mode="after")
    def validate_fp16_config(self) -> FP16Settings:
        """Validate FP16 configuration."""
        if self.use_fp16:
            if self.fp16_loss_scale <= 0:
                msg = "FP16 loss scale must be positive"
                raise ValueError(msg)

            if self.fp16_min_loss_scale >= self.fp16_max_loss_scale:
                msg = "FP16 min loss scale must be less than max loss scale"
                raise ValueError(msg)

            self.fp16_loss_scale = max(self.fp16_loss_scale, self.fp16_min_loss_scale)

            self.fp16_loss_scale = min(self.fp16_loss_scale, self.fp16_max_loss_scale)

        return self


class SamplingSettings(BaseModel):
    """Sampling configuration settings with validation."""

    timestep_respacing: str = Field(default="50", description="Timestep respacing")
    guidance_scale: float = Field(
        default=3.0, ge=0.0, le=50.0, description="Classifier-free guidance scale"
    )
    sampler: SamplerType = Field(default=SamplerType.PLMS, description="Sampling method")
    num_steps: int = Field(default=50, ge=1, le=1000, description="Number of sampling steps")
    use_swinir: bool = Field(default=False, description="Use SwinIR for upscaling")
    test_prompt: str = Field(
        default="a painting of a lady", min_length=1, description="Test prompt"
    )
    sample_bs: int = Field(default=1, ge=1, le=64, description="Sample batch size")
    sample_gs: float = Field(default=4.0, ge=0.0, le=50.0, description="Sample guidance scale")
    eval_prompt_file: Path | None = Field(default=None, description="Evaluation prompt file")

    @field_validator("timestep_respacing")
    @classmethod
    def validate_timestep_respacing(cls, v: str) -> str:
        """Validate timestep respacing format."""
        # Check if it's a number or a valid respacing string
        try:
            int(v)
        except ValueError:
            # Check if it's a valid respacing format like "ddim50"
            if not any(v.startswith(prefix) for prefix in ["ddim", "ddpm"]):
                msg = f"Invalid timestep respacing: {v}"
                raise ValueError(msg)
        return v

    @field_validator("eval_prompt_file")
    @classmethod
    def validate_prompt_file(cls, v: Path | None) -> Path | None:
        """Validate evaluation prompt file."""
        if v is not None and not v.exists():
            msg = f"Evaluation prompt file does not exist: {v}"
            raise ValueError(msg)
        return v


class CheckpointSettings(BaseModel):
    """Checkpoint configuration settings with validation."""

    save_directory: Path = Field(default=Path("checkpoints"), description="Save directory")
    checkpoint_frequency: int = Field(
        default=1000, ge=100, description="Checkpoint frequency in steps"
    )
    sample_frequency: int = Field(
        default=1000, ge=100, description="Sample generation frequency in steps"
    )
    log_frequency: int = Field(default=100, ge=1, description="Logging frequency in steps")
    prefix: str = Field(default="glide-ft-", min_length=1, description="Checkpoint prefix")
    max_checkpoints: int | None = Field(
        default=None, ge=1, description="Maximum checkpoints to keep"
    )
    save_optimizer_state: bool = Field(
        default=True, description="Save optimizer state in checkpoints"
    )

    @model_validator(mode="after")
    def validate_directories(self) -> CheckpointSettings:
        """Create checkpoint directory if it doesn't exist."""
        self.save_directory.mkdir(parents=True, exist_ok=True)
        return self


class SystemSettings(BaseModel):
    """System configuration settings with validation."""

    seed: int = Field(default=0, ge=0, lt=2**32, description="Random seed (0 for performance mode)")
    device: str = Field(default="", description="Device to use (empty for auto-detect)")
    wandb_run_id: str | None = Field(default=None, description="W&B run ID")
    wandb_project: str | None = Field(default=None, description="W&B project name")
    wandb_entity: str | None = Field(default=None, description="W&B entity/username")
    enable_tf32: bool = Field(default=True, description="Enable TF32 operations")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path | None = Field(default=None, description="Log file path")
    compile_model: bool = Field(default=False, description="Use torch.compile")
    compile_backend: str = Field(default="inductor", description="torch.compile backend")
    profile: bool = Field(default=False, description="Enable profiling")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
            raise ValueError(msg)
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string."""
        if v and v not in ["cpu", "cuda"] and not v.startswith("cuda:"):
            msg = f"Invalid device: {v}. Must be 'cpu', 'cuda', or 'cuda:N'"
            raise ValueError(msg)
        return v


class Settings(BaseSettings):
    """Complete application settings with environment variable support."""

    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    fp16: FP16Settings = Field(default_factory=FP16Settings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)

    model_config = SettingsConfigDict(
        env_prefix="GLIDE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def training_mode(self) -> TrainingMode:
        """Determine training mode based on settings."""
        if self.training.use_accelerate:
            return TrainingMode.MULTI_GPU
        if self.fp16.use_fp16:
            return TrainingMode.FP16
        return TrainingMode.SINGLE_GPU

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.training.batch_size * self.training.gradient_accumulation_steps

    @property
    def is_deterministic(self) -> bool:
        """Check if running in deterministic mode."""
        return self.system.seed != 0

    @property
    def is_performance_mode(self) -> bool:
        """Check if running in performance mode."""
        return self.system.seed == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()

    @classmethod
    def from_args(cls, args: Any) -> Settings:
        """Create settings from argparse namespace.

        Args:
            args: Argparse namespace with command-line arguments

        Returns:
            Settings instance populated from arguments
        """
        # Map argparse args to settings structure
        dataset_dict = {}
        model_dict = {}
        training_dict = {}
        fp16_dict = {}
        sampling_dict = {}
        checkpoint_dict = {}
        system_dict = {}

        # Map arguments to appropriate settings groups
        arg_mapping = {
            # Dataset settings
            "data_dir": ("dataset", "data_dir"),
            "use_webdataset": ("dataset", "use_webdataset"),
            "wds_dataset_name": ("dataset", "wds_dataset_name"),
            "wds_image_key": ("dataset", "wds_image_key"),
            "wds_caption_key": ("dataset", "wds_caption_key"),
            "num_workers": ("dataset", "num_workers"),
            "side_x": ("dataset", "side_x"),
            "side_y": ("dataset", "side_y"),
            "resize_ratio": ("dataset", "resize_ratio"),
            # Model settings
            "model_path": ("model", "model_path"),
            "resume_ckpt": ("model", "resume_ckpt"),
            "train_upsample": ("model", "train_upsample"),
            "freeze_transformer": ("model", "freeze_transformer"),
            "freeze_diffusion": ("model", "freeze_diffusion"),
            "randomize_transformer": ("model", "randomize_transformer"),
            "randomize_diffusion": ("model", "randomize_diffusion"),
            "use_sdpa": ("model", "use_sdpa"),
            # Training settings
            "learning_rate": ("training", "learning_rate"),
            "adam_weight_decay": ("training", "adam_weight_decay"),
            "grad_clip": ("training", "grad_clip"),
            "num_epochs": ("training", "num_epochs"),
            "warmup_steps": ("training", "warmup_steps"),
            "batch_size": ("training", "batch_size"),
            "uncond_p": ("training", "uncond_p"),
            "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
            # FP16 settings
            "use_fp16": ("fp16", "use_fp16"),
            "fp16_mode": ("fp16", "fp16_mode"),
            "fp16_loss_scale": ("fp16", "fp16_loss_scale"),
            # Sampling settings
            "timestep_respacing": ("sampling", "timestep_respacing"),
            "guidance_scale": ("sampling", "guidance_scale"),
            "sampler": ("sampling", "sampler"),
            "num_steps": ("sampling", "num_steps"),
            "test_prompt": ("sampling", "test_prompt"),
            "eval_prompt_file": ("sampling", "eval_prompt_file"),
            # Checkpoint settings
            "save_directory": ("checkpoint", "save_directory"),
            "checkpoint_frequency": ("checkpoint", "checkpoint_frequency"),
            "sample_frequency": ("checkpoint", "sample_frequency"),
            "log_frequency": ("checkpoint", "log_frequency"),
            # System settings
            "seed": ("system", "seed"),
            "device": ("system", "device"),
            "debug": ("system", "debug"),
            "log_level": ("system", "log_level"),
        }

        # Group dictionaries
        groups = {
            "dataset": dataset_dict,
            "model": model_dict,
            "training": training_dict,
            "fp16": fp16_dict,
            "sampling": sampling_dict,
            "checkpoint": checkpoint_dict,
            "system": system_dict,
        }

        # Populate dictionaries from args
        for arg_name, (group, setting_name) in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    groups[group][setting_name] = value

        # Create settings
        return cls(
            dataset=DatasetSettings(**dataset_dict) if dataset_dict else DatasetSettings(),
            model=ModelSettings(**model_dict) if model_dict else ModelSettings(),
            training=TrainingSettings(**training_dict) if training_dict else TrainingSettings(),
            fp16=FP16Settings(**fp16_dict) if fp16_dict else FP16Settings(),
            sampling=SamplingSettings(**sampling_dict) if sampling_dict else SamplingSettings(),
            checkpoint=CheckpointSettings(**checkpoint_dict)
            if checkpoint_dict
            else CheckpointSettings(),
            system=SystemSettings(**system_dict) if system_dict else SystemSettings(),
        )

    def validate_for_training(self) -> None:
        """Validate settings are appropriate for training.

        Raises:
            ValueError: If settings are invalid for training
        """
        # Check that we have a data source
        if not self.dataset.data_dir:
            msg = "data_dir is required for training"
            raise ValueError(msg)

        # Check model configuration
        if self.model.freeze_transformer and self.model.freeze_diffusion:
            msg = "Cannot freeze both transformer and diffusion"
            raise ValueError(msg)

        # Check for upsampler-specific settings
        if self.model.train_upsample:
            if self.dataset.side_x != 64 or self.dataset.side_y != 64:
                warnings.warn("Upsampler training typically uses 64x64 base resolution", stacklevel=2)
            if self.training.uncond_p > 0:
                warnings.warn("Upsampler training typically uses uncond_p=0.0", stacklevel=2)

        # Validate FP16 configuration
        if self.fp16.use_fp16 and self.system.device == "cpu":
            msg = "FP16 training not supported on CPU"
            raise ValueError(msg)
