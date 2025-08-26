"""ML-specific configuration settings for GLIDE training.

Separates ML hyperparameters, optimizer configs, and scheduler configs
from general training settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAMAX = "adamax"
    LION = "lion"  # Lion optimizer (if available)


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    STEP = "step"
    MULTISTEP = "multistep"
    PLATEAU = "plateau"
    ONE_CYCLE = "one_cycle"
    CYCLIC = "cyclic"


class GradientClippingType(str, Enum):
    """Gradient clipping strategies."""

    NONE = "none"
    VALUE = "value"
    NORM = "norm"
    ADAPTIVE = "adaptive"


class OptimizerConfig(BaseModel):
    """Optimizer configuration with comprehensive hyperparameters."""

    optimizer_type: OptimizerType = Field(
        default=OptimizerType.ADAMW,
        description="Type of optimizer to use"
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        le=10.0,
        description="Base learning rate"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay (L2 regularization)"
    )

    # Adam/AdamW specific
    adam_betas: tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Adam beta parameters"
    )
    adam_eps: float = Field(
        default=1e-8,
        gt=0.0,
        description="Adam epsilon for numerical stability"
    )
    amsgrad: bool = Field(
        default=False,
        description="Use AMSGrad variant of Adam"
    )

    # SGD specific
    momentum: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description="SGD momentum"
    )
    dampening: float = Field(
        default=0.0,
        ge=0.0,
        description="SGD dampening for momentum"
    )
    nesterov: bool = Field(
        default=False,
        description="Use Nesterov momentum"
    )

    # RMSprop specific
    rmsprop_alpha: float = Field(
        default=0.99,
        gt=0.0,
        lt=1.0,
        description="RMSprop smoothing constant"
    )
    rmsprop_centered: bool = Field(
        default=False,
        description="Use centered RMSprop"
    )

    # Lion optimizer specific (if using)
    lion_betas: tuple[float, float] = Field(
        default=(0.9, 0.99),
        description="Lion beta parameters"
    )

    # Gradient accumulation
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of gradient accumulation steps"
    )

    # Per-parameter options
    bias_correction: bool = Field(
        default=True,
        description="Apply bias correction in Adam"
    )
    decoupled_weight_decay: bool = Field(
        default=True,
        description="Use decoupled weight decay (AdamW style)"
    )

    @field_validator("adam_betas", "lion_betas")
    @classmethod
    def validate_betas(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate beta parameters."""
        beta1, beta2 = v
        if not (0.0 <= beta1 < 1.0):
            msg = f"Beta1 must be in [0, 1), got {beta1}"
            raise ValueError(msg)
        if not (0.0 <= beta2 < 1.0):
            msg = f"Beta2 must be in [0, 1), got {beta2}"
            raise ValueError(msg)
        return v

    def to_optimizer_kwargs(self) -> dict[str, Any]:
        """Convert to optimizer kwargs."""
        base_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay if self.decoupled_weight_decay else 0.0,
        }

        if self.optimizer_type in [OptimizerType.ADAM, OptimizerType.ADAMW]:
            base_kwargs.update({
                "betas": self.adam_betas,
                "eps": self.adam_eps,
                "amsgrad": self.amsgrad,
            })
        elif self.optimizer_type == OptimizerType.SGD:
            base_kwargs.update({
                "momentum": self.momentum,
                "dampening": self.dampening,
                "nesterov": self.nesterov,
            })
        elif self.optimizer_type == OptimizerType.RMSPROP:
            base_kwargs.update({
                "alpha": self.rmsprop_alpha,
                "eps": self.adam_eps,
                "centered": self.rmsprop_centered,
                "momentum": self.momentum,
            })

        return base_kwargs


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    scheduler_type: SchedulerType = Field(
        default=SchedulerType.CONSTANT,
        description="Type of LR scheduler"
    )

    # Warmup settings
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of warmup steps"
    )
    warmup_start_factor: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Starting factor for warmup (multiplied by base LR)"
    )
    warmup_method: Literal["linear", "exponential"] = Field(
        default="linear",
        description="Warmup method"
    )

    # Cosine annealing
    cosine_t_max: int | None = Field(
        default=None,
        gt=0,
        description="Maximum iterations for cosine annealing"
    )
    cosine_eta_min: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum learning rate for cosine annealing"
    )
    cosine_restarts: list[int] = Field(
        default_factory=list,
        description="Restart points for cosine annealing with restarts"
    )

    # Step/MultiStep
    step_size: int = Field(
        default=30,
        gt=0,
        description="Step size for StepLR"
    )
    step_gamma: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Multiplicative factor for StepLR"
    )
    multistep_milestones: list[int] = Field(
        default_factory=list,
        description="Milestones for MultiStepLR"
    )

    # Exponential
    exponential_gamma: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Multiplicative factor for ExponentialLR"
    )

    # Polynomial
    polynomial_power: float = Field(
        default=1.0,
        gt=0.0,
        description="Power for polynomial decay"
    )
    polynomial_total_iters: int | None = Field(
        default=None,
        gt=0,
        description="Total iterations for polynomial decay"
    )

    # OneCycle
    one_cycle_max_lr: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum learning rate for OneCycleLR"
    )
    one_cycle_pct_start: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description="Percentage of cycle spent increasing LR"
    )

    # Plateau
    plateau_mode: Literal["min", "max"] = Field(
        default="min",
        description="Mode for ReduceLROnPlateau"
    )
    plateau_factor: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Factor for ReduceLROnPlateau"
    )
    plateau_patience: int = Field(
        default=10,
        gt=0,
        description="Patience for ReduceLROnPlateau"
    )
    plateau_threshold: float = Field(
        default=1e-4,
        gt=0.0,
        description="Threshold for ReduceLROnPlateau"
    )

    @model_validator(mode="after")
    def validate_scheduler_config(self) -> SchedulerConfig:
        """Validate scheduler configuration."""
        if self.scheduler_type == SchedulerType.COSINE and self.cosine_t_max is None:
            msg = "cosine_t_max required for cosine scheduler"
            raise ValueError(msg)

        if self.scheduler_type == SchedulerType.MULTISTEP and not self.multistep_milestones:
            msg = "multistep_milestones required for multistep scheduler"
            raise ValueError(msg)

        if self.scheduler_type == SchedulerType.ONE_CYCLE and self.one_cycle_max_lr is None:
            msg = "one_cycle_max_lr required for one_cycle scheduler"
            raise ValueError(msg)

        return self


class GradientConfig(BaseModel):
    """Gradient processing configuration."""

    clip_type: GradientClippingType = Field(
        default=GradientClippingType.NORM,
        description="Type of gradient clipping"
    )
    clip_value: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping threshold"
    )
    clip_norm_type: float = Field(
        default=2.0,
        gt=0.0,
        description="Norm type for gradient clipping (e.g., 2 for L2)"
    )

    # Adaptive clipping
    adaptive_clip_percentile: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Percentile for adaptive gradient clipping"
    )
    adaptive_clip_history_size: int = Field(
        default=100,
        gt=0,
        description="History size for adaptive clipping"
    )

    # Gradient noise
    add_gradient_noise: bool = Field(
        default=False,
        description="Add Gaussian noise to gradients"
    )
    gradient_noise_gamma: float = Field(
        default=0.55,
        gt=0.0,
        le=1.0,
        description="Gradient noise gamma (decay factor)"
    )
    gradient_noise_eta: float = Field(
        default=0.01,
        gt=0.0,
        description="Gradient noise eta (base noise level)"
    )

    # Gradient accumulation
    accumulate_gradients: bool = Field(
        default=True,
        description="Enable gradient accumulation"
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing for memory efficiency"
    )

    # Gradient monitoring
    log_gradient_norm: bool = Field(
        default=True,
        description="Log gradient norms"
    )
    detect_anomaly: bool = Field(
        default=False,
        description="Enable anomaly detection (slower)"
    )


class RegularizationConfig(BaseModel):
    """Regularization configuration."""

    # Dropout
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout rate"
    )

    # Label smoothing
    label_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Label smoothing factor"
    )

    # Weight constraints
    max_weight_norm: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum weight norm constraint"
    )

    # Spectral normalization
    use_spectral_norm: bool = Field(
        default=False,
        description="Apply spectral normalization"
    )
    spectral_norm_power_iterations: int = Field(
        default=1,
        ge=1,
        description="Power iterations for spectral norm"
    )

    # EMA
    use_ema: bool = Field(
        default=False,
        description="Use exponential moving average of weights"
    )
    ema_decay: float = Field(
        default=0.9999,
        gt=0.0,
        lt=1.0,
        description="EMA decay rate"
    )
    ema_update_every: int = Field(
        default=1,
        ge=1,
        description="EMA update frequency"
    )


@dataclass
class MLHyperparameters:
    """Complete ML hyperparameter configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def create_optimizer(
        self,
        model: nn.Module,
        filter_fn: callable | None = None,
    ) -> torch.optim.Optimizer:
        """Create optimizer from configuration.
        
        Args:
            model: Model to optimize
            filter_fn: Optional function to filter parameters
            
        Returns:
            Configured optimizer
        """
        # Get parameters
        params = filter(filter_fn, model.parameters()) if filter_fn else model.parameters()

        # Get optimizer class
        optimizer_map = {
            OptimizerType.ADAM: torch.optim.Adam,
            OptimizerType.ADAMW: torch.optim.AdamW,
            OptimizerType.SGD: torch.optim.SGD,
            OptimizerType.RMSPROP: torch.optim.RMSprop,
            OptimizerType.ADAGRAD: torch.optim.Adagrad,
            OptimizerType.ADADELTA: torch.optim.Adadelta,
            OptimizerType.ADAMAX: torch.optim.Adamax,
        }

        optimizer_cls = optimizer_map.get(self.optimizer.optimizer_type)
        if optimizer_cls is None:
            msg = f"Unsupported optimizer: {self.optimizer.optimizer_type}"
            raise ValueError(msg)

        # Create optimizer
        return optimizer_cls(params, **self.optimizer.to_optimizer_kwargs())

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int | None = None,
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler from configuration.
        
        Args:
            optimizer: Optimizer to schedule
            num_training_steps: Total number of training steps
            
        Returns:
            Configured scheduler or None for constant LR
        """
        if self.scheduler.scheduler_type == SchedulerType.CONSTANT:
            return None

        scheduler_config = self.scheduler

        # Create base scheduler
        if scheduler_config.scheduler_type == SchedulerType.LINEAR:
            # Custom linear scheduler
            return self._create_linear_scheduler(optimizer, num_training_steps)

        if scheduler_config.scheduler_type == SchedulerType.COSINE:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.cosine_t_max,
                eta_min=scheduler_config.cosine_eta_min,
            )

        elif scheduler_config.scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.cosine_t_max or 10,
                T_mult=1,
                eta_min=scheduler_config.cosine_eta_min,
            )

        elif scheduler_config.scheduler_type == SchedulerType.STEP:
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.step_gamma,
            )

        elif scheduler_config.scheduler_type == SchedulerType.MULTISTEP:
            scheduler = MultiStepLR(
                optimizer,
                milestones=scheduler_config.multistep_milestones,
                gamma=scheduler_config.step_gamma,
            )

        elif scheduler_config.scheduler_type == SchedulerType.EXPONENTIAL:
            scheduler = ExponentialLR(
                optimizer,
                gamma=scheduler_config.exponential_gamma,
            )

        elif scheduler_config.scheduler_type == SchedulerType.POLYNOMIAL:
            scheduler = PolynomialLR(
                optimizer,
                total_iters=scheduler_config.polynomial_total_iters or num_training_steps,
                power=scheduler_config.polynomial_power,
            )

        elif scheduler_config.scheduler_type == SchedulerType.ONE_CYCLE:
            if num_training_steps is None:
                msg = "num_training_steps required for OneCycleLR"
                raise ValueError(msg)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=scheduler_config.one_cycle_max_lr,
                total_steps=num_training_steps,
                pct_start=scheduler_config.one_cycle_pct_start,
            )

        elif scheduler_config.scheduler_type == SchedulerType.PLATEAU:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.plateau_mode,
                factor=scheduler_config.plateau_factor,
                patience=scheduler_config.plateau_patience,
                threshold=scheduler_config.plateau_threshold,
            )

        else:
            msg = f"Unsupported scheduler: {scheduler_config.scheduler_type}"
            raise ValueError(msg)

        # Add warmup if configured
        if scheduler_config.warmup_steps > 0:
            scheduler = self._add_warmup(scheduler, optimizer, scheduler_config)

        return scheduler

    def _create_linear_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int | None,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create linear learning rate scheduler."""
        if num_training_steps is None:
            msg = "num_training_steps required for linear scheduler"
            raise ValueError(msg)

        def lr_lambda(step: int) -> float:
            return max(0.0, 1.0 - step / num_training_steps)

        return LambdaLR(optimizer, lr_lambda)

    def _add_warmup(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Add warmup to existing scheduler."""
        # Create warmup scheduler
        if config.warmup_method == "linear":
            def warmup_lambda(step: int) -> float:
                if step >= config.warmup_steps:
                    return 1.0
                return config.warmup_start_factor + (
                    1.0 - config.warmup_start_factor
                ) * step / config.warmup_steps
        else:  # exponential
            def warmup_lambda(step: int) -> float:
                if step >= config.warmup_steps:
                    return 1.0
                return config.warmup_start_factor ** (
                    1.0 - step / config.warmup_steps
                )

        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)

        # Combine with main scheduler
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[config.warmup_steps],
        )


def create_ml_config_from_args(args: Any) -> MLHyperparameters:
    """Create ML configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        MLHyperparameters configuration
    """
    # Create optimizer config
    optimizer_config = OptimizerConfig(
        optimizer_type=OptimizerType.ADAMW,
        learning_rate=getattr(args, "learning_rate", 1e-4),
        weight_decay=getattr(args, "adam_weight_decay", 0.01),
        adam_betas=(
            getattr(args, "adam_beta1", 0.9),
            getattr(args, "adam_beta2", 0.999),
        ),
        adam_eps=getattr(args, "adam_eps", 1e-8),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
    )

    # Create scheduler config
    scheduler_config = SchedulerConfig(
        scheduler_type=SchedulerType.CONSTANT,
        warmup_steps=getattr(args, "warmup_steps", 0),
    )

    # Create gradient config
    gradient_config = GradientConfig(
        clip_type=GradientClippingType.NORM if getattr(args, "grad_clip", 0) > 0 else GradientClippingType.NONE,
        clip_value=getattr(args, "grad_clip", 1.0),
        gradient_checkpointing=getattr(args, "activation_checkpointing", False),
    )

    # Create regularization config
    regularization_config = RegularizationConfig(
        use_ema=getattr(args, "use_ema", False),
        ema_decay=getattr(args, "ema_decay", 0.9999),
    )

    return MLHyperparameters(
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        gradient=gradient_config,
        regularization=regularization_config,
    )
