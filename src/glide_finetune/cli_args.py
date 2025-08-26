"""Enhanced command-line argument parsing with validation and type converters."""

from __future__ import annotations

import argparse
from pathlib import Path

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.cli_args")


# Custom type converters with validation
def path_type(path_str: str) -> Path:
    """Convert string to Path with validation."""
    path = Path(path_str)
    if not path.exists():
        msg = f"Path does not exist: {path}"
        raise argparse.ArgumentTypeError(msg)
    return path


def optional_path_type(path_str: str) -> Path | None:
    """Convert string to optional Path."""
    if not path_str or path_str.lower() == "none":
        return None
    return Path(path_str)


def positive_int(value: str) -> int:
    """Validate positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        msg = f"Value must be positive: {value}"
        raise argparse.ArgumentTypeError(msg)
    return ivalue


def non_negative_int(value: str) -> int:
    """Validate non-negative integer."""
    ivalue = int(value)
    if ivalue < 0:
        msg = f"Value must be non-negative: {value}"
        raise argparse.ArgumentTypeError(msg)
    return ivalue


def positive_float(value: str) -> float:
    """Validate positive float."""
    fvalue = float(value)
    if fvalue <= 0:
        msg = f"Value must be positive: {value}"
        raise argparse.ArgumentTypeError(msg)
    return fvalue


def probability_float(value: str) -> float:
    """Validate probability value [0, 1]."""
    fvalue = float(value)
    if not 0 <= fvalue <= 1:
        msg = f"Probability must be in [0, 1]: {value}"
        raise argparse.ArgumentTypeError(msg)
    return fvalue


def learning_rate_type(value: str) -> float:
    """Validate learning rate."""
    fvalue = float(value)
    if fvalue <= 0 or fvalue > 1:
        msg = f"Learning rate must be in (0, 1]: {value}"
        raise argparse.ArgumentTypeError(msg)
    return fvalue


def batch_size_type(value: str) -> int:
    """Validate batch size."""
    ivalue = int(value)
    if ivalue <= 0 or ivalue > 1024:
        msg = f"Batch size must be in [1, 1024]: {value}"
        raise argparse.ArgumentTypeError(msg)
    return ivalue


def device_type(value: str) -> str:
    """Validate device string."""
    if value not in ["", "cpu", "cuda"] and not value.startswith("cuda:"):
        msg = f"Device must be 'cpu', 'cuda', or 'cuda:N': {value}"
        raise argparse.ArgumentTypeError(
            msg
        )
    return value


def seed_type(value: str) -> int | None:
    """Validate seed value."""
    if value.lower() == "none":
        return None
    ivalue = int(value)
    if ivalue < 0 or ivalue >= 2**32:
        msg = f"Seed must be in [0, 2^32): {value}"
        raise argparse.ArgumentTypeError(msg)
    return ivalue


class ArgumentValidator:
    """Validator for parsed arguments."""

    @staticmethod
    def validate_data_args(args: argparse.Namespace) -> None:
        """Validate data-related arguments."""
        # WebDataset validation
        if args.use_optimized_loader and not args.use_webdataset:
            msg = "--use_optimized_loader requires --use_webdataset"
            raise ValueError(msg)

        if args.use_optimized_loader and not args.bloom_filter_path:
            logger.warning("--use_optimized_loader without --bloom_filter_path, falling back to standard loader")

        # Image size validation
        if args.side_x % 8 != 0 or args.side_y % 8 != 0:
            msg = "Image dimensions must be divisible by 8"
            raise ValueError(msg)

        if args.train_upsample and (args.side_x != 64 or args.side_y != 64):
            logger.warning("Upsampler training typically uses 64x64 base resolution")

    @staticmethod
    def validate_model_args(args: argparse.Namespace) -> None:
        """Validate model-related arguments."""
        if args.freeze_transformer and args.freeze_diffusion:
            msg = "Cannot freeze both transformer and diffusion (nothing to train)"
            raise ValueError(msg)

        if args.randomize_transformer and args.randomize_diffusion:
            msg = "Cannot randomize both transformer and diffusion"
            raise ValueError(msg)

        if args.randomize_transformer and args.freeze_transformer:
            msg = "Cannot randomize and freeze transformer simultaneously"
            raise ValueError(msg)

        if args.randomize_diffusion and args.freeze_diffusion:
            msg = "Cannot randomize and freeze diffusion simultaneously"
            raise ValueError(msg)

    @staticmethod
    def validate_training_args(args: argparse.Namespace) -> None:
        """Validate training-related arguments."""
        if args.gradient_accumulation_steps < 1:
            msg = "gradient_accumulation_steps must be >= 1"
            raise ValueError(msg)

        if args.warmup_steps > 0 and args.warmup_start_lr >= args.learning_rate:
            msg = "warmup_start_lr must be less than learning_rate"
            raise ValueError(msg)

        if args.grad_clip < 0:
            msg = "grad_clip must be non-negative (0 to disable)"
            raise ValueError(msg)

    @staticmethod
    def validate_fp16_args(args: argparse.Namespace) -> None:
        """Validate FP16-related arguments."""
        if args.use_fp16 and args.fp16_loss_scale <= 0:
            msg = "fp16_loss_scale must be positive when using FP16"
            raise ValueError(msg)

        if args.use_fp16 and args.fp16_mode not in ["auto", "conservative", "aggressive"]:
            msg = "fp16_mode must be one of: auto, conservative, aggressive"
            raise ValueError(msg)

    @staticmethod
    def validate_all(args: argparse.Namespace) -> None:
        """Run all validation checks."""
        ArgumentValidator.validate_data_args(args)
        ArgumentValidator.validate_model_args(args)
        ArgumentValidator.validate_training_args(args)
        ArgumentValidator.validate_fp16_args(args)

        # Path validation
        if args.resume_ckpt and not Path(args.resume_ckpt).exists():
            msg = f"Resume checkpoint not found: {args.resume_ckpt}"
            raise ValueError(msg)

        if args.eval_prompt_file and not Path(args.eval_prompt_file).exists():
            msg = f"Evaluation prompt file not found: {args.eval_prompt_file}"
            raise ValueError(msg)


def enhance_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Enhance an existing argument parser with custom types and validation.
    
    This function modifies the parser to use custom type converters for
    better validation and error messages.
    
    Args:
        parser: Existing argument parser to enhance
        
    Returns:
        Enhanced argument parser
    """
    # Get all argument groups
    for group in parser._action_groups:
        for action in group._group_actions:
            # Update type converters based on argument name/type
            if action.dest in ["batch_size"]:
                action.type = batch_size_type
            elif action.dest in ["learning_rate", "warmup_start_lr"]:
                action.type = learning_rate_type
            elif action.dest in ["uncond_p", "resize_ratio", "grad_clip", "fp16_loss_scale"]:
                action.type = positive_float
            elif action.dest in ["num_epochs", "num_workers", "side_x", "side_y",
                                 "gradient_accumulation_steps", "warmup_steps",
                                 "log_frequency", "sample_frequency", "save_frequency"]:
                action.type = positive_int
            elif action.dest == "seed":
                action.type = seed_type
            elif action.dest == "device":
                action.type = device_type
            elif action.dest in ["data_dir", "checkpoints_dir", "save_directory"]:
                if action.required:
                    action.type = path_type
                else:
                    action.type = optional_path_type

    return parser


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create a new enhanced argument parser with all validation.
    
    This creates a new parser with organized argument groups and
    built-in validation.
    
    Returns:
        Enhanced argument parser
    """
    parser = argparse.ArgumentParser(
        description="GLIDE Fine-tuning with Enhanced Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration group
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to training data directory or WebDataset pattern",
    )
    data_group.add_argument(
        "--batch_size",
        type=batch_size_type,
        default=1,
        help="Training batch size (1-1024)",
    )
    data_group.add_argument(
        "--side_x",
        type=positive_int,
        default=64,
        help="Image width (must be divisible by 8)",
    )
    data_group.add_argument(
        "--side_y",
        type=positive_int,
        default=64,
        help="Image height (must be divisible by 8)",
    )
    data_group.add_argument(
        "--uncond_p",
        type=probability_float,
        default=0.2,
        help="Probability of unconditional training [0, 1]",
    )

    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_path",
        type=optional_path_type,
        help="Path to pretrained model checkpoint",
    )
    model_group.add_argument(
        "--train_upsample",
        action="store_true",
        help="Train upsampler model instead of base",
    )
    model_group.add_argument(
        "--freeze_transformer",
        action="store_true",
        help="Freeze transformer weights",
    )
    model_group.add_argument(
        "--freeze_diffusion",
        action="store_true",
        help="Freeze diffusion weights",
    )

    # Training configuration group
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--learning_rate",
        type=learning_rate_type,
        default=1e-5,
        help="Learning rate (0, 1]",
    )
    training_group.add_argument(
        "--num_epochs",
        type=positive_int,
        default=100,
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--seed",
        type=seed_type,
        default=None,
        help="Random seed (0 for performance mode, None for random)",
    )
    training_group.add_argument(
        "--device",
        type=device_type,
        default="",
        help="Device to use (cpu, cuda, cuda:N)",
    )

    # FP16 configuration group
    fp16_group = parser.add_argument_group("Mixed Precision Configuration")
    fp16_group.add_argument(
        "--use_fp16",
        action="store_true",
        help="Enable FP16 mixed precision training",
    )
    fp16_group.add_argument(
        "--fp16_mode",
        choices=["auto", "conservative", "aggressive"],
        default="auto",
        help="FP16 conversion mode",
    )
    fp16_group.add_argument(
        "--fp16_loss_scale",
        type=positive_float,
        default=256.0,
        help="Initial FP16 loss scale",
    )

    # Logging configuration group
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument(
        "--log_frequency",
        type=positive_int,
        default=100,
        help="Log metrics every N steps",
    )
    logging_group.add_argument(
        "--sample_frequency",
        type=positive_int,
        default=500,
        help="Generate samples every N steps",
    )
    logging_group.add_argument(
        "--save_frequency",
        type=positive_int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    logging_group.add_argument(
        "--checkpoints_dir",
        type=optional_path_type,
        default=Path("./checkpoints"),
        help="Directory for saving checkpoints",
    )

    return parser
