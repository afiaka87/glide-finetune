"""GLIDE finetune package for text-to-image model training."""

from .checkpoint_manager import CheckpointManager
from .cli_args import ArgumentValidator, create_enhanced_parser, enhance_argument_parser
from .clip_evaluator import ClipScorer as CLIPEvaluator
from .dynamic_loss_scaler import DynamicLossScaler
from .fp16_training import FP16TrainingStep, SelectiveFP16Converter
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .loaders.loader import TextImageDataset
from .master_weight_manager import MasterWeightManager
from .memory_conscious_evaluator import MemoryConstrainedClipEvaluator
from .memory_manager import GPUMemoryMonitor, ModelMemoryManager, ModelStateManager
from .metrics_tracker import MetricsTracker
from .network_swinir import SwinIR
from .settings import (
    CheckpointSettings,
    DatasetSettings,
    FP16Mode,
    FP16Settings,
    ModelSettings,
    SamplerType,
    SamplingSettings,
    Settings,
    SystemSettings,
    TrainingMode,
    TrainingSettings,
)
from .swinir_upscaler import UpscaleSR
from .text_encoder_cache import TextEncoderCache
from .utils import (
    BloomFilter,
    count_parameters,
    create_gaussian_diffusion,
    create_model_and_diffusion,
    create_optimizer,
    freeze_layers,
    get_distributed_rank,
    get_distributed_world_size,
    get_layer_info,
    get_layer_precision_map,
    get_logger,
    is_distributed,
    load_checkpoint,
    normalize_image,
    preprocess_image,
    randomize_model_weights,
    save_checkpoint,
    setup_distributed,
    setup_logging,
    unfreeze_layers,
)
from .loaders.wds_loader import glide_wds_loader
from .loaders.wds_loader_distributed import create_distributed_wds_dataloader, distributed_wds_loader
from .loaders.wds_loader_optimized import create_optimized_dataloader, glide_wds_loader_optimized
from .loaders.wds_resumable_loader import glide_wds_resumable_loader

__version__ = "0.2.0"

__all__ = [
    # CLI and argument parsing
    "ArgumentValidator",
    # Utilities (exported from utils submodule)
    "BloomFilter",
    "CLIPEvaluator",
    # Core classes
    "CheckpointManager",
    "CheckpointSettings",
    "DatasetSettings",
    "DynamicLossScaler",
    "FP16Mode",
    "FP16Settings",
    # Samplers - removed (they are functions, not classes)
    # FP16 training
    "FP16TrainingStep",
    "GPUMemoryMonitor",
    "MasterWeightManager",
    "MemoryConstrainedClipEvaluator",
    "MetricsTracker",
    "ModelMemoryManager",
    "ModelSettings",
    "ModelStateManager",
    "SamplerType",
    "SamplingSettings",
    "SelectiveFP16Converter",
    # Settings and configuration
    "Settings",
    # Models
    "SwinIR",
    "SystemSettings",
    "TextEncoderCache",
    # Data loading
    "TextImageDataset",
    "TrainingMode",
    "TrainingSettings",
    "UpscaleSR",
    # Version
    "__version__",
    "convert_module_to_f16",
    "convert_module_to_f32",
    "count_parameters",
    "create_distributed_wds_dataloader",
    "create_enhanced_parser",
    "create_gaussian_diffusion",
    "create_model_and_diffusion",
    "create_optimized_dataloader",
    "create_optimizer",
    "distributed_wds_loader",
    "enhance_argument_parser",
    "freeze_layers",
    "get_distributed_rank",
    "get_distributed_world_size",
    "get_layer_info",
    "get_layer_precision_map",
    "get_logger",
    "glide_wds_loader",
    "glide_wds_loader_optimized",
    "glide_wds_resumable_loader",
    "is_distributed",
    "load_checkpoint",
    "normalize_image",
    "preprocess_image",
    "randomize_model_weights",
    "save_checkpoint",
    "setup_distributed",
    "setup_logging",
    "unfreeze_layers",
]
