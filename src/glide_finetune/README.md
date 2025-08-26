# GLIDE Finetune Module Documentation

This directory contains the core implementation for finetuning OpenAI's GLIDE text-to-image diffusion models. The codebase is organized into modular components for training, evaluation, data loading, and optimization.

## Core Training Files

### `glide_finetune.py`
Main training utilities and image generation functions.
- `create_image_grid()` - Creates grid of PIL images for visualization
- `DEFAULT_TEST_PROMPTS` - Curated evaluation prompts
- Text encoder caching integration
- Training loop integration with metrics and WandB

### `training_pipeline.py`
Core training pipeline logic separated from CLI.
- `InterruptHandler` - Graceful shutdown handling
- `setup_training()` - Initialize training components
- Training loop implementation with checkpointing
- Warmup scheduler creation

### `noisy_clip_finetune.py`
CLIP-guided noisy training implementation for improved quality.

### `training_types.py`
Type definitions and dataclasses for training components.

## Model Management

### `model_loader.py`
Unified model loading and configuration.
- Model loading from checkpoints
- Architecture detection and conversion
- Base/upsampler model handling

### `checkpoint_manager.py`
Robust checkpoint saving and recovery system.
- `CheckpointManager` - Atomic saves, auto-recovery, interrupt handling
- Checkpoint versioning and metadata
- Resume from interrupted training

### `text_encoder_cache.py`
Caching for frozen text encoder outputs.
- `TextEncoderCache` - LRU cache for text embeddings
- `cached_text_encoder_forward()` - Cached forward pass wrapper
- Reduces redundant computation for frozen transformers

## Data Loading

### `loader.py`
Local dataset loading for image-text pairs.
- `TextImageDataset` - PyTorch dataset for local files
- `random_resized_crop()` - Random crop augmentation
- `get_image_files_dict()` - File discovery utilities
- Caption loading and tokenization

### `wds_loader.py`
WebDataset loader for streaming large-scale datasets.
- `glide_wds_loader()` - Main WebDataset pipeline
- Support for LAION, Alamy, synthetic datasets
- NSFW filtering and aspect ratio constraints
- Metadata handling and filtering

### `wds_loader_distributed.py`
Distributed WebDataset loading for multi-GPU training.
- Sharding and worker coordination
- Distributed sampler integration

### `wds_loader_optimized.py`
Performance-optimized WebDataset loading.
- Batched processing
- Memory-efficient streaming

### `wds_resumable_loader.py`
WebDataset loader with resume capability.
- State tracking for interrupted training
- Deterministic resumption

## Evaluation & Metrics

### `clip_evaluator.py`
CLIP-based image quality evaluation.
- `CLIPEvaluator` - CLIP score computation
- `EvaluationConfig` - Evaluation settings
- Base model comparison and win-rate metrics
- Integration with OpenCLIP models

### `metrics_tracker.py`
Comprehensive training metrics tracking.
- `MetricsTracker` - Central metrics aggregation
- `RollingAverage` - Smoothed metric tracking
- Gradient statistics and memory monitoring
- CLIP score integration
- `print_model_info()` - Model parameter statistics

### `memory_conscious_evaluator.py`
Memory-efficient evaluation for large batches.
- Streaming evaluation to reduce memory usage
- Batch processing with memory limits

### `enhanced_samplers.py`
Advanced sampling algorithms for generation.
- DPM++, Euler, and other samplers
- Sampling configuration

## FP16/Mixed Precision Training

### `fp16_training.py`
Production-ready mixed precision training orchestration.
- `FP16TrainingConfig` - Comprehensive FP16 settings
- `FP16TrainingStep` - Mixed precision training wrapper
- `SelectiveFP16Converter` - Intelligent layer precision selection
- NaN recovery and stability features

### `fp16_util.py`
Legacy FP16 utilities (being phased out).
- `EMA` - Exponential moving average
- Basic FP16 conversion utilities

### `dynamic_loss_scaler.py`
Advanced loss scaling for FP16 training.
- `DynamicLossScaler` - Adaptive loss scaling
- `NaNRecoverySystem` - Automatic NaN detection and recovery
- Gradient overflow detection
- Conservative scaling (starts at 256)

### `master_weight_manager.py`
FP32 master weight management for FP16 training.
- `MasterWeightManager` - FP32 weight copies
- Gradient accumulation in FP32
- Parameter synchronization

## Specialized Components

### `swinir_upscaler.py`
SwinIR super-resolution upscaling.
- `UpscaleSR` - 2x/4x/8x upscaling
- Integration with HuggingFace transformers
- FP16 inference support

### `network_swinir.py`
SwinIR network architecture implementation.

## Configuration

### `cli_args.py`
Command-line argument parsing and validation.
- Comprehensive training arguments
- WebDataset configuration
- Sampling and evaluation settings

### `settings.py`
Global settings and configuration management.

### `ml_settings.py`
Machine learning specific settings.
- Optimizer configurations
- Learning rate schedules
- Training hyperparameters

## Training Strategies

### `strategies/`
Different training strategy implementations.

#### `single_gpu.py`
Single GPU training strategy.

#### `multi_gpu.py`
Multi-GPU distributed training using Accelerate.
- `MultiGPUStrategy` - Distributed training orchestration
- Gradient synchronization
- Model sharding

#### `fp16.py`
FP16 training strategy integration.

## Augmentation

### `augmentation/`
Data augmentation techniques.

#### `cutout_augmentation.py`
CLIP-guided cutout augmentation.
- `CutoutConfig` - Augmentation settings
- `TimestepAwareCutouts` - Timestep-scaled cutouts
- Multi-scale cutout generation

## Utilities

### `utils/`
Shared utility functions organized by domain.

#### Core Utilities
- `glide_util.py` - GLIDE model utilities, sampling functions
- `train_util.manipulations` - Training helpers, image conversions
- `common_utils.py` - General purpose utilities

#### Model & Training
- `model_utils.py` - Model manipulation and optimization
- `gradient_utils.py` - Gradient processing and clipping
- `freeze_utils.py` - Layer freezing utilities
- `layer_utils.py` - Layer manipulation helpers

#### System & Infrastructure
- `device_utils.py` - GPU/device management
- `distributed_utils.py` - Distributed training utilities
- `logging_utils.py` - Logging configuration and helpers
- `seed_utils.py` - Deterministic seed management
- `randomize_utils.py` - Randomization utilities

#### Data Processing
- `image_processing.py` - Image manipulation utilities
- `bloom_filter_utils.py` - Bloom filter for deduplication

## Evaluation Module

### `evaluation/`
Dedicated evaluation components.

#### `sampler.py`
Sampling utilities for evaluation.
- `SamplingConfig` - Sampling parameters
- Consistent seed management
- Batch generation

#### `scorer.py`
Quality scoring and metrics computation.
- CLIP score calculation
- FID/IS metrics
- Comparative evaluation

#### `clip_evaluator.py`
CLIP-based evaluation implementation.
- Model comparison
- Win-rate computation

## Key Functions by Category

### Training Loop
- `training_pipeline.setup_training()` - Initialize all components
- `training_pipeline.train_step()` - Single training iteration
- `checkpoint_manager.save_checkpoint()` - Save training state

### Model Operations
- `model_loader.load_model()` - Load GLIDE models
- `text_encoder_cache.cached_forward()` - Cached text encoding
- `glide_util.sample()` - Generate images from text

### Data Processing
- `loader.TextImageDataset.__getitem__()` - Load image-caption pairs
- `wds_loader.glide_wds_loader()` - Stream WebDataset samples
- `image_processing.trim_white_padding_pil()` - Remove image borders

### Optimization
- `fp16_training.FP16TrainingStep.step()` - Mixed precision step
- `dynamic_loss_scaler.scale_loss()` - Scale loss for FP16
- `master_weight_manager.sync_weights()` - Update FP32 masters

### Evaluation
- `clip_evaluator.evaluate()` - Compute CLIP scores
- `metrics_tracker.log_metrics()` - Log training metrics
- `sampler.generate_samples()` - Create evaluation images

## Usage Patterns

1. **Training**: Main entry via `train_glide.py` → `training_pipeline` → model/data/optimizer setup
2. **FP16 Training**: Enable via `--use_fp16` → `fp16_training` orchestrates mixed precision
3. **Evaluation**: `clip_evaluator` + `metrics_tracker` provide quality metrics
4. **Checkpointing**: `checkpoint_manager` handles saves/resumes automatically
5. **Data Loading**: Choose between `loader` (local) or `wds_loader` (streaming)

## Architecture Notes

- Modular design allows swapping components (samplers, strategies, data sources)
- FP16 implementation is production-ready with full stability features
- Checkpoint system is atomic and corruption-resistant
- WebDataset integration supports multiple dataset formats
- Evaluation is memory-conscious for large-scale testing