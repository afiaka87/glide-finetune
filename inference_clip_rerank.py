#!/usr/bin/env python3
"""
Inference script for GLIDE with CLIP re-ranking.

This script generates multiple images from a single prompt and uses CLIP to re-rank
them based on similarity to the prompt text. Supports both OpenAI and LAION CLIP models,
ESRGAN upscaling, and CLIP ensemble ranking.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import wandb

from glide_finetune.glide_util import load_model, sample
from glide_finetune.train_util import save_image_compressed
from glide_finetune.esrgan import ESRGANUpsampler

# Enable performance optimizations
torch.set_grad_enabled(False)  # Disable gradients globally for inference
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cudnn
torch.set_float32_matmul_precision('high')  # Use TF32 for matmul operations

# Set environment variables for additional optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image."""
    # Move to CPU and convert to numpy
    img_np = tensor.cpu().numpy()
    
    # Convert from [-1, 1] to [0, 255]
    img_np = (img_np + 1) * 127.5
    img_np = img_np.clip(0, 255).astype(np.uint8)
    
    # Transpose from CHW to HWC
    img_np = img_np.transpose(1, 2, 0)
    
    return Image.fromarray(img_np)


def load_clip_for_ranking(model_name: str, device: str = "cuda", cache_dir: Optional[str] = None, 
                         compile_model: bool = False, compile_mode: str = "reduce-overhead"):
    """Load CLIP model for image-text similarity ranking. Supports OpenAI and LAION models."""
    
    # OpenAI CLIP models
    openai_models = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", 
                     "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"]
    
    if model_name in openai_models:
        # Use OpenAI CLIP
        try:
            import clip
        except ImportError:
            raise ImportError(
                "Please install OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git"
            )
        
        print(f"Loading OpenAI CLIP model: {model_name}")
        try:
            clip_model, clip_preprocess = clip.load(model_name, device=device, download_root=cache_dir, jit=False)
            clip_model.eval()
        except Exception as e:
            print(f"\nError loading {model_name}: {e}")
            raise
        
        # Compile CLIP model if requested
        if compile_model and hasattr(torch, 'compile'):
            print(f"  Compiling CLIP model with mode: {compile_mode}")
            clip_model = torch.compile(clip_model, mode=compile_mode, fullgraph=False)
        
        print(f"\n  ✓ {model_name} loaded successfully")
        
        # Create wrapper functions for consistent interface
        def encode_text(text_list):
            tokens = clip.tokenize(text_list, truncate=True).to(device)
            with torch.no_grad():
                features = clip_model.encode_text(tokens)
            return features
        
        def encode_image(images):
            with torch.no_grad():
                features = clip_model.encode_image(images)
            return features
            
        return clip_model, clip_preprocess, encode_text, encode_image
    
    else:
        # Use LAION open_clip
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "Please install open_clip for LAION models: pip install open_clip_torch"
            )
        
        # Parse model name (format: "modelname/dataset")
        if "/" in model_name:
            model_arch, pretrained = model_name.split("/", 1)
        else:
            # Default to openai pretrained if not specified
            model_arch = model_name
            pretrained = "openai"
        
        print(f"Loading LAION CLIP model: {model_arch} (pretrained: {pretrained})")
        
        # List available models to help user
        available = open_clip.list_pretrained()
        matching = [p for m, p in available if m.lower() == model_arch.lower()]
        
        if pretrained not in matching and len(matching) > 0:
            print(f"Available pretrained versions for {model_arch}: {matching}")
        
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            model_arch, 
            pretrained=pretrained,
            device=device,
            cache_dir=cache_dir
        )
        
        tokenizer = open_clip.get_tokenizer(model_arch)
        clip_model.eval()
        
        # Compile CLIP model if requested
        if compile_model and hasattr(torch, 'compile'):
            print(f"  Compiling CLIP model with mode: {compile_mode}")
            clip_model = torch.compile(clip_model, mode=compile_mode, fullgraph=False)
        
        print(f"  ✓ {model_name} loaded successfully")
        
        # Create wrapper functions
        def encode_text(text_list):
            tokens = tokenizer(text_list).to(device)
            with torch.no_grad():
                features = clip_model.encode_text(tokens)
            return features
        
        def encode_image(images):
            with torch.no_grad():
                features = clip_model.encode_image(images)
            return features
            
        return clip_model, clip_preprocess, encode_text, encode_image


def compute_clip_scores(
    images: List[Image.Image],
    prompt: str,
    clip_preprocess,
    encode_text_fn,
    encode_image_fn,
    device: str = "cuda",
    use_amp: bool = False
) -> np.ndarray:
    """Compute CLIP similarity scores between images and prompt."""
    # Preprocess images
    image_tensors = torch.stack([clip_preprocess(img) for img in images]).to(device)
    
    # Get image and text features with optional AMP
    if use_amp and device == "cuda":
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            image_features = encode_image_fn(image_tensors)
            text_features = encode_text_fn([prompt])
    else:
        image_features = encode_image_fn(image_tensors)
        text_features = encode_text_fn([prompt])
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity scores
    similarity = (100.0 * image_features @ text_features.T).squeeze()
    
    return similarity.cpu().numpy()


def generate_and_rank(
    model,
    options,
    prompt: str,
    num_samples: int,
    sampler_name: str,
    num_steps: int,
    guidance_scale: float,
    device: str,
    clip_models: List[Tuple],  # List of (preprocess, encode_text, encode_image) tuples
    batch_size: int = 4,
    seed: Optional[int] = None,
    esrgan: Optional[ESRGANUpsampler] = None,
    use_amp: bool = False,
) -> Tuple[List[Image.Image], List[Image.Image], np.ndarray, int, float]:
    """Generate multiple images and rank them with CLIP ensemble."""
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    all_images = []
    
    # Generate images in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} images in {num_batches} batches...")
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        print(f"  Batch {batch_idx + 1}/{num_batches} (size: {current_batch_size})...")
        
        # Sample with optional AMP
        if use_amp and device == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                samples = sample(
                    glide_model=model,
                    glide_options=options,
                    side_x=64,
                    side_y=64,
                    prompt=prompt,
                    batch_size=current_batch_size,
                    guidance_scale=guidance_scale,
                    device=device,
                    prediction_respacing=str(num_steps),
                    sampler_name=sampler_name,
                )
        else:
            samples = sample(
                glide_model=model,
                glide_options=options,
                side_x=64,
                side_y=64,
                prompt=prompt,
                batch_size=current_batch_size,
                guidance_scale=guidance_scale,
                device=device,
                prediction_respacing=str(num_steps),
                sampler_name=sampler_name,
            )
        
        # Convert to PIL images
        for i in range(samples.shape[0]):
            img = tensor_to_pil(samples[i])
            all_images.append(img)
    
    generation_time = time.time() - start_time
    print(f"Generation complete in {generation_time:.2f}s")
    
    # Upsample images if ESRGAN is provided
    upsampled_images = []
    if esrgan is not None:
        print("Upsampling images with ESRGAN...")
        for idx, img in enumerate(all_images):
            print(f"  Upsampling image {idx + 1}/{len(all_images)}...")
            upsampled_img = esrgan.upsample_pil(img)
            upsampled_images.append(upsampled_img)
        images_for_ranking = upsampled_images
    else:
        images_for_ranking = all_images
    
    # Compute CLIP scores with ensemble
    print("Computing CLIP scores...")
    all_scores = []
    
    for model_idx, (clip_preprocess, encode_text_fn, encode_image_fn) in enumerate(clip_models):
        scores = compute_clip_scores(
            images_for_ranking, prompt, clip_preprocess, encode_text_fn, encode_image_fn, device, use_amp
        )
        all_scores.append(scores)
        if len(clip_models) > 1:
            print(f"  Model {model_idx + 1}/{len(clip_models)} scores computed")
    
    # Average scores across models
    if len(all_scores) > 1:
        ensemble_scores = np.mean(all_scores, axis=0)
        print(f"Ensemble averaging across {len(clip_models)} models")
    else:
        ensemble_scores = all_scores[0]
    
    # Get best image index
    best_idx = np.argmax(ensemble_scores)
    
    return all_images, upsampled_images, ensemble_scores, best_idx, generation_time


def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from a text file."""
    with open(filepath, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def list_available_models():
    """List all available CLIP models."""
    print("\nAvailable CLIP Models:")
    print("=" * 60)
    
    print("\nOpenAI CLIP Models:")
    print("-" * 30)
    openai_models = [
        "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
        "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"
    ]
    for model in openai_models:
        print(f"  {model}")
    
    try:
        import open_clip
        print("\nLAION/OpenCLIP Models:")
        print("-" * 30)
        print("Format: model_name/pretrained_dataset")
        print("\nCommon models:")
        common_models = [
            "ViT-B-32/laion2b_s34b_b79k",
            "ViT-B-32/laion400m_e31",
            "ViT-B-32/laion400m_e32",
            "ViT-L-14/laion2b_s32b_b82k",
            "ViT-L-14/laion400m_e31",
            "ViT-L-14/laion400m_e32",
            "ViT-H-14/laion2b_s32b_b79k",
            "ViT-g-14/laion2b_s12b_b42k",
            "ViT-bigG-14/laion2b_s39b_b160k",
            "convnext_base_w/laion2b_s13b_b82k",
            "convnext_base_w_320/laion_aesthetic_s13b_b82k",
            "coca_ViT-L-14/laion2b_s13b_b90k",
        ]
        for model in common_models:
            print(f"  {model}")
        
        print("\nFor full list of available pretrained models, run:")
        print("  python -c \"import open_clip; print(open_clip.list_pretrained())\"")
        
    except ImportError:
        print("\nLAION models not available (install open_clip_torch)")
    
    print("=" * 60)


def wandb_setup_inference(args, run_dir: Path):
    """Setup wandb for inference logging."""
    config = {
        "mode": "inference",
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "sampler": args.sampler,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "clip_models": args.clip_model,
        "use_esrgan": args.use_esrgan,
        "device": args.device,
        "use_fp16": args.use_fp16,
        "compile": args.compile,
        "compile_clip": args.compile_clip,
        "compile_mode": args.compile_mode,
        "amp": args.amp,
        "output_dir": str(run_dir),
    }
    
    if args.use_esrgan:
        config["esrgan_model"] = args.esrgan_model
    
    # Initialize wandb run
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config,
        tags=["inference", "clip-rerank"],
        notes=f"GLIDE inference with CLIP re-ranking on {args.checkpoint}",
    )
    
    return run


def log_results_to_wandb(
    wandb_run,
    prompt: str,
    prompt_idx: int,
    images: List[Image.Image],
    upsampled_images: List[Image.Image],
    scores: np.ndarray,
    best_idx: int,
    generation_time: float,
):
    """Log inference results to wandb with galleries and metrics."""
    if wandb_run is None:
        return
    
    # Create wandb.Table for detailed metrics
    columns = ["prompt", "image_idx", "clip_score", "is_best", "64x64_image", "256x256_image"]
    table_data = []
    
    # Sort indices by score for organized display
    sorted_indices = np.argsort(scores)[::-1]
    
    # Log individual images and build table data
    wandb_images_64 = []
    wandb_images_256 = []
    
    for rank, idx in enumerate(sorted_indices):
        score = scores[idx]
        is_best = idx == best_idx
        caption = f"Rank {rank+1} | Score: {score:.2f}"
        if is_best:
            caption += " | ★ BEST"
        
        # Create wandb images
        wandb_img_64 = wandb.Image(images[idx], caption=caption)
        wandb_images_64.append(wandb_img_64)
        
        wandb_img_256 = None
        if upsampled_images and len(upsampled_images) > idx:
            wandb_img_256 = wandb.Image(upsampled_images[idx], caption=caption)
            wandb_images_256.append(wandb_img_256)
        
        # Add row to table
        table_data.append([
            prompt,
            idx,
            score,
            is_best,
            wandb_img_64,
            wandb_img_256 if wandb_img_256 else None
        ])
    
    # Create table
    metrics_table = wandb.Table(columns=columns, data=table_data)
    
    # Log metrics and images
    log_dict = {
        f"prompt_{prompt_idx:04d}/metrics_table": metrics_table,
        f"prompt_{prompt_idx:04d}/gallery_64x64": wandb_images_64,
        f"prompt_{prompt_idx:04d}/best_score": scores[best_idx],
        f"prompt_{prompt_idx:04d}/mean_score": np.mean(scores),
        f"prompt_{prompt_idx:04d}/std_score": np.std(scores),
        f"prompt_{prompt_idx:04d}/generation_time": generation_time,
    }
    
    if wandb_images_256:
        log_dict[f"prompt_{prompt_idx:04d}/gallery_256x256"] = wandb_images_256
    
    # Create grids if we have a square number of images
    if len(images) in [4, 16, 64]:
        grid_size = int(np.sqrt(len(images)))
        
        # Create 64x64 grid
        grid_64 = Image.new("RGB", (images[0].width * grid_size, images[0].height * grid_size))
        for idx, img in enumerate(images):
            row = idx // grid_size
            col = idx % grid_size
            grid_64.paste(img, (col * img.width, row * img.height))
        
        log_dict[f"prompt_{prompt_idx:04d}/grid_64x64"] = wandb.Image(
            grid_64, caption=f"{prompt} | 64x64 grid"
        )
        
        # Create 256x256 grid if available
        if upsampled_images:
            grid_256 = Image.new("RGB", 
                (upsampled_images[0].width * grid_size, upsampled_images[0].height * grid_size))
            for idx, img in enumerate(upsampled_images):
                row = idx // grid_size
                col = idx % grid_size
                grid_256.paste(img, (col * img.width, row * img.height))
            
            log_dict[f"prompt_{prompt_idx:04d}/grid_256x256"] = wandb.Image(
                grid_256, caption=f"{prompt} | 256x256 grid"
            )
    
    # Log the best image prominently
    log_dict[f"prompt_{prompt_idx:04d}/best_64x64"] = wandb.Image(
        images[best_idx], caption=f"Best | Score: {scores[best_idx]:.2f}"
    )
    if upsampled_images:
        log_dict[f"prompt_{prompt_idx:04d}/best_256x256"] = wandb.Image(
            upsampled_images[best_idx], caption=f"Best | Score: {scores[best_idx]:.2f}"
        )
    
    wandb_run.log(log_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with GLIDE and re-rank with CLIP"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="synthetic-1m-dalle-high-quality.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "upsample", "base-inpaint", "upsample-inpaint"],
        help="Type of model to load",
    )
    
    # Generation arguments
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation",
    )
    prompt_group.add_argument(
        "--prompt_file",
        type=str,
        default="examples/trippy_prompts_32.txt",
        help="File containing line-separated prompts (default: examples/trippy_prompts_32.txt)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of images to generate (default: 16)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["plms", "ddim", "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras"],
        help="Sampler to use (default: euler)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of sampling steps (default: 30)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    # CLIP arguments
    parser.add_argument(
        "--clip_model",
        type=str,
        nargs="+",
        default=["ViT-L/14"],
        help="CLIP model(s) for ranking. Multiple models will be ensembled.",
    )
    parser.add_argument(
        "--clip_cache_dir",
        type=str,
        default="./clip_models",
        help="Directory to cache CLIP models",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available CLIP models and exit",
    )
    
    # ESRGAN arguments
    parser.add_argument(
        "--use_esrgan",
        action="store_true",
        help="Use ESRGAN to upsample images before ranking (64x64 -> 256x256)",
    )
    parser.add_argument(
        "--esrgan_model",
        type=str,
        default="RealESRGAN_x4plus",
        choices=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"],
        help="ESRGAN model to use",
    )
    parser.add_argument(
        "--esrgan_cache_dir",
        type=str,
        default="./esrgan_models",
        help="Directory to cache ESRGAN models",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inference",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save all generated images (not just the best)",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 precision for model weights",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for GLIDE model (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile_clip",
        action="store_true",
        help="Also compile CLIP models for faster ranking",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Torch compile mode (reduce-overhead is fastest for inference)",
    )
    parser.add_argument(
        "--compile_cache_dir",
        type=str,
        default="./torch_compile_cache",
        help="Directory to cache compiled models (speeds up subsequent runs)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use Automatic Mixed Precision (AMP) for faster inference",
    )
    
    # Wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="glide-inference",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to auto-generated)",
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        list_available_models()
        return
    
    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        try:
            prompts = load_prompts_from_file(args.prompt_file)
            if not prompts:
                parser.error(f"No prompts found in {args.prompt_file}")
            print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
        except FileNotFoundError:
            parser.error(f"Prompt file not found: {args.prompt_file}")
    else:
        # Use default prompt file if neither specified
        try:
            prompts = load_prompts_from_file("examples/trippy_prompts_32.txt")
            print(f"Using default prompts from examples/trippy_prompts_32.txt ({len(prompts)} prompts)")
        except FileNotFoundError:
            parser.error("No prompt specified and default prompt file not found")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next available subdirectory
    existing_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not existing_dirs:
        next_num = 0
    else:
        numbers = [int(d.name) for d in existing_dirs]
        next_num = max(numbers) + 1
    
    run_dir = output_dir / f"{next_num:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    
    # Initialize wandb if requested
    wandb_run = None
    if args.wandb:
        wandb_run = wandb_setup_inference(args, run_dir)
        print(f"Wandb run initialized: {wandb_run.name}")
    
    # Load GLIDE model
    print(f"\nLoading GLIDE model from {args.checkpoint}...")
    model, _, options = load_model(
        glide_path=args.checkpoint,
        use_fp16=args.use_fp16,
        model_type=args.model_type,
    )
    model.eval()
    model = model.to(args.device)
    
    # Apply torch.compile if requested
    if args.compile and hasattr(torch, 'compile'):
        # Set up compile caching
        cache_dir = Path(args.compile_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable compile caching
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        
        # Set cache directory via environment variable
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache_dir)
        
        print(f"Compiling model with mode: {args.compile_mode}")
        print(f"Cache directory: {cache_dir}")
        print("Note: First run will be slower due to compilation, subsequent runs will use cache")
        
        # Compile with caching enabled
        model = torch.compile(
            model, 
            mode=args.compile_mode,
            fullgraph=False,  # Allow graph breaks for better caching
            dynamic=False,    # Static shapes for better caching
        )
    elif args.compile:
        print("Warning: torch.compile not available in this PyTorch version")
    
    # Load ESRGAN if requested
    esrgan = None
    if args.use_esrgan:
        print(f"\nLoading ESRGAN model: {args.esrgan_model}")
        esrgan = ESRGANUpsampler(
            model_name=args.esrgan_model,
            device=args.device,
            cache_dir=args.esrgan_cache_dir
        )
    
    # Load CLIP models
    clip_models = []
    print(f"\nLoading {len(args.clip_model)} CLIP model(s)...")
    if args.compile_clip:
        print("CLIP models will be compiled for faster inference")
    sys.stdout.flush()
    for i, model_name in enumerate(args.clip_model):
        print(f"  [{i+1}/{len(args.clip_model)}] Loading: {model_name}")
        sys.stdout.flush()
        _, clip_preprocess, encode_text_fn, encode_image_fn = load_clip_for_ranking(
            model_name, args.device, args.clip_cache_dir,
            compile_model=args.compile_clip,
            compile_mode=args.compile_mode
        )
        clip_models.append((clip_preprocess, encode_text_fn, encode_image_fn))
        sys.stdout.flush()
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(prompts):
        if len(prompts) > 1:
            print(f"\n{'='*60}")
            print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
            print(f"{'='*60}")
        
        # Create prompt-specific output directory
        if len(prompts) > 1:
            prompt_dir = run_dir / f"prompt_{prompt_idx:04d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
        else:
            prompt_dir = run_dir
        
        # Generate and rank
        print(f"\nStarting generation and ranking...")
        if args.amp:
            print("Using Automatic Mixed Precision (AMP)")
        sys.stdout.flush()
        
        images, upsampled_images, scores, best_idx, generation_time = generate_and_rank(
            model=model,
            options=options,
            prompt=prompt,
            num_samples=args.num_samples,
            sampler_name=args.sampler,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            device=args.device,
            clip_models=clip_models,
            batch_size=args.batch_size,
            seed=args.seed if args.seed is not None else prompt_idx,  # Different seed per prompt if not specified
            esrgan=esrgan,
            use_amp=args.amp,
        )
    
        # Save results
        print(f"\nCLIP Scores (higher is better):")
        print("-" * 40)
        
        # Sort indices by score for display
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            score = scores[idx]
            status = "★ BEST" if idx == best_idx else ""
            print(f"Image {idx:2d}: {score:6.2f} {status}")
            
            # Save image if requested
            if args.save_all or idx == best_idx:
                # Save original 64x64 image
                if idx == best_idx:
                    filename = "best_64x64.png"
                else:
                    filename = f"sample_{idx:02d}_score_{score:.2f}_64x64.png"
                
                img_path = prompt_dir / filename
                save_image_compressed(images[idx], img_path)
                
                # Save upsampled image if available
                if upsampled_images and len(upsampled_images) > idx:
                    if idx == best_idx:
                        filename = "best.png"
                    else:
                        filename = f"sample_{idx:02d}_score_{score:.2f}.png"
                    
                    img_path = prompt_dir / filename
                    save_image_compressed(upsampled_images[idx], img_path)
    
        # Create a grid of all images
        if args.save_all and len(images) in [4, 16, 64]:
            print("\nCreating image grids...")
            
            # Create grid for original images
            grid_size = int(np.sqrt(len(images)))
            grid_width = images[0].width * grid_size
            grid_height = images[0].height * grid_size
            
            grid = Image.new("RGB", (grid_width, grid_height))
            
            for idx, img in enumerate(images):
                row = idx // grid_size
                col = idx % grid_size
                grid.paste(img, (col * img.width, row * img.height))
            
            grid_path = prompt_dir / "grid_64x64.png"
            save_image_compressed(grid, grid_path)
            print(f"Saved 64x64 grid to: {grid_path}")
            
            # Create grid for upsampled images if available
            if upsampled_images:
                grid_width = upsampled_images[0].width * grid_size
                grid_height = upsampled_images[0].height * grid_size
                
                grid = Image.new("RGB", (grid_width, grid_height))
                
                for idx, img in enumerate(upsampled_images):
                    row = idx // grid_size
                    col = idx % grid_size
                    grid.paste(img, (col * img.width, row * img.height))
                
                grid_path = prompt_dir / "grid.png"
                save_image_compressed(grid, grid_path)
                print(f"Saved upsampled grid to: {grid_path}")
        
        # Save generation info
        info_path = prompt_dir / "info.txt"
        with open(info_path, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"Sampler: {args.sampler}\n")
            f.write(f"Steps: {args.steps}\n")
            f.write(f"Guidance Scale: {args.guidance_scale}\n")
            f.write(f"Num Samples: {args.num_samples}\n")
            f.write(f"CLIP Models ({len(args.clip_model)}): {', '.join(args.clip_model)}\n")
            if args.use_esrgan:
                f.write(f"ESRGAN Model: {args.esrgan_model}\n")
            f.write(f"Best Index: {best_idx}\n")
            f.write(f"Best Score: {scores[best_idx]:.2f}\n")
            f.write(f"\nAll Scores:\n")
            for idx, score in enumerate(scores):
                f.write(f"  Image {idx}: {score:.2f}\n")
        
        best_filename = "best.png" if upsampled_images else "best_64x64.png"
        print(f"\n✅ Best image saved to: {prompt_dir}/{best_filename}")
        print(f"   CLIP score: {scores[best_idx]:.2f}")
        if len(args.clip_model) > 1:
            print(f"   (Ensemble of {len(args.clip_model)} models)")
        
        # Log results to wandb
        log_results_to_wandb(
            wandb_run=wandb_run,
            prompt=prompt,
            prompt_idx=prompt_idx,
            images=images,
            upsampled_images=upsampled_images,
            scores=scores,
            best_idx=best_idx,
            generation_time=generation_time,
        )
    
    # Final summary for multiple prompts
    if len(prompts) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: Processed {len(prompts)} prompts")
        print(f"Results saved to: {run_dir}")
        print(f"{'='*60}")
    
    # Finish wandb run
    if wandb_run:
        # Log final summary
        wandb_run.summary["total_prompts"] = len(prompts)
        wandb_run.summary["total_images_generated"] = len(prompts) * args.num_samples
        wandb_run.summary["output_directory"] = str(run_dir)
        
        # Mark run as finished
        wandb_run.finish()
        print("\nWandb run finished successfully")


if __name__ == "__main__":
    main()