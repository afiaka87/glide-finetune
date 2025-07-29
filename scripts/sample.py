#!/usr/bin/env python3
"""Script for sampling from GLIDE models with various samplers and options."""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from glide_finetune.esrgan import ESRGANUpsampler
from glide_finetune.glide_util import load_model, sample


def get_vram_usage() -> dict:
    """Get current VRAM usage statistics."""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0, "total_gb": 0}


def print_vram_usage(label: str = ""):
    """Print current VRAM usage."""
    usage = get_vram_usage()
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}VRAM: {usage['allocated_gb']:.2f}/{usage['total_gb']:.2f} GB allocated, "
        f"{usage['reserved_gb']:.2f} GB reserved"
    )


def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from a text file."""
    with open(filepath, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    valid_counts = [2, 4, 8, 16, 32]
    if len(prompts) not in valid_counts:
        raise ValueError(
            f"Prompt file must contain exactly {', '.join(map(str, valid_counts))} prompts. "
            f"Found {len(prompts)} prompts."
        )

    return prompts


def create_image_grid(images: List[Image.Image]) -> Image.Image:
    """Create a square grid from a list of images."""
    n = len(images)
    grid_size = int(np.sqrt(n))

    if grid_size * grid_size != n:
        raise ValueError(f"Number of images ({n}) must be a perfect square")

    img_width, img_height = images[0].size
    grid_width = img_width * grid_size
    grid_height = img_height * grid_size

    grid = Image.new("RGB", (grid_width, grid_height))

    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid.paste(img, (col * img_width, row * img_height))

    return grid


def get_next_output_dir(base_dir: Path) -> Path:
    """Get the next available output directory with 5-digit numbering."""
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find existing directories
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]

    if not existing_dirs:
        next_num = 0
    else:
        # Get the highest number
        numbers = [int(d.name) for d in existing_dirs]
        next_num = max(numbers) + 1

    # Create directory with 5-digit format
    output_dir = base_dir / f"{next_num:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


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


def sample_single(
    model,
    options,
    prompt: str,
    sampler_name: str,
    num_steps: int,
    guidance_scale: float,
    device: str,
    batch_size: int = 1,
    seed: Optional[int] = None,
    use_karras: bool = False,
    esrgan: Optional[ESRGANUpsampler] = None,
) -> Tuple[List[Image.Image], List[Optional[Image.Image]], float]:
    """Sample images with a single sampler and return timing."""

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Prepare sampler kwargs
    sampler_kwargs = {}
    if sampler_name in ["euler", "euler_a", "dpm++_2m"] and use_karras:
        sampler_kwargs["use_karras"] = True

    # Time the sampling
    start_time = time.time()

    # Sample
    samples = sample(
        glide_model=model,
        glide_options=options,
        side_x=64,
        side_y=64,
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        device=device,
        prediction_respacing=str(num_steps),
        sampler_name=sampler_name,
        sampler_kwargs=sampler_kwargs,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # Convert to PIL images
    images = [tensor_to_pil(samples[i]) for i in range(samples.shape[0])]

    # Upsample with ESRGAN if requested
    upsampled_images = []
    if esrgan is not None:
        print_vram_usage("Before ESRGAN upsampling")
        for img in images:
            upsampled_img = esrgan.upsample_pil(img)
            upsampled_images.append(upsampled_img)
        print_vram_usage("After ESRGAN upsampling")
    else:
        upsampled_images = [None] * len(images)

    return images, upsampled_images, elapsed


def run_benchmark(
    model,
    options,
    prompt: str,
    samplers: List[str],
    num_steps: int,
    guidance_scale: float,
    device: str,
    output_dir: Path,
    use_karras: bool = False,
    esrgan: Optional[ESRGANUpsampler] = None,
) -> None:
    """Run benchmark comparing all samplers."""

    print(f"\n{'=' * 60}")
    print("BENCHMARK MODE")
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Device: {device}")
    print(f"Karras schedule: {use_karras}")
    print(f"{'=' * 60}\n")

    # Fixed seed for fair comparison
    seed = 42

    results = {}
    all_images = []

    for sampler_name in samplers:
        print(f"\nTesting {sampler_name}...")

        try:
            images, upsampled_images, elapsed = sample_single(
                model=model,
                options=options,
                prompt=prompt,
                sampler_name=sampler_name,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=device,
                seed=seed,
                use_karras=use_karras,
                esrgan=esrgan,
            )

            # Save individual image
            img_path = output_dir / f"{sampler_name}.png"
            images[0].save(img_path)

            # Save upsampled image if available
            if upsampled_images[0] is not None:
                upsampled_path = output_dir / f"{sampler_name}_esrgan.png"
                upsampled_images[0].save(upsampled_path)

            all_images.extend(images)
            results[sampler_name] = elapsed

            print(f"✓ {sampler_name}: {elapsed:.2f}s")

        except Exception as e:
            print(f"✗ {sampler_name} failed: {str(e)}")
            results[sampler_name] = None

    # Print summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")

    for sampler_name, elapsed in results.items():
        if elapsed is not None:
            print(f"{sampler_name:20s} {elapsed:8.2f}s")
        else:
            print(f"{sampler_name:20s}   FAILED")

    # Save comparison grid if we have results
    if all_images:
        # Pad to make square grid if needed
        n = len(all_images)
        grid_size = int(np.ceil(np.sqrt(n)))
        while len(all_images) < grid_size * grid_size:
            # Create blank image
            all_images.append(Image.new("RGB", all_images[0].size, (128, 128, 128)))

        grid = create_image_grid(all_images[: grid_size * grid_size])
        grid_path = output_dir / "benchmark_grid.png"
        grid.save(grid_path)
        print(f"\nSaved benchmark grid to: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample from GLIDE models")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to model checkpoint (empty for pretrained base model)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "upsample", "base-inpaint", "upsample-inpaint"],
        help="Type of model to load",
    )

    # Prompt arguments (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to use for generation",
    )
    prompt_group.add_argument(
        "--prompt_file",
        type=str,
        help="File containing prompts (must have 2, 4, 8, 16, or 32 lines)",
    )

    # Sampling arguments
    parser.add_argument(
        "--sampler",
        type=str,
        default="all",
        help="Sampler to use (plms, ddim, euler, euler_a, dpm++_2m, dpm++_2m_karras, all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--use_karras",
        action="store_true",
        help="Use Karras sigma schedule for applicable samplers",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/eval",
        help="Base directory for outputs",
    )

    # Special modes
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode comparing all samplers with fixed seed",
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
        help="Use FP16 precision",
    )

    # ESRGAN upsampling arguments
    parser.add_argument(
        "--use_esrgan",
        action="store_true",
        help="Use ESRGAN to upsample 64x64 images to 256x256",
    )
    parser.add_argument(
        "--esrgan_cache_dir",
        type=str,
        default="./esrgan_models",
        help="Directory to cache ESRGAN model weights",
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    print_vram_usage("Initial")

    model, _, options = load_model(
        glide_path=args.model_path,
        use_fp16=args.use_fp16,
        model_type=args.model_type,
    )
    model.eval()
    model = model.to(args.device)

    print_vram_usage("After loading GLIDE")

    # Initialize ESRGAN if requested
    esrgan = None
    if args.use_esrgan:
        print("\nLoading ESRGAN model...")
        esrgan = ESRGANUpsampler(device=args.device, cache_dir=args.esrgan_cache_dir)
        print_vram_usage("After loading ESRGAN")

    # Get prompts
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
    else:
        prompts = [args.prompt]

    # Get output directory
    output_base = Path(args.output_dir)
    output_dir = get_next_output_dir(output_base)
    print(f"Output directory: {output_dir}")

    # Get samplers to use
    if args.sampler == "all":
        # Use the fixed samplers that work with GLIDE
        samplers = ["plms", "ddim", "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras"]
    else:
        samplers = [args.sampler]

    # Run benchmark mode if requested
    if args.benchmark:
        if len(prompts) > 1:
            print("Warning: Benchmark mode uses only the first prompt")

        run_benchmark(
            model=model,
            options=options,
            prompt=prompts[0],
            samplers=samplers,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            device=args.device,
            output_dir=output_dir,
            use_karras=args.use_karras,
            esrgan=esrgan,
        )
        return

    # Normal sampling mode
    all_images = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt}")

        for sampler_name in samplers:
            print(f"  Sampling with {sampler_name}...")

            try:
                images, upsampled_images, elapsed = sample_single(
                    model=model,
                    options=options,
                    prompt=prompt,
                    sampler_name=sampler_name,
                    num_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_karras=args.use_karras,
                    esrgan=esrgan,
                )

                # Save individual images
                for batch_idx, (img, upsampled_img) in enumerate(
                    zip(images, upsampled_images)
                ):
                    if len(samplers) == 1 and args.batch_size == 1:
                        # Simple naming when single sampler and batch
                        filename = f"sample{prompt_idx + 1}.png"
                        upsampled_filename = f"sample{prompt_idx + 1}_esrgan.png"
                    else:
                        # More complex naming for multiple samplers/batches
                        filename = f"prompt{prompt_idx + 1}_{sampler_name}_batch{batch_idx + 1}.png"
                        upsampled_filename = f"prompt{prompt_idx + 1}_{sampler_name}_batch{batch_idx + 1}_esrgan.png"

                    img_path = output_dir / filename
                    img.save(img_path)
                    all_images.append(img)

                    # Save upsampled image if available
                    if upsampled_img is not None:
                        upsampled_path = output_dir / upsampled_filename
                        upsampled_img.save(upsampled_path)

                print(f"    ✓ Generated {len(images)} image(s) in {elapsed:.2f}s")

            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")

    # Create and save grid if we have multiple images
    if len(all_images) > 1 and len(all_images) in [4, 16, 64, 256]:
        print("\nCreating image grid...")
        grid = create_image_grid(all_images)
        grid_path = output_dir / "grid.png"
        grid.save(grid_path)
        print(f"Saved grid to: {grid_path}")
    elif len(all_images) > 1:
        print(
            f"\nNote: Generated {len(all_images)} images. Grid requires exactly 4, 16, 64, or 256 images."
        )

    print(f"\nAll outputs saved to: {output_dir}")

    # Final VRAM report
    print_vram_usage("Final")


if __name__ == "__main__":
    main()
