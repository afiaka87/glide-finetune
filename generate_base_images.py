#!/usr/bin/env python3
"""
Generate 32 base model images for super-resolution training evaluation.
These images will be used as fixed inputs during SR model training.
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from glide_finetune.glide_util import (
    load_model,
    sample,
    get_tokens_and_mask,
)
from glide_text2im.tokenizer.bpe import get_encoder


def main():
    parser = argparse.ArgumentParser(description="Generate base model images for SR evaluation")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="glide_model_cache/base.pt",
        help="Path to base model checkpoint",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="data/generated-captions-1k.txt",
        help="File containing prompts (will use first 32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images/base_64x64",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=32,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["plms", "ddim", "euler", "euler_a", "dpm"],
        help="Sampler to use",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=50,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 precision",
    )
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    with open(args.prompt_file, "r") as f:
        lines = f.readlines()

    # Skip the first line (header) and get actual prompts
    all_prompts = [line.strip() for line in lines[1:] if line.strip()]

    # Use first N prompts
    prompts = all_prompts[:args.num_images]
    print(f"Using {len(prompts)} prompts for generation")

    # Print first few prompts for verification
    print("\nFirst 3 prompts:")
    for i, prompt in enumerate(prompts[:3], 1):
        print(f"  {i}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load model
    print(f"Loading base model from {args.base_model_path}...")
    precision = "bf16" if args.use_bf16 else "fp32"
    model, diffusion, options = load_model(
        glide_path=args.base_model_path,
        precision=precision,
        model_type="base",
        activation_checkpointing=False,
    )
    model = model.to(args.device)
    model.eval()

    # Initialize tokenizer
    tokenizer = get_encoder()
    options["tokenizer"] = tokenizer

    # Update options for sampling
    options["device"] = args.device

    # Generate images in batches
    print(f"Generating {args.num_images} images with {args.sampler} sampler...")
    print(f"Guidance scale: {args.guidance_scale}, Steps: {args.sampler_steps}")

    all_images = []
    prompt_idx = 0

    with torch.no_grad():
        pbar = tqdm(total=args.num_images, desc="Generating images")

        for idx, prompt in enumerate(prompts):
            # Generate samples with classifier-free guidance
            # The sample function handles tokenization internally
            samples = sample(
                glide_model=model,
                glide_options=options,
                side_x=64,
                side_y=64,
                prompt=prompt,
                batch_size=1,
                guidance_scale=args.guidance_scale,
                device=args.device,
                prediction_respacing=str(args.sampler_steps),
                upsample_enabled=False,
                sampler=args.sampler,
                sampler_eta=1.0 if args.sampler == "euler_a" else 0.0,
            )

            # Save the generated image
            img_tensor = samples[0]  # Get first (and only) sample

            # Convert from [-1, 1] to [0, 255]
            img_tensor = (img_tensor + 1) * 127.5
            img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)
            img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

            # Save image and caption with MS COCO style naming (00001.png, 00001.txt)
            img = Image.fromarray(img_array)
            base_name = f"{idx+1:05d}"  # Start from 00001, not 00000
            img_path = os.path.join(args.output_dir, f"{base_name}.png")
            img.save(img_path)

            # Save caption alongside with same basename
            caption_path = os.path.join(args.output_dir, f"{base_name}.txt")
            with open(caption_path, "w") as f:
                f.write(prompt)

            all_images.append(img_path)
            pbar.update(1)

        pbar.close()

    print(f"\n✓ Generated {len(all_images)} images")
    print(f"✓ Saved to {args.output_dir}/")

    # Create a summary file
    summary_path = os.path.join(args.output_dir, "generation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Base Image Generation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {args.base_model_path}\n")
        f.write(f"Number of images: {len(all_images)}\n")
        f.write(f"Sampler: {args.sampler}\n")
        f.write(f"Steps: {args.sampler_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Precision: {precision}\n")

    print(f"✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()