#!/usr/bin/env python3
"""
Generate super-resolution (256x256) images from base 64x64 images.
Reads base images and captions from data/images/base_64x64/
Saves upsampled images to data/images/sr_256x256/
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from glide_finetune.glide_util import (
    load_model,
    sample,
)
from glide_text2im.tokenizer.bpe import get_encoder


def load_base_images_and_captions(base_dir):
    """Load all base images and their corresponding captions."""
    # Find all PNG files
    image_files = sorted(glob.glob(os.path.join(base_dir, "*.png")))

    images_and_captions = []
    for img_path in image_files:
        # Get corresponding caption file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        caption_path = os.path.join(base_dir, f"{base_name}.txt")

        if not os.path.exists(caption_path):
            print(f"Warning: No caption found for {img_path}, skipping...")
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load caption
        with open(caption_path, "r") as f:
            caption = f.read().strip()

        images_and_captions.append({
            "image": img,
            "caption": caption,
            "base_name": base_name
        })

    return images_and_captions


def main():
    parser = argparse.ArgumentParser(description="Generate SR images from base images")
    parser.add_argument(
        "--sr_model_path",
        type=str,
        default="glide_model_cache/upsample.pt",
        help="Path to upsampler model checkpoint",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/images/base_64x64",
        help="Directory containing base 64x64 images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images/sr_256x256",
        help="Output directory for 256x256 upsampled images",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["plms", "ddim", "euler", "euler_a", "dpm"],
        help="Sampler to use for super-resolution",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=27,
        help="Number of sampling steps for SR (27 is optimal)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale for SR (0.0 for upsampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=420,
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
    parser.add_argument(
        "--upsample_temp",
        type=float,
        default=0.997,
        help="Temperature for upsampling (default from GLIDE)",
    )
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load base images and captions
    print(f"Loading base images from {args.base_dir}...")
    images_and_captions = load_base_images_and_captions(args.base_dir)
    print(f"Found {len(images_and_captions)} image-caption pairs")

    if len(images_and_captions) == 0:
        print("No images found! Please run generate_base_images.py first.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load upsampler model
    print(f"Loading upsampler model from {args.sr_model_path}...")
    precision = "bf16" if args.use_bf16 else "fp32"
    sr_model, sr_diffusion, sr_options = load_model(
        glide_path=args.sr_model_path,
        precision=precision,
        model_type="upsample",
        activation_checkpointing=False,
    )
    sr_model = sr_model.to(args.device)
    sr_model.eval()

    # Initialize tokenizer
    tokenizer = get_encoder()
    sr_options["tokenizer"] = tokenizer
    sr_options["device"] = args.device

    # Generate super-resolution images
    print(f"Generating {len(images_and_captions)} SR images with {args.sampler} sampler...")
    print(f"Guidance scale: {args.guidance_scale}, Steps: {args.sampler_steps}")
    print(f"Upsample temperature: {args.upsample_temp}")

    with torch.no_grad():
        pbar = tqdm(total=len(images_and_captions), desc="Upsampling images")

        for item in images_and_captions:
            base_image = item["image"]
            caption = item["caption"]
            base_name = item["base_name"]

            # Ensure base image is 64x64
            if base_image.size != (64, 64):
                base_image = base_image.resize((64, 64), Image.LANCZOS)

            # Convert base image to tensor [-1, 1]
            base_array = np.array(base_image).astype(np.float32) / 127.5 - 1
            base_tensor = torch.from_numpy(base_array).permute(2, 0, 1).unsqueeze(0)
            base_tensor = base_tensor.to(args.device)

            # Save the base image as a low-res reference (for debugging)
            temp_path = "/tmp/temp_base.png"
            base_image.save(temp_path)

            # Generate upsampled image
            # For upsampling, side_x and side_y refer to the INPUT size (64x64)
            # The model will output 256x256
            samples = sample(
                glide_model=sr_model,
                glide_options=sr_options,
                side_x=64,  # Input size
                side_y=64,  # Input size
                prompt=caption,
                batch_size=1,
                guidance_scale=args.guidance_scale,
                device=args.device,
                prediction_respacing=str(args.sampler_steps),
                upsample_enabled=True,
                image_to_upsample=temp_path,  # Pass path to base image
                upsample_temp=args.upsample_temp,
                sampler=args.sampler,
                sampler_eta=1.0 if args.sampler == "euler_a" else 0.0,
            )

            # Save the upsampled image
            sr_tensor = samples[0]  # Get first (and only) sample

            # Convert from [-1, 1] to [0, 255]
            sr_tensor = (sr_tensor + 1) * 127.5
            sr_tensor = sr_tensor.clamp(0, 255).to(torch.uint8)
            sr_array = sr_tensor.permute(1, 2, 0).cpu().numpy()

            # Save SR image with same basename
            sr_img = Image.fromarray(sr_array)
            sr_img_path = os.path.join(args.output_dir, f"{base_name}.png")
            sr_img.save(sr_img_path)

            # Save caption alongside (copy from base)
            sr_caption_path = os.path.join(args.output_dir, f"{base_name}.txt")
            with open(sr_caption_path, "w") as f:
                f.write(caption)

            pbar.update(1)

        pbar.close()

    print(f"\n✓ Generated {len(images_and_captions)} SR images")
    print(f"✓ Saved to {args.output_dir}/")

    # Create a summary file
    summary_path = os.path.join(args.output_dir, "sr_generation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Super-Resolution Generation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"SR Model: {args.sr_model_path}\n")
        f.write(f"Base images from: {args.base_dir}\n")
        f.write(f"Number of images: {len(images_and_captions)}\n")
        f.write(f"Output resolution: 256x256\n")
        f.write(f"Sampler: {args.sampler}\n")
        f.write(f"Steps: {args.sampler_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Upsample temperature: {args.upsample_temp}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Precision: {precision}\n")

    print(f"✓ Summary saved to {summary_path}")

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == "__main__":
    main()