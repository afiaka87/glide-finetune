"""
Prepare reference image statistics for FID/KID evaluation.

Extracts human-related images from a WebDataset (e.g. datacomp-small),
resizes to 64x64, and caches them as a tensor for use during training eval.

Usage:
    uv run python prepare_reference_stats.py \
        --data_dir ~/Data/datacomp-small/shards \
        --output reference_stats_humans_64.pt \
        --num_images 5000
"""

import argparse
import io
import os
import re
from glob import glob

import PIL.Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

# Keywords that indicate humans in captions
HUMAN_KEYWORDS = re.compile(
    r"\b(person|people|man|woman|boy|girl|child|children|kid|portrait|face|"
    r"human|crowd|family|couple|baby|teenager|adult|elderly|worker|student|"
    r"athlete|dancer|musician|chef|doctor|nurse|teacher|player|singer|actor)\b",
    re.IGNORECASE,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reference stats for FID/KID evaluation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to WebDataset shards directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reference_stats_humans_64.pt",
        help="Output path for cached reference images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5000,
        help="Target number of reference images to extract",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Resize images to this resolution",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="jpg",
        help="Key for images in tar files",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="txt",
        help="Key for captions in tar files",
    )
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)

    # Find tar files
    if os.path.isdir(data_dir):
        tar_files = sorted(glob(os.path.join(data_dir, "*.tar")))
    else:
        tar_files = sorted(glob(data_dir))

    if not tar_files:
        raise ValueError(f"No tar files found at {data_dir}")

    print(f"Found {len(tar_files)} tar files")
    print(f"Target: {args.num_images} human images at {args.image_size}x{args.image_size}")

    import webdataset as wds

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),  # [0, 1]
    ])

    images_collected = []
    captions_collected = []

    dataset = wds.WebDataset(tar_files, handler=wds.handlers.warn_and_continue)

    pbar = tqdm(total=args.num_images, desc="Collecting human images", unit="img")

    for sample in dataset:
        if len(images_collected) >= args.num_images:
            break

        # Check for required keys
        if args.image_key not in sample or args.caption_key not in sample:
            continue

        caption = sample[args.caption_key].decode("utf-8", errors="ignore")

        # Filter for human-related captions
        if not HUMAN_KEYWORDS.search(caption):
            continue

        # Load and process image
        try:
            img = PIL.Image.open(io.BytesIO(sample[args.image_key])).convert("RGB")
            img_tensor = transform(img)  # [3, H, W] float [0, 1]
            # Convert to uint8 for compact storage
            img_uint8 = (img_tensor * 255).to(torch.uint8)
            images_collected.append(img_uint8)
            captions_collected.append(caption)
            pbar.update(1)
        except Exception:
            continue

    pbar.close()

    if not images_collected:
        raise ValueError("No human images found! Check your data and keyword filters.")

    # Stack into tensor
    images_tensor = torch.stack(images_collected)  # [N, 3, H, W] uint8
    print(f"\nCollected {len(images_collected)} human images")
    print(f"Tensor shape: {images_tensor.shape}")
    print(f"Storage size: {images_tensor.nbytes / 1024 / 1024:.1f} MB")

    # Save
    torch.save(
        {
            "images": images_tensor,
            "num_images": len(images_collected),
            "image_size": args.image_size,
            "captions_sample": captions_collected[:10],
        },
        args.output,
    )
    print(f"Saved to {args.output}")
    print(f"\nSample captions:")
    for c in captions_collected[:5]:
        print(f"  - {c[:100]}")


if __name__ == "__main__":
    main()
