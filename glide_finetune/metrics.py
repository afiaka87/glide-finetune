"""
Evaluation metrics for GLIDE fine-tuning.

Provides CLIP score (text-image alignment) and FID/KID (distribution quality)
computation during training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm

# Lazy imports to avoid loading heavy models at import time
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip(device="cpu"):
    """Lazily load OpenCLIP model. Cached across calls."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    _clip_model = _clip_model.to(device)
    return _clip_model, _clip_preprocess, _clip_tokenizer


def compute_clip_scores(
    images: List[Image.Image],
    prompts: List[str],
    device: str = "cuda",
) -> dict:
    """
    Compute CLIP cosine similarity between images and their prompts.

    Args:
        images: List of PIL images (any size, will be preprocessed)
        prompts: List of text prompts (same length as images)
        device: Device for CLIP inference

    Returns:
        Dict with clip_score_mean, clip_score_std, clip_scores (per-image)
    """
    if not images or not prompts:
        return {"clip_score_mean": 0.0, "clip_score_std": 0.0, "clip_scores": []}

    model, preprocess, tokenizer = _load_clip(device)

    # Preprocess images
    image_tensors = torch.stack([preprocess(img) for img in images]).to(device)

    # Tokenize text
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensors)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarity per pair
        scores = (image_features * text_features).sum(dim=-1).cpu().tolist()

    # Move CLIP back to CPU to free VRAM
    _clip_model.cpu()
    torch.cuda.empty_cache()

    return {
        "clip_score_mean": float(np.mean(scores)),
        "clip_score_std": float(np.std(scores)),
        "clip_scores": scores,
    }


def compute_fid_kid(
    glide_model,
    glide_diffusion,
    glide_options,
    eval_prompts: List[str],
    reference_stats_path: str,
    device: str = "cuda",
    num_samples: int = 500,
    batch_size: int = 1,
    guidance_scale: float = 4.0,
    sampler: str = "euler",
    sampler_steps: int = 30,
    side_x: int = 64,
    side_y: int = 64,
) -> dict:
    """
    Generate images and compute FID + KID against a pre-cached reference set.

    Args:
        glide_model: The GLIDE model (will be set to eval mode temporarily)
        glide_diffusion: The diffusion process
        glide_options: Model options dict
        eval_prompts: List of prompts to generate from (cycled if needed)
        reference_stats_path: Path to .pt file with cached reference Inception features
        device: Device for generation and metric computation
        num_samples: Number of images to generate
        batch_size: Batch size for generation (1 for CFG)
        guidance_scale: Classifier-free guidance scale
        sampler: Sampler to use
        sampler_steps: Number of diffusion steps
        side_x: Image width
        side_y: Image height

    Returns:
        Dict with fid, kid_mean, kid_std
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from glide_finetune.glide_util import sample
    from glide_finetune.train_util import pred_to_pil

    # Load reference stats
    ref_data = torch.load(reference_stats_path, map_location="cpu", weights_only=True)
    ref_images = ref_data["images"]  # [N, 3, 64, 64] uint8

    # Initialize metrics (use float64 for FID stability)
    kid_subset = min(100, num_samples, len(ref_images))
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.set_dtype(torch.float64)
    kid = KernelInceptionDistance(
        feature=2048, subset_size=kid_subset, normalize=True
    ).to(device)
    kid.set_dtype(torch.float64)

    # Feed reference images in batches
    ref_batch_size = 64
    for i in range(0, len(ref_images), ref_batch_size):
        batch = ref_images[i : i + ref_batch_size].to(device)
        # torchmetrics expects [N, 3, H, W] float [0, 1] when normalize=True
        batch_float = batch.float() / 255.0
        fid.update(batch_float, real=True)
        kid.update(batch_float, real=True)

    # Generate images and feed to metrics
    glide_model.eval()
    n_generated = 0

    pbar = tqdm(total=num_samples, desc="FID/KID eval", unit="img")
    while n_generated < num_samples:
        prompt_idx = n_generated % len(eval_prompts)
        prompt = eval_prompts[prompt_idx]

        samples = sample(
            glide_model=glide_model,
            glide_options=glide_options,
            side_x=side_x,
            side_y=side_y,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            device=device,
            prediction_respacing=str(sampler_steps),
            sampler=sampler,
        )

        # samples is [B, 3, H, W] in [-1, 1] — convert to [0, 1]
        samples_float = (samples.clamp(-1, 1) + 1) / 2.0
        fid.update(samples_float.to(device), real=False)
        kid.update(samples_float.to(device), real=False)

        n_generated += samples.shape[0]
        pbar.update(samples.shape[0])

    pbar.close()
    glide_model.train()

    # Compute metrics
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()

    # Cleanup
    del fid, kid
    torch.cuda.empty_cache()

    return {
        "fid": fid_score,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
    }
