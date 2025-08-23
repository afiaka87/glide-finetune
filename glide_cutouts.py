"""
CLIP-guided cutouts for GLIDE diffusion models.
Implements the "make cutouts" algorithm from early CLIP days on top of GLIDE.

Clean version with:
- Basic MakeCutouts algorithm (no augmentations)
- Single noise-aware CLIP model (no ensemble)
- Enhanced samplers (Euler, Euler-A, DPM++)
- GIF creation with phase labeling
- No torch.compile or caching
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import torchvision.transforms as transforms
from glide_finetune.enhanced_samplers import enhance_glide_diffusion
import glob
import os


class MakeCutouts(nn.Module):
    """
    Generate multiple random square crops (cutouts) from images for CLIP guidance.
    Based on the classic "make cutouts" algorithm from early CLIP experiments.
    """

    def __init__(
        self,
        cut_size: int = 224,
        cutn: int = 32,
        cut_pow: float = 1.0,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        
        # Register CLIP normalization constants as buffers for stability
        self.register_buffer("clip_mean", th.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", th.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(
        self, input_tensor: th.Tensor, timestep: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        Generate random cutouts from input tensor.

        Args:
            input_tensor: [batch_size, channels, height, width]
            timestep: Current timestep (optional, for timestep-aware sizing)

        Returns:
            cutouts: [batch_size * cutn, channels, cut_size, cut_size]
        """
        batch_size, channels, height, width = input_tensor.shape
        min_size = min(height, width)
        cutouts = []

        # Timestep-aware cutout sizing
        if timestep is not None:
            # Normalize timestep to [0, 1] where 1 is early (noisy), 0 is late (clean)  
            t_norm = (timestep[0].float() / 1000.0).clamp(0, 1)
            # Cosine schedule: smoother size progression, reduces jitter
            size_multiplier = 0.25 + 0.75 * 0.5 * (1 - th.cos(th.pi * t_norm))  # Range [0.25, 1.0]

        # Create deterministic generator for reproducible cutouts
        gen = th.Generator(device=input_tensor.device)
        if timestep is not None:
            gen.manual_seed(42 + int(timestep[0].item()))  # Deterministic per timestep
        
        for i in range(self.cutn):
            # Multi-scale approach: mix different cutout sizes for better guidance
            if i < self.cutn // 3:
                size_factor = 0.8 * size_multiplier  # Large cutouts
            elif i < 2 * self.cutn // 3:
                size_factor = 0.6 * size_multiplier  # Medium cutouts
            else:
                size_factor = 0.4 * size_multiplier  # Small cutouts

            # Calculate target size based on cutout size and scale factors
            target_size = int(self.cut_size * size_factor)
            # Apply power distribution for randomness with deterministic generator
            random_factor = th.rand([], generator=gen, device=input_tensor.device) ** self.cut_pow
            size = int(target_size * (0.5 + 0.5 * random_factor))  # Range: 50%-100% of target
            
            # Ensure size is within reasonable bounds
            size = max(16, min(size, min(height, width)))

            # Random position - ensure we don't go out of bounds
            max_offsetx = max(0, width - size)
            max_offsety = max(0, height - size)

            offsetx = (
                th.randint(0, max_offsetx + 1, (), generator=gen, device=input_tensor.device) 
                if max_offsetx > 0 else th.tensor(0, device=input_tensor.device)
            )
            offsety = (
                th.randint(0, max_offsety + 1, (), generator=gen, device=input_tensor.device)
                if max_offsety > 0 else th.tensor(0, device=input_tensor.device)
            )

            # Extract cutout
            cutout = input_tensor[
                :, :, offsety : offsety + size, offsetx : offsetx + size
            ]

            # Resize to target size
            cutout = F.interpolate(
                cutout,
                size=(self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )

            cutouts.append(cutout)

        # Stack all cutouts
        return th.cat(cutouts, dim=0)


class CLIPGuidedCutouts:
    """
    CLIP guidance using cutouts for GLIDE diffusion sampling.
    """

    def __init__(
        self,
        clip_model,
        cutout_size: int = 64,
        num_cutouts: int = 32,
        cutout_power: float = 1.0,
        guidance_scale: float = 5.0,
    ):
        self.clip_model = clip_model
        self.make_cutouts = MakeCutouts(cutout_size, num_cutouts, cutout_power)
        self.guidance_scale = guidance_scale
        
        # Text embedding cache for performance
        self._text_cache = {}

        # CLIP preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    
    def _get_text_embeddings_cached(self, prompts):
        """Get text embeddings with caching for performance."""
        key = tuple(prompts) if isinstance(prompts, (list, tuple)) else (prompts,)
        if key not in self._text_cache:
            with th.no_grad():
                with th.amp.autocast('cuda', enabled=False):
                    self._text_cache[key] = self.clip_model.text_embeddings(prompts)
        return self._text_cache[key]

    def cond_fn(self, prompts: List[str]):
        """
        Create a conditioning function for diffusion sampling.

        Args:
            prompts: List of text prompts for guidance

        Returns:
            Conditioning function compatible with GLIDE's p_sample_loop
        """
        # Encode text prompts using cached method with proper normalization
        target_embeddings = self._get_text_embeddings_cached(prompts)
        target_embeddings = th.nn.functional.normalize(target_embeddings, dim=-1, eps=1e-6)

        def _cond_fn(x, t, **kwargs):
            """
            Conditioning function called during diffusion sampling.

            Args:
                x: Current noisy image [batch_size, 3, height, width]
                t: Current timestep

            Returns:
                Gradient for guidance
            """
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)

                # Convert from [-1, 1] to [0, 1] for CLIP
                images = (x_in + 1) / 2
                images = th.clamp(images, 0, 1)

                # Generate cutouts with timestep awareness
                cutouts = self.make_cutouts(images, t)

                # Normalize for CLIP
                cutouts = self.normalize(cutouts)

                # Use actual timestep from diffusion process
                cutout_timesteps = t[0].repeat(cutouts.shape[0])

                # Denormalize cutouts back to [-1, 1] range for CLIP's image_embeddings  
                # Use registered buffers for stability
                denormalized = cutouts * self.make_cutouts.clip_std + self.make_cutouts.clip_mean  # Back to [0,1]
                denormalized = denormalized * 2.0 - 1.0  # Convert to [-1,1] for vendor CLIP
                
                # Safety assertions
                th._assert(denormalized.dtype == th.float32, "CLIP expects float32 for stability")
                th._assert((denormalized >= -1.1).all() and (denormalized <= 1.1).all(), "CLIP inputs should be in [-1,1] range")

                # Enforce FP32 for CLIP under AMP for numerical stability
                with th.amp.autocast('cuda', enabled=False):
                    image_embeddings = self.clip_model.image_embeddings(
                        denormalized.float(), cutout_timesteps
                    )

                # Compute similarity loss
                batch_size = x.shape[0]

                # Reshape embeddings to match batch structure
                image_embeddings = image_embeddings.view(
                    batch_size, self.make_cutouts.cutn, -1
                )
                target_embeddings_expanded = target_embeddings.unsqueeze(1).expand(
                    -1, self.make_cutouts.cutn, -1
                )

                # Normalize embeddings explicitly for stable cosine similarity
                image_embeddings_norm = th.nn.functional.normalize(image_embeddings, dim=-1, eps=1e-6)
                target_embeddings_norm = th.nn.functional.normalize(target_embeddings_expanded, dim=-1, eps=1e-6)
                
                # Compute cosine similarity using normalized dot product (more stable)
                similarities = (image_embeddings_norm * target_embeddings_norm).sum(dim=-1)

                # Average similarity across cutouts (negative for loss minimization)
                loss = -similarities.mean()

                # Compute gradients with safety checks
                grad = th.autograd.grad(loss, x_in, create_graph=False)[0]
                
                # Safety check for NaN or infinite gradients
                if th.isnan(grad).any() or th.isinf(grad).any():
                    return th.zeros_like(x_in)

                # Adaptive guidance based on timestep (stronger early, weaker later)
                max_timestep = 1000.0
                timestep_normalized = t[0].float() / max_timestep

                # Adaptive guidance scale: stronger early, weaker later (like working version)
                adaptive_scale = self.guidance_scale * (0.1 + 0.9 * timestep_normalized)

                # Clamp gradient magnitude for stability
                final_grad = grad * adaptive_scale
                grad_norm = th.norm(final_grad)
                if grad_norm > 1.0:  # Prevent explosion
                    final_grad = final_grad / grad_norm

                return final_grad

        return _cond_fn


class CLIPGuidedSuperResCutouts:
    """
    CLIP guidance using cutouts for GLIDE super-resolution upsampling.
    Focus on detail enhancement and consistency with low-res input.
    """

    def __init__(
        self,
        clip_model,
        cutout_size: int = 64,
        num_cutouts: int = 16,
        cutout_power: float = 0.8,
        guidance_scale: float = 2.0,
    ):
        self.clip_model = clip_model
        self.make_cutouts = MakeCutouts(cutout_size, num_cutouts, cutout_power)
        self.guidance_scale = guidance_scale
        
        # Text embedding cache for performance
        self._text_cache = {}

        # CLIP preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    
    def _get_text_embeddings_cached(self, prompts):
        """Get text embeddings with caching for performance."""
        key = tuple(prompts) if isinstance(prompts, (list, tuple)) else (prompts,)
        if key not in self._text_cache:
            with th.no_grad():
                with th.amp.autocast('cuda', enabled=False):
                    self._text_cache[key] = self.clip_model.text_embeddings(prompts)
        return self._text_cache[key]

    def cond_fn(self, prompts: List[str], low_res_image: th.Tensor):
        """
        Create a conditioning function for super-resolution diffusion sampling.

        Args:
            prompts: List of text prompts for guidance
            low_res_image: Low resolution input image [batch_size, 3, 64, 64]

        Returns:
            Conditioning function compatible with GLIDE's upsampler
        """
        # Encode text prompts using cached method with proper normalization
        target_embeddings = self._get_text_embeddings_cached(prompts)
        target_embeddings = th.nn.functional.normalize(target_embeddings, dim=-1, eps=1e-6)

        # Encode low-res image for consistency guidance
        low_res_normalized = self.normalize((low_res_image + 1) / 2)

        # Denormalize for CLIP image embeddings using buffers for consistency
        low_res_denorm = low_res_normalized * self.make_cutouts.clip_std + self.make_cutouts.clip_mean
        low_res_denorm = low_res_denorm * 2.0 - 1.0

        # Get low-res CLIP embeddings for consistency
        dummy_timestep = th.zeros(
            low_res_denorm.shape[0], device=low_res_image.device, dtype=th.long
        )
        low_res_clip_emb = self.clip_model.image_embeddings(
            low_res_denorm, dummy_timestep
        )

        def _cond_fn(x, t, **kwargs):
            """
            Conditioning function for super-resolution with dual guidance:
            1. Text prompt guidance (what to generate)
            2. Low-res consistency guidance (maintain structure)
            """
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)

                # Convert from [-1, 1] to [0, 1] for CLIP
                images = (x_in + 1) / 2
                images = th.clamp(images, 0, 1)

                # Generate cutouts with timestep awareness
                cutouts = self.make_cutouts(images, t)

                # Normalize for CLIP
                cutouts = self.normalize(cutouts)

                # Use actual timesteps for noise-aware CLIP
                cutout_timesteps = t[0].repeat(cutouts.shape[0])

                # Denormalize for CLIP's image_embeddings
                mean = th.tensor(
                    [0.48145466, 0.4578275, 0.40821073], device=cutouts.device
                ).view(1, 3, 1, 1)
                std = th.tensor(
                    [0.26862954, 0.26130258, 0.27577711], device=cutouts.device
                ).view(1, 3, 1, 1)
                denormalized = cutouts * std + mean  # Back to [0,1]
                denormalized = denormalized * 2.0 - 1.0  # Convert to [-1,1] for vendor CLIP

                # Get CLIP embeddings for current cutouts
                image_embeddings = self.clip_model.image_embeddings(
                    denormalized, cutout_timesteps
                )

                # Compute similarity loss
                batch_size = x.shape[0]

                # Reshape embeddings to match batch structure
                image_embeddings = image_embeddings.view(
                    batch_size, self.make_cutouts.cutn, -1
                )
                target_embeddings_expanded = target_embeddings.unsqueeze(1).expand(
                    -1, self.make_cutouts.cutn, -1
                )

                # Text guidance: Compute cosine similarity for each cutout
                text_similarities = F.cosine_similarity(
                    image_embeddings, target_embeddings_expanded, dim=-1
                )
                text_loss = -text_similarities.mean()

                # Consistency guidance: Ensure SR maintains low-res structure
                # Downsample 256x256 SR to 64x64 for CLIP comparison with low-res
                images_64 = F.interpolate(
                    images, size=(64, 64), mode="bilinear", align_corners=False
                )
                full_image_norm = self.normalize(images_64)
                full_image_denorm = full_image_norm * std + mean
                full_image_denorm = full_image_denorm * 2.0 - 1.0

                # Use timestep for full image embedding
                full_timestep = t[0].repeat(batch_size)
                full_image_emb = self.clip_model.image_embeddings(
                    full_image_denorm, full_timestep
                )

                # Consistency loss: SR should be similar to low-res structure
                consistency_similarities = F.cosine_similarity(
                    full_image_emb, low_res_clip_emb, dim=-1
                )
                consistency_loss = -(consistency_similarities.mean() * 0.3)

                # Combined loss
                total_loss = text_loss + consistency_loss

                # Compute gradients with safety checks
                grad = th.autograd.grad(total_loss, x_in, create_graph=False)[0]
                
                # Safety check for NaN or infinite gradients
                if th.isnan(grad).any() or th.isinf(grad).any():
                    return th.zeros_like(x_in)

                # Adaptive guidance for SR (gentler than base model)
                max_timestep = 1000.0
                timestep_normalized = t[0].float() / max_timestep
                # SR guidance: moderate early, gentle later (like working version)
                adaptive_scale = self.guidance_scale * (0.2 + 0.8 * timestep_normalized)

                # Clamp gradient magnitude for stability
                final_grad = grad * adaptive_scale
                grad_norm = th.norm(final_grad)
                if grad_norm > 0.5:  # Gentler clamping for SR
                    final_grad = final_grad / grad_norm * 0.5

                return final_grad

        return _cond_fn


# Default shape for GLIDE base model: batch_size=1, channels=3 (RGB), height=64, width=64
DEFAULT_BASE_SHAPE = (1, 3, 64, 64)


def save_frame(tensor, step, stage, frame_dir):
    """Save a single frame from tensor."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    from torchvision.utils import save_image

    # Convert from [-1, 1] to [0, 1]
    normalized = (tensor + 1) * 0.5
    filename = f"{stage}_step_{step:03d}.png"
    save_image(normalized, os.path.join(frame_dir, filename))


def sample_with_universal_frame_saving(
    sampler_fn,
    sampler_kwargs,
    num_steps,
    frame_dir,
    stage_name,
    save_frames=True,
):
    """
    Universal wrapper that adds frame saving to any GLIDE sampler.
    
    For samplers without built-in frame saving, creates representative frames
    by sampling the generation process at regular intervals.
    """
    if not save_frames:
        return sampler_fn(**sampler_kwargs)
    
    # Save initial noise frame
    shape = sampler_kwargs.get('shape', DEFAULT_BASE_SHAPE)
    device = sampler_kwargs.get('device', 'cuda')
    
    # Create initial noise and save it
    initial_noise = th.randn(*shape, device=device)
    save_frame(initial_noise, 0, stage_name, frame_dir)
    
    # Run the actual sampling
    # Note: **sampler_kwargs expands keyword arguments generically - this works for most
    # GLIDE samplers but isn't robust to all possible sampler signatures
    samples = sampler_fn(**sampler_kwargs)
    
    # For samplers without built-in frame saving, create representative frames
    # by interpolating between noise and final result
    frame_interval = max(1, num_steps // 20)  # ~20 frames total
    
    for i in range(1, num_steps + 1):
        if i % frame_interval == 0 or i == num_steps:
            # Create interpolated frame between noise and result
            alpha = i / num_steps
            interpolated = (1 - alpha) * initial_noise + alpha * samples
            save_frame(interpolated, i, stage_name, frame_dir)
    
    return samples


def create_gif_from_frames(frame_dir, output_path, duration=100, base_steps=30, sr_steps=27):
    """Create end-to-end GIF showing complete generation process.

    Shows base generation (upscaled with bicubic) → super-resolution refinement.
    Creates a seamless transition from coarse to refined detail.
    
    Args:
        frame_dir: Directory containing frame images
        output_path: Output path for the GIF
        duration: Base frame duration in milliseconds
        base_steps: Number of base generation steps (for duration scaling)
        sr_steps: Number of super-resolution steps (for duration scaling)
    """
    from PIL import Image, ImageDraw, ImageFont

    # Get all frame files
    base_frames = sorted(glob.glob(os.path.join(frame_dir, "base_step_*.png")))
    sr_frames = sorted(glob.glob(os.path.join(frame_dir, "sr_step_*.png")))

    if not base_frames and not sr_frames:
        print("No frames found to create GIF")
        return

    print(
        f"🎬 Creating end-to-end GIF: {len(base_frames)} base + {len(sr_frames)} SR frames"
    )

    # Determine target size (use SR size if available, otherwise scale base)
    target_size = (256, 256)  # Default SR size
    if sr_frames:
        sr_img = Image.open(sr_frames[0])
        target_size = sr_img.size

    images = []

    # Phase 1: Base generation frames (upscaled with bicubic)
    print("📐 Processing base frames (bicubic upscaling)...")
    for i, frame_path in enumerate(base_frames):
        img = Image.open(frame_path)

        # Upscale base frames using bicubic (LANCZOS) for smooth interpolation
        upscaled = img.resize(target_size, Image.Resampling.LANCZOS)

        # Add subtle label for first and last base frames
        if i == 0:
            labeled_img = add_phase_label(
                upscaled, "Base Generation (64x64→256x256)", (10, 10)
            )
            images.append(labeled_img)
        elif i == len(base_frames) - 1:
            labeled_img = add_phase_label(
                upscaled, "Base Complete", (10, 10), bg_color=(0, 0, 100, 120)
            )
            images.append(labeled_img)
        else:
            images.append(upscaled)

    # Transition pause with label
    if base_frames and sr_frames:
        print("⏸️  Adding transition pause...")
        transition_img = add_phase_label(
            images[-1],
            "Starting Super-Resolution...",
            (10, 10),
            bg_color=(100, 0, 100, 120),
        )
        for _ in range(8):  # Longer pause to emphasize transition
            images.append(transition_img)

    # Phase 2: Super-resolution frames (native 256x256)
    print("🔍 Processing super-resolution frames...")
    for i, frame_path in enumerate(sr_frames):
        img = Image.open(frame_path)

        # Ensure SR frames are correct size
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Add label for first and last SR frames
        if i == 0:
            labeled_img = add_phase_label(
                img, "Super-Resolution (256x256 native)", (10, 10)
            )
            images.append(labeled_img)
        elif i == len(sr_frames) - 1:
            labeled_img = add_phase_label(
                img, "Final Result", (10, 10), bg_color=(0, 100, 0, 120)
            )
            images.append(labeled_img)
        else:
            images.append(img)

    # Final result pause (longer to appreciate the details)
    if images:
        print("🎯 Adding final result pause...")
        for _ in range(15):  # Hold final result longer
            images.append(images[-1])

    # Save GIF with optimized settings
    if images:
        print(f"💾 Saving GIF with {len(images)} total frames...")

        # Use different durations for different phases
        durations = []
        base_count = len(base_frames)
        transition_count = 8 if base_frames and sr_frames else 0
        sr_count = len(sr_frames)
        final_pause = 15

        # Scale frame durations based on step count for consistent total viewing time
        # More steps = faster frames to keep reasonable GIF length
        base_duration_scale = max(0.5, min(2.0, 30 / max(base_steps, 1)))
        sr_duration_scale = max(0.5, min(2.0, 27 / max(sr_steps, 1)))
        
        # Base frames: scaled by step count
        durations.extend([int(duration * base_duration_scale)] * base_count)
        # Transition: slower
        durations.extend([duration * 2] * transition_count)
        # SR frames: scaled by step count, slightly slower to see detail improvement
        durations.extend([int(duration * 1.5 * sr_duration_scale)] * sr_count)
        # Final pause: much slower
        durations.extend([duration * 3] * final_pause)

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
        print(f"✨ End-to-end GIF saved to {output_path}")


def add_phase_label(
    img, text, position=(10, 10), bg_color=(0, 0, 0, 100), text_color=(255, 255, 255)
):
    """Add a semi-transparent label to an image."""
    from PIL import Image, ImageDraw, ImageFont

    # Create a copy to avoid modifying original
    labeled = img.copy().convert("RGBA")

    # Create overlay for text background
    overlay = Image.new("RGBA", labeled.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to use a nice font, fall back to default
    try:
        # Use a reasonable font size based on image size
        font_size = max(12, min(20, labeled.size[0] // 20))
        # Try multiple font paths for cross-platform compatibility
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/TTF/arial.ttf",  # Arch Linux
            "/usr/share/fonts/arial.ttf",  # Other Linux
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        if font is None:
            raise Exception("No fonts found")
    except:
        try:
            font = ImageFont.load_default()
        except:
            # If all else fails, skip text rendering
            return img

    # Get text size
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Fallback for older PIL versions
        text_width, text_height = draw.textsize(text, font=font)

    # Draw background rectangle
    padding = 6
    bg_rect = [
        position[0] - padding,
        position[1] - padding,
        position[0] + text_width + padding,
        position[1] + text_height + padding,
    ]
    draw.rectangle(bg_rect, fill=bg_color)

    # Draw text
    draw.text(position, text, fill=text_color, font=font)

    # Composite with original image
    result = Image.alpha_composite(labeled, overlay)
    return result.convert("RGB")


def euler_sample_loop_with_frames(
    diffusion,
    model,
    shape,
    cond_fn,
    model_kwargs,
    device,
    frame_dir,
    stage_name,
    progress=True,
    num_steps=30,
    noise=None,
):
    """Euler sampling with frame saving."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # Initialize noise (use provided noise or generate random)
    img = noise if noise is not None else th.randn(*shape, device=device)
    save_frame(img, 0, stage_name, frame_dir)

    # Get timesteps using the specified num_steps
    total_timesteps = diffusion.num_timesteps
    if num_steps >= total_timesteps:
        indices = list(range(total_timesteps))[::-1]
    else:
        # Create evenly spaced timesteps
        step_size = total_timesteps // num_steps
        indices = [total_timesteps - 1 - i * step_size for i in range(num_steps)]
        indices = sorted(indices, reverse=True)

    from tqdm.auto import tqdm

    indices_iter = (
        tqdm(indices, desc=f"{stage_name.title()} sampling") if progress else indices
    )

    for i, timestep_idx in enumerate(indices_iter):
        t = th.tensor([timestep_idx] * shape[0], device=device)

        with th.no_grad():
            # Use GLIDE's p_mean_variance to get proper x_start prediction
            out = diffusion.p_mean_variance(
                model, img, t, clip_denoised=True, model_kwargs=model_kwargs or {}
            )

            # Apply classifier guidance if specified
            if cond_fn is not None:
                out = diffusion.condition_score(
                    cond_fn, out, img, t, model_kwargs=model_kwargs or {}
                )

            # Get epsilon from x_start prediction
            epsilon = diffusion._predict_eps_from_xstart(img, t, out["pred_xstart"])

            # Extract alpha values
            from glide_text2im.gaussian_diffusion import _extract_into_tensor

            alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, img.shape)
            alpha_bar_prev = _extract_into_tensor(
                diffusion.alphas_cumprod_prev, t, img.shape
            )

            # Euler step (deterministic)
            mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev) * epsilon
            )

            img = mean_pred

            # Save frame every few steps
            if i % max(1, len(indices) // 20) == 0:  # Save ~20 frames total
                save_frame(img, i + 1, stage_name, frame_dir)

    # Save final frame
    save_frame(img, len(indices), stage_name, frame_dir)
    return img


def sample_with_cutouts(
    model,
    diffusion,
    clip_model,
    prompts: List[str],
    batch_size: int = 1,
    image_size: int = 64,
    cutout_size: int = 64,
    num_cutouts: int = 32,
    cutout_power: float = 1.0,
    guidance_scale: float = 5.0,
    sampler: str = "ddim",
    num_steps: int = 30,
    save_frames: bool = False,
    frame_dir: str = "frames",
    device=None,
    **model_kwargs,
):
    """
    Sample from GLIDE using CLIP-guided cutouts.

    Args:
        model: GLIDE model
        diffusion: Diffusion process
        clip_model: CLIP model for guidance
        prompts: Text prompts for generation
        batch_size: Number of images to generate
        image_size: Output image size
        cutout_size: Size of individual cutouts for CLIP
        num_cutouts: Number of cutouts per image
        cutout_power: Power for cutout size distribution
        guidance_scale: Strength of CLIP guidance
        sampler: Sampling method ("ddim", "euler", "euler_ancestral", "dpm")
        num_steps: Number of diffusion steps
        device: Device to run on
        **model_kwargs: Additional arguments for model

    Returns:
        Generated images tensor
    """

    # Create cutout guidance
    cutout_guidance = CLIPGuidedCutouts(
        clip_model,
        cutout_size,
        num_cutouts,
        cutout_power,
        guidance_scale,
    )
    # Move to correct device to ensure buffers are on GPU
    cutout_guidance.make_cutouts.to(device)

    # Create conditioning function
    cond_fn = cutout_guidance.cond_fn(prompts)

    # Enhance diffusion with additional samplers
    diffusion = enhance_glide_diffusion(diffusion)

    # Sample with guidance using specified sampler
    if sampler == "euler":
        if save_frames:
            samples = euler_sample_loop_with_frames(
                diffusion=diffusion,
                model=model,
                shape=(batch_size, 3, image_size, image_size),
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                frame_dir=frame_dir,
                stage_name="base",
                progress=True,
                num_steps=num_steps,
            )
        else:
            samples = diffusion.euler_sample_loop(
                model,
                (batch_size, 3, image_size, image_size),
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                num_steps=num_steps,
            )
    elif sampler == "euler_ancestral":
        # Use universal frame saving wrapper for euler_ancestral
        samples = sample_with_universal_frame_saving(
            diffusion.euler_ancestral_sample_loop,
            {
                'model': model,
                'shape': (batch_size, 3, image_size, image_size),
                'device': device,
                'clip_denoised': True,
                'progress': True,
                'model_kwargs': model_kwargs,
                'cond_fn': cond_fn,
                'num_steps': num_steps,
            },
            num_steps,
            frame_dir if save_frames else None,
            "base",
            save_frames
        )
    elif sampler == "dpm":
        samples = diffusion.dpm_solver_sample_loop(
            model,
            (batch_size, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            num_steps=num_steps,
        )
    else:  # Default to DDIM
        samples = diffusion.p_sample_loop(
            model,
            (batch_size, 3, image_size, image_size),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )

    return samples


def sample_sr_with_cutouts(
    model_up,
    diffusion_up,
    clip_model,
    prompts: List[str],
    low_res_samples: th.Tensor,
    batch_size: int = 1,
    cutout_size: int = 64,
    num_cutouts: int = 32,
    cutout_power: float = 0.8,
    guidance_scale: float = 2.0,
    sampler: str = "plms",
    num_steps: int = 27,
    upsample_temp: float = 0.997,
    save_frames: bool = False,
    frame_dir: str = "frames",
    device=None,
    **model_kwargs,
):
    """
    Super-resolution sampling with CLIP-guided cutouts.
    Always uses GLIDE's default PLMS sampler with fast27 timestep respacing.

    Args:
        model_up: GLIDE upsampler model
        diffusion_up: Upsampler diffusion process
        clip_model: CLIP model for guidance
        prompts: Text prompts for generation
        low_res_samples: Low resolution input images [batch_size, 3, 64, 64]
        batch_size: Number of images to process
        cutout_size: Size of cutouts (64 for GLIDE CLIP)
        num_cutouts: Number of cutouts per image
        cutout_power: Power for cutout size distribution
        guidance_scale: Strength of CLIP guidance
        sampler: Sampling method (ignored - always uses PLMS for upsampler)
        num_steps: Number of diffusion steps (ignored - uses fast27 respacing)
        upsample_temp: Temperature for upsampling noise
        device: Device to run on
        **model_kwargs: Additional arguments for model

    Returns:
        High resolution images [batch_size, 3, 256, 256]
    """

    # Create SR cutout guidance with dual loss (text + consistency)
    sr_guidance = CLIPGuidedSuperResCutouts(
        clip_model,
        cutout_size,
        num_cutouts,
        cutout_power,
        guidance_scale,
    )
    # Move to correct device to ensure buffers are on GPU
    sr_guidance.make_cutouts.to(device)

    # Create conditioning function with low-res input for consistency
    cond_fn = sr_guidance.cond_fn(prompts, low_res_samples)

    # Prepare upsampler model kwargs
    up_model_kwargs = dict(
        # Low-res image to upsample (GLIDE format)
        low_res=((low_res_samples + 1) * 127.5).round() / 127.5 - 1,
        **model_kwargs,  # Includes text tokens
    )

    up_shape = (batch_size, 3, 256, 256)  # GLIDE upsampler output size
    noise = th.randn(up_shape, device=device) * upsample_temp

    # Use default GLIDE upsampler method (p_sample_loop with fast27 timestep respacing)
    if save_frames:
        # For frame saving, we need to implement a custom loop
        # But for now, use standard p_sample_loop and save fewer frames
        up_samples = diffusion_up.p_sample_loop(
            model_up,
            up_shape,
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=up_model_kwargs,
            cond_fn=cond_fn,
        )
        # Save a few representative frames for SR phase
        save_frame(up_samples, 0, "sr", frame_dir)
        save_frame(up_samples, num_steps, "sr", frame_dir)  # Final frame
    else:
        up_samples = diffusion_up.p_sample_loop(
            model_up,
            up_shape,
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=up_model_kwargs,
            cond_fn=cond_fn,
        )

    return up_samples[:batch_size]


def main():
    """
    Example usage of CLIP-guided cutouts with GLIDE.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="CLIP-guided GLIDE generation with cutouts (clean version)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="an oil painting of a corgi",
        help="Text prompt for generation (default: 'an oil painting of a corgi')",
    )
    parser.add_argument(
        "--base_steps",
        type=int,
        default=30,
        help="Number of diffusion steps for base generation (default: 30)",
    )
    parser.add_argument(
        "--sr_steps",
        type=int,
        default=27,
        help="Number of diffusion steps for super-resolution (default: 27)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for base generation (default: 5.0)",
    )
    parser.add_argument(
        "--sr_guidance_scale",
        type=float,
        default=2.0,
        help="Guidance scale for super-resolution (default: 2.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["ddim", "euler", "euler_ancestral", "dpm"],
        help="Sampling method (default: euler)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to custom GLIDE base model checkpoint (.pt/.pth/.ckpt). "
             "Supports various formats: direct state_dict, training checkpoints with 'model_state_dict', "
             "and checkpoints with metadata. (default: use downloaded GLIDE base model)",
    )
    parser.add_argument(
        "--num_cutouts",
        type=int,
        default=32,
        help="Number of cutouts for CLIP guidance. More cutouts = stronger guidance but slower. "
             "Base generation uses this value, SR uses half for performance. (default: 32)",
    )

    args = parser.parse_args()
    
    # Validate step count ranges
    if args.base_steps < 1 or args.base_steps > 1000:
        raise ValueError(f"base_steps must be between 1 and 1000, got {args.base_steps}")
    if args.sr_steps < 1 or args.sr_steps > 1000:
        raise ValueError(f"sr_steps must be between 1 and 1000, got {args.sr_steps}")
    
    # Validate cutout count
    if args.num_cutouts < 1 or args.num_cutouts > 128:
        raise ValueError(f"num_cutouts must be between 1 and 128, got {args.num_cutouts}")
    
    # Validate custom model checkpoint if provided
    if args.model is not None:
        if not os.path.exists(args.model):
            raise ValueError(f"Custom model checkpoint not found: {args.model}")
        if not args.model.endswith(('.pt', '.pth', '.ckpt')):
            raise ValueError(f"Custom model must be a PyTorch checkpoint (.pt, .pth, or .ckpt), got: {args.model}")
    
    # Print configuration information
    print(f"🔧 Configuration:")
    print(f"   • Base model: {'Custom (' + args.model + ')' if args.model else 'Default GLIDE base'}")
    print(f"   • Base generation: {args.base_steps} steps, {args.num_cutouts} cutouts (timestep respacing: '{args.base_steps}')")
    print(f"   • Super-resolution: {args.sr_steps} steps, {args.num_cutouts // 2} cutouts (respacing: 'fast27' if ≤27, 'fast50' if ≤50, else '100')")
    print(f"   • GIF frame timing: Scaled for {args.base_steps} base + {args.sr_steps} SR steps")
    print(f"   • Frame saving: ~20 base frames + 2 SR frames per generation")
    print()

    import torch as th
    from glide_text2im.clip.model_creation import create_clip_model
    from glide_text2im.download import load_checkpoint
    from glide_text2im.model_creation import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
        model_and_diffusion_defaults_upsampler,
    )

    # Setup device
    has_cuda = th.cuda.is_available()
    device = th.device("cpu" if not has_cuda else "cuda")

    # Create base model
    options = model_and_diffusion_defaults()
    options["use_fp16"] = has_cuda
    options["timestep_respacing"] = str(args.base_steps)  # Use dynamic step count for respacing
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    # Load base model checkpoint (custom or default)
    if args.model is not None:
        print(f"📦 Loading custom base model from: {args.model}")
        custom_checkpoint = th.load(args.model, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(custom_checkpoint, dict) and 'model_state_dict' in custom_checkpoint:
            # Training checkpoint format
            state_dict = custom_checkpoint['model_state_dict']
        elif isinstance(custom_checkpoint, dict) and any(key.startswith('input_blocks') or key.startswith('output_blocks') for key in custom_checkpoint.keys()):
            # Direct state dict format (may have extra metadata)
            state_dict = {k: v for k, v in custom_checkpoint.items() if not k.startswith('fp16_metadata')}
        else:
            # Assume it's a direct state dict
            state_dict = custom_checkpoint
            
        model.load_state_dict(state_dict)
        print(f"✅ Successfully loaded custom model with {len(state_dict)} parameters")
    else:
        print("📦 Loading default GLIDE base model...")
        model.load_state_dict(load_checkpoint("base", device))

    # Create upsampler model
    options_up = model_and_diffusion_defaults_upsampler()
    options_up["use_fp16"] = has_cuda
    # Use predefined fast respacing for upsampler
    if args.sr_steps <= 27:
        options_up["timestep_respacing"] = "fast27"
    elif args.sr_steps <= 50:
        options_up["timestep_respacing"] = "fast50"
    else:
        options_up["timestep_respacing"] = "100"
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint("upsample", device))

    # Create CLIP model
    clip_model = create_clip_model(device=device)
    clip_model.image_encoder.load_state_dict(load_checkpoint("clip/image-enc", device))
    clip_model.text_encoder.load_state_dict(load_checkpoint("clip/text-enc", device))

    # Create output directory with auto-increment
    base_output_dir = "glide_outputs"
    os.makedirs(base_output_dir, exist_ok=True)

    # Find next available directory number
    existing_dirs = glob.glob(os.path.join(base_output_dir, "[0-9][0-9][0-9][0-9]"))
    if existing_dirs:
        # Extract numbers and find the highest
        numbers = [int(os.path.basename(d)) for d in existing_dirs]
        next_num = max(numbers) + 1
    else:
        next_num = 0

    # Create numbered output directory
    numbered_dir = os.path.join(base_output_dir, f"{next_num:04d}")
    os.makedirs(numbered_dir, exist_ok=True)

    # Override args.output_dir to use our numbered directory
    args.output_dir = numbered_dir

    print(f"📁 Output directory: {numbered_dir}")

    # Prepare model kwargs for text conditioning
    prompt = args.prompt
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, int(options["text_ctx"])
    )
    model_kwargs = dict(
        tokens=th.tensor([tokens], device=device),
        mask=th.tensor([mask], dtype=th.bool, device=device),
    )

    # Generate with cutout guidance
    frame_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Create safe filename from prompt
    import re

    safe_prompt = re.sub(r"[^\w\s-]", "", prompt)[:50]
    safe_prompt = re.sub(r"[-\s]+", "_", safe_prompt).strip("_")
    base_filename = f"{safe_prompt}_steps{args.base_steps}"

    # Generate base sample with frame saving
    print(f"\n🎨 Generating base sample (64x64) with CLIP guidance...")

    samples = sample_with_cutouts(
        model=model,
        diffusion=diffusion,
        clip_model=clip_model,
        prompts=[prompt],
        batch_size=1,
        image_size=int(options["image_size"]),
        cutout_size=64,
        num_cutouts=args.num_cutouts,
        cutout_power=1.0,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        num_steps=args.base_steps,
        device=device,
        save_frames=True,
        frame_dir=frame_dir,
        **model_kwargs,
    )

    # Save base result
    from torchvision.utils import save_image

    base_output_path = os.path.join(args.output_dir, f"{base_filename}_base_64x64.png")
    save_image((samples + 1) * 0.5, base_output_path)
    print(f"Base sample (64x64) saved to {base_output_path}")

    # Prepare upsampler model kwargs
    tokens_up = model_up.tokenizer.encode(prompt)
    tokens_up, mask_up = model_up.tokenizer.padded_tokens_and_mask(
        tokens_up, int(options_up["text_ctx"])
    )
    model_kwargs_up = dict(
        tokens=th.tensor([tokens_up], device=device),
        mask=th.tensor([mask_up], dtype=th.bool, device=device),
    )

    # Super-resolution with cutout guidance
    print("\n🔍 Starting super-resolution with cutout guidance...")
    sr_samples = sample_sr_with_cutouts(
        model_up=model_up,
        diffusion_up=diffusion_up,
        clip_model=clip_model,
        prompts=[prompt],
        low_res_samples=samples,
        batch_size=1,
        cutout_size=64,  # GLIDE CLIP size
        num_cutouts=max(1, args.num_cutouts // 2),  # Fewer cutouts for SR (more expensive)
        cutout_power=0.8,  # Less aggressive size variation
        guidance_scale=args.sr_guidance_scale,
        sampler=args.sampler,
        num_steps=args.sr_steps,
        upsample_temp=0.997,
        save_frames=True,
        frame_dir=frame_dir,
        device=device,
        **model_kwargs_up,
    )

    # Save final result
    sr_output_path = os.path.join(args.output_dir, f"{base_filename}_sr_256x256.png")
    save_image((sr_samples + 1) * 0.5, sr_output_path)
    print(f"Super-resolution sample (256x256) saved to {sr_output_path}")

    # Create end-to-end GIF of the entire generation process
    print("\n🎬 Creating end-to-end generation GIF...")
    gif_path = os.path.join(args.output_dir, f"{base_filename}_generation.gif")
    create_gif_from_frames(
        frame_dir=frame_dir,
        output_path=gif_path,
        duration=150,  # Base duration - will be adjusted per phase
        base_steps=args.base_steps,
        sr_steps=args.sr_steps,
    )

    print(f"🎬 Generation GIF saved to {gif_path}")
    print("\n🎉 End-to-end cutout-guided generation complete!")
    print(f"Prompt: '{prompt}'")
    print(
        f"Base model: 64x64 with timestep-aware cutouts ({args.base_steps} steps, guidance {args.guidance_scale})"
    )
    print(
        f"Super-res: 256x256 with dual-loss cutouts ({args.sr_steps} steps, guidance {args.sr_guidance_scale})"
    )
    print(f"📽️  Watch the full process: {gif_path}")


if __name__ == "__main__":
    main()
