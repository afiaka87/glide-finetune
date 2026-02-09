import os
import random
import time
import glob
from typing import Tuple

import numpy as np
import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm
import wandb

from glide_finetune import glide_util, train_util


def base_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    reals = batch[2].to(device, non_blocking=True, memory_format=th.channels_last)
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where
                - tokens is a tensor of shape (batch_size, seq_len),
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    low_res_image = batch[2].to(device, non_blocking=True, memory_format=th.channels_last)
    high_res_image = batch[3].to(device, non_blocking=True, memory_format=th.channels_last)
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device
    )
    noise = th.randn_like(
        high_res_image, device=device
    )  # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(
        high_res_image, timesteps, noise=noise
    ).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    ema_model=None,  # EMA model wrapper
    sample_bs: int = 1,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = "30",  # respacing for inference - using 30 steps with Euler
    sample_sampler: str = "euler",  # sampler for inference - Euler is fast and deterministic
    eval_base_sampler: str = "euler",  # sampler for base model evaluation
    eval_sr_sampler: str = "euler",  # sampler for super-resolution evaluation
    eval_base_sampler_steps: int = 30,  # number of steps for base model evaluation
    eval_sr_sampler_steps: int = 17,  # number of steps for super-resolution evaluation
    prompt: str = "",  # prompt for inference, not training
    prompt_file: str = "eval_captions.txt",  # file with prompts to sample from
    sample_batch_size: int = 8,  # number of prompts to randomly sample at each interval
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    sample_interval: int = 500,
    wandb_run=None,
    gradient_accumulation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    eval_sr_base_images="data/images/base_64x64",
    upsampler_model=None,
    upsampler_options=None,
    use_sr_eval: bool = False,
    use_lora: bool = False,
    lora_save_steps: int = 1000,
    save_checkpoint_interval: int = 5000,
    eval_interval: int = 5000,
    reference_stats: str = "",
):
    if train_upsample:
        train_step = upsample_train_step  # type: ignore
    else:
        train_step = base_train_step  # type: ignore

    # Load eval prompts if available
    eval_prompts = []
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            eval_prompts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(eval_prompts)} eval prompts from {prompt_file}")
    else:
        print(f"No {prompt_file} found, using fixed prompt: {prompt}")

    # Model should already be on correct device - moved in load_model_with_lora or before EMA creation
    glide_model.to(memory_format=th.channels_last)
    glide_model.train()

    # torch.compile for training speedup
    if th.cuda.is_available() and not getattr(glide_model, "_compiled", False):
        th.set_float32_matmul_precision("high")
        print("torch.compile: wrapping model...")
        compile_t0 = time.time()
        glide_model = th.compile(glide_model)
        glide_model._compiled = True  # type: ignore
        print(f"torch.compile: wrapped in {time.time() - compile_t0:.1f}s (compilation deferred to first forward/backward)")

    # Move EMA model to the same device if it exists
    if ema_model is not None:
        ema_model.to(device)

    log: dict = {}
    needs_warmup = getattr(glide_model, "_compiled", False)

    # Initialize timing for samples/sec calculation
    start_time = time.time()
    last_log_time = start_time
    samples_processed = 0
    accumulated_loss = 0.0
    current_loss = 0.0  # For logging the most recent loss

    # Zero gradients at the start
    optimizer.zero_grad()

    print("Waiting for first batch from dataloader (workers are loading tar files)...")
    dl_t0 = time.time()
    pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", unit="step", dynamic_ncols=True)
    for train_idx, batch in pbar:
        if train_idx == 0:
            print(f"First batch received in {time.time() - dl_t0:.1f}s")

        # Compute loss
        if needs_warmup:
            print("torch.compile: compiling forward pass (this may take ~40s)...")
            fwd_t0 = time.time()

        loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )

        if needs_warmup:
            print(f"torch.compile: forward compiled in {time.time() - fwd_t0:.1f}s")
            print("torch.compile: compiling backward pass (this may take ~40s)...")
            bwd_t0 = time.time()

        # Scale loss by gradient accumulation steps
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        if needs_warmup:
            print(f"torch.compile: backward compiled in {time.time() - bwd_t0:.1f}s")
            print(f"torch.compile: warmup complete, total {time.time() - fwd_t0:.1f}s")
            needs_warmup = False
        
        # Accumulate the loss for logging
        accumulated_loss += loss.item()
        current_loss = loss.item()  # Store current loss for immediate logging

        # Calculate samples per second
        batch_size = (
            batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
        )
        samples_processed += batch_size

        # Calculate current time and average samples/sec (needed for logging)
        current_time = time.time()
        total_elapsed = current_time - start_time
        avg_samples_per_sec = (
            samples_processed / total_elapsed if total_elapsed > 0 else 0
        )

        # Log loss to wandb every step
        wandb_run.log({
            "loss_step": current_loss,
            "iter": train_idx,
            "samples_per_sec": avg_samples_per_sec,
            "total_samples": samples_processed,
        })

        # Update tqdm bar with current loss every step
        pbar.set_postfix(loss=f"{current_loss:.4f}", sps=f"{avg_samples_per_sec:.1f}")

        # Update weights every gradient_accumulation_steps
        if (train_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Update EMA after optimizer step (as per OpenAI's guided-diffusion)
            if ema_model is not None:
                ema_model.update()

            # Calculate the averaged loss for this accumulation
            avg_accumulated_loss = accumulated_loss / gradient_accumulation_steps

            # Log the averaged (per-optimizer-step) loss
            log = {
                **log,
                "loss": avg_accumulated_loss,
            }
            wandb_run.log(log)

            # Reset accumulated loss
            accumulated_loss = 0.0

        # Sample from the model at sample_interval
        if train_idx > 0 and train_idx % sample_interval == 0:
            glide_model.eval()
            # Swap in EMA weights for evaluation
            if ema_model is not None:
                ema_model.swap()

            # Select prompts for sampling
            if eval_prompts:
                # Use all eval prompts in order for consistent tracking
                sample_prompts = eval_prompts
            else:
                # Use the fixed prompt multiple times
                sample_prompts = [prompt] * sample_batch_size

            print(
                f"Sampling {len(sample_prompts)} images from model at iteration {train_idx}"
            )

            # Generate images for all prompts
            all_images_64 = []  # 64x64 base images
            all_images_256 = []  # 256x256 SR images (if enabled)
            wandb_images_64 = []
            wandb_images_256 = []
            wandb_images_input_64 = []  # Input base images for SR training
            wandb_images_reference_256 = []  # Reference SR images for comparison

            # For upsampler training, load base images from directory
            if train_upsample and os.path.exists(eval_sr_base_images):
                base_image_files = sorted(glob.glob(os.path.join(eval_sr_base_images, "*.png")))
                base_caption_files = sorted(glob.glob(os.path.join(eval_sr_base_images, "*.txt")))

                # Limit to available images or sample_prompts, whichever is smaller
                num_to_eval = min(len(sample_prompts), len(base_image_files))
                sample_prompts = sample_prompts[:num_to_eval]
                base_image_files = base_image_files[:num_to_eval]

                # Load corresponding captions if available
                if len(base_caption_files) >= num_to_eval:
                    loaded_captions = []
                    for caption_file in base_caption_files[:num_to_eval]:
                        with open(caption_file, 'r') as f:
                            loaded_captions.append(f.read().strip())
                    sample_prompts = loaded_captions

            for idx, sample_prompt in enumerate(sample_prompts):
                print(
                    f"  [{idx + 1}/{len(sample_prompts)}] {sample_prompt[:80]}..."
                    if len(sample_prompt) > 80
                    else f"  [{idx + 1}/{len(sample_prompts)}] {sample_prompt}"
                )

                # Map 'standard' to appropriate sampler
                base_sampler = (
                    "plms" if eval_base_sampler == "standard" else eval_base_sampler
                )

                # For upsampler training, use specific base image
                current_image_to_upsample = None
                if train_upsample and os.path.exists(eval_sr_base_images):
                    if idx < len(base_image_files):
                        current_image_to_upsample = base_image_files[idx]

                        # Load and log the input base image
                        from PIL import Image
                        input_base_img = Image.open(current_image_to_upsample).convert("RGB")
                        wandb_images_input_64.append(wandb.Image(input_base_img, caption=f"Input: {sample_prompt[:50]}..."))

                        # Load reference SR image if it exists
                        base_name = os.path.splitext(os.path.basename(current_image_to_upsample))[0]
                        ref_sr_path = os.path.join("data/images/sr_256x256", f"{base_name}.png")
                        if os.path.exists(ref_sr_path):
                            ref_sr_img = Image.open(ref_sr_path).convert("RGB")
                            wandb_images_reference_256.append(wandb.Image(ref_sr_img, caption=f"Reference: {sample_prompt[:50]}..."))

                # First, always generate 64x64 base images (or upsample if training SR)
                samples_64 = glide_util.sample(
                    glide_model=glide_model,
                    glide_options=glide_options,
                    side_x=side_x,
                    side_y=side_y,
                    prompt=sample_prompt,
                    batch_size=sample_bs,
                    guidance_scale=sample_gs,
                    device=device,
                    prediction_respacing=str(eval_base_sampler_steps),
                    sampler=base_sampler,  # type: ignore
                    upsample_enabled=train_upsample,
                    image_to_upsample=current_image_to_upsample if train_upsample else None,
                )

                # For upsampler training, samples_64 is actually 256x256 output
                if train_upsample:
                    # samples_64 is 256x256 when upsampling
                    pil_image_256 = train_util.pred_to_pil(samples_64)
                    all_images_256.append(pil_image_256)
                    wandb_images_256.append(wandb.Image(pil_image_256, caption=f"Generated: {sample_prompt[:50]}..."))
                else:
                    # Normal 64x64 base model output
                    pil_image_64 = train_util.pred_to_pil(samples_64)
                    all_images_64.append(pil_image_64)
                    wandb_images_64.append(wandb.Image(pil_image_64, caption=f"{sample_prompt} (64x64)"))

                # Generate 256x256 with SR if requested and we're training base model
                if use_sr_eval and not train_upsample and upsampler_model is not None:
                    print(f"    Generating 256x256 with super-resolution...")

                    sr_sampler = (
                        "plms" if eval_sr_sampler == "standard" else eval_sr_sampler
                    )

                    # Use the 64x64 samples as input to upsampler
                    samples_256 = glide_util.sample_with_superres(
                        base_model=glide_model,
                        base_options=glide_options,
                        upsampler_model=upsampler_model,
                        upsampler_options=upsampler_options,
                        prompt=sample_prompt,
                        batch_size=sample_bs,
                        guidance_scale=sample_gs,
                        device=device,
                        base_respacing=str(eval_base_sampler_steps),
                        upsampler_respacing=str(eval_sr_sampler_steps),
                        upsample_temp=0.997,
                        base_sampler=base_sampler,  # type: ignore
                        upsampler_sampler=sr_sampler,  # type: ignore
                    )

                    # Convert 256x256 to PIL and store
                    pil_image_256 = train_util.pred_to_pil(samples_256)
                    all_images_256.append(pil_image_256)
                    wandb_images_256.append(wandb.Image(pil_image_256, caption=f"{sample_prompt} (256x256)"))

            # Create and save grid images
            wandb_log_dict = {"iter": train_idx}

            # For upsampler training, create comparison galleries
            if train_upsample:
                # Log input base images, generated SR, and reference SR
                if wandb_images_input_64:
                    wandb_log_dict["input_base_64px"] = wandb_images_input_64
                if wandb_images_256:
                    wandb_log_dict["generated_sr_256px"] = wandb_images_256
                if wandb_images_reference_256:
                    wandb_log_dict["reference_sr_256px"] = wandb_images_reference_256

                # Create comparison table for easy viewing - using WandB Table for better display
                if len(wandb_images_input_64) > 0 and len(wandb_images_256) > 0:
                    print(f"Creating comparison table with {len(wandb_images_input_64)} input images and {len(wandb_images_256)} generated images")

                    # Create a table to show all comparisons
                    comparison_table = wandb.Table(columns=["Sample #", "Prompt", "Input (64px)", "Generated (256px)", "Reference (256px)"])

                    # Add all samples to the table
                    num_samples = min(len(wandb_images_input_64), len(wandb_images_256))
                    for i in range(num_samples):
                        # Get prompt text
                        prompt_text = ""
                        if i < len(sample_prompts):
                            prompt_text = sample_prompts[i][:100] if sample_prompts[i] else ""

                        # Get reference image if available
                        ref_image = None
                        if i < len(wandb_images_reference_256):
                            ref_image = wandb_images_reference_256[i]

                        row_data = [
                            i + 1,  # Sample number
                            prompt_text,  # Truncated prompt
                            wandb_images_input_64[i],  # Input image
                            wandb_images_256[i],  # Generated image
                            ref_image  # Reference image (can be None)
                        ]
                        comparison_table.add_data(*row_data)

                    print(f"Added {num_samples} rows to comparison table")
                    wandb_log_dict["sr_comparison_table"] = comparison_table

                    # Also create a comparison grid image for quick viewing
                    # This will show input, generated, reference side by side for all samples
                    if len(wandb_images_input_64) > 0:
                        try:
                            from PIL import Image
                            import numpy as np

                            # Create side-by-side comparisons
                            comparison_rows = []
                            for i in range(min(len(wandb_images_input_64), len(wandb_images_256))):
                                row_images = []

                                # Get input image (resize to 256x256 for display)
                                # wandb.Image objects store the PIL image in the _image attribute
                                try:
                                    # Try to get the underlying PIL image
                                    if hasattr(wandb_images_input_64[i], '_image'):
                                        input_img = wandb_images_input_64[i]._image
                                    else:
                                        # If it's already a PIL image or path, handle it
                                        input_img = wandb_images_input_64[i]

                                    # Convert to PIL Image if needed
                                    if isinstance(input_img, str):
                                        input_img = Image.open(input_img)
                                    elif isinstance(input_img, np.ndarray):
                                        input_img = Image.fromarray(input_img)

                                    if isinstance(input_img, Image.Image):
                                        input_resized = input_img.resize((256, 256), Image.LANCZOS)
                                        row_images.append(input_resized)
                                except Exception as e:
                                    print(f"Warning: Could not process input image {i}: {e}")

                                # Get generated image
                                try:
                                    if hasattr(wandb_images_256[i], '_image'):
                                        gen_img = wandb_images_256[i]._image
                                    else:
                                        gen_img = wandb_images_256[i]

                                    if isinstance(gen_img, str):
                                        gen_img = Image.open(gen_img)
                                    elif isinstance(gen_img, np.ndarray):
                                        gen_img = Image.fromarray(gen_img)

                                    if isinstance(gen_img, Image.Image):
                                        row_images.append(gen_img)
                                except Exception as e:
                                    print(f"Warning: Could not process generated image {i}: {e}")

                                # Get reference image if available
                                if i < len(wandb_images_reference_256):
                                    try:
                                        if hasattr(wandb_images_reference_256[i], '_image'):
                                            ref_img = wandb_images_reference_256[i]._image
                                        else:
                                            ref_img = wandb_images_reference_256[i]

                                        if isinstance(ref_img, str):
                                            ref_img = Image.open(ref_img)
                                        elif isinstance(ref_img, np.ndarray):
                                            ref_img = Image.fromarray(ref_img)

                                        if isinstance(ref_img, Image.Image):
                                            row_images.append(ref_img)
                                    except Exception as e:
                                        print(f"Warning: Could not process reference image {i}: {e}")

                                # Concatenate images horizontally if we have them
                                if len(row_images) >= 2:
                                    # Ensure all images are same height
                                    height = 256
                                    resized_row = []
                                    for img in row_images:
                                        if img.height != height:
                                            aspect = img.width / img.height
                                            new_width = int(height * aspect)
                                            img = img.resize((new_width, height), Image.LANCZOS)
                                        resized_row.append(img)

                                    # Concatenate horizontally with 5px spacing
                                    spacing = 5
                                    total_width = sum(img.width for img in resized_row) + spacing * (len(resized_row) - 1)
                                    row_img = Image.new('RGB', (total_width, height), (128, 128, 128))
                                    x_offset = 0
                                    for img in resized_row:
                                        row_img.paste(img, (x_offset, 0))
                                        x_offset += img.width + spacing

                                    comparison_rows.append(row_img)

                            # Stack all rows vertically
                            if comparison_rows:
                                spacing = 5
                                total_height = sum(img.height for img in comparison_rows) + spacing * (len(comparison_rows) - 1)
                                max_width = max(img.width for img in comparison_rows)

                                full_comparison = Image.new('RGB', (max_width, total_height), (128, 128, 128))
                                y_offset = 0
                                for row_img in comparison_rows:
                                    # Center each row horizontally
                                    x_offset = (max_width - row_img.width) // 2
                                    full_comparison.paste(row_img, (x_offset, y_offset))
                                    y_offset += row_img.height + spacing

                                # Save and log the comparison grid
                                comparison_grid_path = os.path.join(outputs_dir, f"{train_idx}_comparison_grid.png")
                                full_comparison.save(comparison_grid_path)
                                wandb_log_dict["sr_comparison_grid"] = wandb.Image(
                                    full_comparison,
                                    caption=f"Comparison grid: Input (64px upscaled) | Generated (256px) | Reference (256px) - {len(comparison_rows)} samples"
                                )
                                print(f"Created comparison grid with {len(comparison_rows)} samples")
                        except Exception as e:
                            print(f"Warning: Could not create comparison grid: {e}")

            # Skip 64x64 grid for upsampler training (we have 256x256 instead)
            if not train_upsample and len(all_images_64) > 1:
                # Use auto mode for optimal grid layout
                grid_image_64 = train_util.make_grid(
                    all_images_64,
                    mode='auto',
                    pad_to_power_of_2=False,
                    background_color=(0, 0, 0)
                )

                # Save 64x64 grid
                grid_save_path_64 = os.path.join(
                    outputs_dir, f"{train_idx}_grid_64px.png"
                )
                grid_image_64.save(grid_save_path_64)
                # Also save as current_grid.png for easy monitoring
                grid_image_64.save("current_grid.png")
                print(f"Saved 64x64 grid with {len(all_images_64)} images to {grid_save_path_64}")

                # Add to wandb log dict
                wandb_log_dict.update({
                    "sample_grid_64px": wandb.Image(
                        grid_save_path_64, caption=f"Grid of {len(all_images_64)} samples (64x64)"
                    ),
                    "sample_gallery_64px": wandb_images_64,
                })
            elif not train_upsample and len(all_images_64) == 1:
                # Single 64x64 image case
                sample_save_path_64 = os.path.join(outputs_dir, f"{train_idx}_64px.png")
                all_images_64[0].save(sample_save_path_64)
                # Also save as current_grid.png for easy monitoring (single image)
                all_images_64[0].save("current_grid.png")
                print(f"Saved 64x64 sample {sample_save_path_64}")

                wandb_log_dict.update({
                    "samples_64px": wandb_images_64[0],
                })

            # Create and save grid images for 256x256 if SR evaluation is enabled
            if all_images_256:
                if len(all_images_256) > 1:
                    # Use auto mode for optimal grid layout
                    grid_image_256 = train_util.make_grid(
                        all_images_256,
                        mode='auto',
                        pad_to_power_of_2=False,
                        background_color=(0, 0, 0)
                    )

                    # Save 256x256 grid
                    grid_save_path_256 = os.path.join(
                        outputs_dir, f"{train_idx}_grid_256px.png"
                    )
                    grid_image_256.save(grid_save_path_256)
                    # Save as current_grid.png for easy monitoring (prefer 256px version)
                    grid_image_256.save("current_grid.png")
                    print(f"Saved 256x256 grid with {len(all_images_256)} images to {grid_save_path_256}")

                    # Add to wandb log dict
                    wandb_log_dict.update({
                        "sample_grid_256px": wandb.Image(
                            grid_save_path_256, caption=f"Grid of {len(all_images_256)} samples (256x256)"
                        ),
                        "sample_gallery_256px": wandb_images_256,
                    })
                else:
                    # Single 256x256 image case
                    sample_save_path_256 = os.path.join(outputs_dir, f"{train_idx}_256px.png")
                    all_images_256[0].save(sample_save_path_256)
                    # Save as current_grid.png for easy monitoring (prefer 256px version)
                    all_images_256[0].save("current_grid.png")
                    print(f"Saved 256x256 sample {sample_save_path_256}")

                    wandb_log_dict.update({
                        "samples_256px": wandb_images_256[0],
                    })

            # Compute CLIP scores on generated samples
            clip_images = all_images_64 if all_images_64 else all_images_256
            if clip_images and sample_prompts:
                try:
                    from glide_finetune.metrics import compute_clip_scores

                    clip_results = compute_clip_scores(
                        clip_images[:len(sample_prompts)],
                        sample_prompts[:len(clip_images)],
                        device=device,
                    )
                    wandb_log_dict["clip_score_mean"] = clip_results["clip_score_mean"]
                    wandb_log_dict["clip_score_std"] = clip_results["clip_score_std"]
                    print(
                        f"CLIP score: {clip_results['clip_score_mean']:.4f} "
                        f"(+/- {clip_results['clip_score_std']:.4f})"
                    )
                except Exception as e:
                    print(f"Warning: CLIP score computation failed: {e}")

            # Log everything to wandb
            wandb_run.log(wandb_log_dict)

            # Swap training weights back in
            if ema_model is not None:
                ema_model.swap()
            glide_model.train()

        # FID/KID evaluation at eval_interval
        if (
            eval_interval > 0
            and reference_stats
            and train_idx > 0
            and train_idx % eval_interval == 0
        ):
            glide_model.eval()
            # Swap in EMA weights for evaluation
            if ema_model is not None:
                ema_model.swap()
            try:
                from glide_finetune.metrics import compute_fid_kid

                # Use eval prompts if available, else use default human prompts
                fid_prompts = eval_prompts if eval_prompts else [prompt] * 100
                print(f"\nComputing FID/KID at step {train_idx}...")
                fid_kid_results = compute_fid_kid(
                    glide_model=glide_model,
                    glide_diffusion=glide_diffusion,
                    glide_options=glide_options,
                    eval_prompts=fid_prompts,
                    reference_stats_path=reference_stats,
                    device=device,
                    num_samples=500,
                    guidance_scale=sample_gs,
                    sampler=eval_base_sampler,
                    sampler_steps=eval_base_sampler_steps,
                    side_x=side_x,
                    side_y=side_y,
                )
                wandb_run.log({
                    "fid": fid_kid_results["fid"],
                    "kid_mean": fid_kid_results["kid_mean"],
                    "kid_std": fid_kid_results["kid_std"],
                    "iter": train_idx,
                })
                print(
                    f"FID: {fid_kid_results['fid']:.2f}, "
                    f"KID: {fid_kid_results['kid_mean']:.4f} "
                    f"(+/- {fid_kid_results['kid_std']:.4f})"
                )
            except Exception as e:
                print(f"Warning: FID/KID computation failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Swap training weights back in
                if ema_model is not None:
                    ema_model.swap()
                glide_model.train()

        # Save LoRA adapter if enabled and at save interval
        if (
            use_lora
            and lora_save_steps > 0
            and train_idx % lora_save_steps == 0
            and train_idx > 0
        ):
            from glide_finetune.lora_wrapper import save_lora_checkpoint

            lora_save_path = os.path.join(
                checkpoints_dir, f"lora_adapter_{epoch}_{train_idx}"
            )
            save_lora_checkpoint(
                glide_model,
                lora_save_path,
                metadata={
                    "epoch": epoch,
                    "step": train_idx,
                    "loss": current_loss,
                },
            )
            print(f"Saved LoRA adapter to {lora_save_path}")

        if (
            save_checkpoint_interval > 0
            and train_idx % save_checkpoint_interval == 0
            and train_idx > 0
        ):
            if use_lora:
                # For LoRA, save adapter separately
                from glide_finetune.lora_wrapper import save_lora_checkpoint

                lora_save_path = os.path.join(
                    checkpoints_dir, f"lora_checkpoint_{epoch}_{train_idx}"
                )
                save_lora_checkpoint(
                    glide_model,
                    lora_save_path,
                    metadata={
                        "epoch": epoch,
                        "step": train_idx,
                        "loss": current_loss,
                    },
                )
                print(f"Saved LoRA checkpoint to {lora_save_path}")
            else:
                # Save full model if not using LoRA
                train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)

                # Save EMA model if available
                if ema_model is not None:
                    train_util.save_ema_model(ema_model, checkpoints_dir, train_idx, epoch, ema_model.decay)
                    print(f"Saved EMA checkpoint with decay {ema_model.decay}")

    pbar.close()

    # Flush any remaining accumulated gradients from a partial accumulation
    if (train_idx + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        if ema_model is not None:
            ema_model.update()

    print("Finished training, saving final checkpoint")
    if use_lora:
        from glide_finetune.lora_wrapper import save_lora_checkpoint

        final_lora_path = os.path.join(checkpoints_dir, f"lora_final_{epoch}")
        save_lora_checkpoint(
            glide_model,
            final_lora_path,
            metadata={
                "epoch": epoch,
                "step": train_idx,
                "final": True,
            },
        )
        print(f"Saved final LoRA adapter to {final_lora_path}")
    else:
        train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
        # Save final EMA model if available
        if ema_model is not None:
            train_util.save_ema_model(ema_model, checkpoints_dir, train_idx, epoch, ema_model.decay)
            print(f"Saved final EMA checkpoint with decay {ema_model.decay}")
