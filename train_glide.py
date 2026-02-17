import argparse
from glob import glob
import os
from braceexpand import braceexpand
import tarfile
import json
from pathlib import Path
import hashlib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import torch as th
from tqdm import trange, tqdm

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import (
    load_model,
    load_model_with_lora,
    load_latent_model,
)
from glide_finetune.loader import TextImageDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader


def get_tar_file_info(tar_path):
    """Get file stats for a tar file (size and modification time)."""
    try:
        stat = os.stat(tar_path)
        return {"size": stat.st_size, "mtime": stat.st_mtime, "path": tar_path}
    except OSError:
        return None


def validate_single_tar(tar_path):
    """
    Validate a single tar file. Used for parallel processing.

    Args:
        tar_path: Path to tar file to validate

    Returns:
        Tuple of (tar_path, is_valid, error_message)
    """
    try:
        # Try to open and read the tar file
        with tarfile.open(tar_path, "r") as tf:
            # Quick validation - just try to get members list
            _ = tf.getmembers()
        return (tar_path, True, None)
    except (tarfile.ReadError, EOFError) as e:
        return (tar_path, False, f"Corrupted/incomplete: {type(e).__name__}")
    except Exception as e:
        return (tar_path, False, str(e))


def validate_tar_files(
    tar_files,
    skip_validation=False,
    use_cache=True,
    cache_dir="./cache",
    verbose=True,
    num_workers=None,
):
    """
    Validate tar files by attempting to read their contents.
    Caches validation results to avoid re-checking valid files.
    Uses parallel processing for faster validation.

    Args:
        tar_files: List of tar file paths to validate
        skip_validation: If True, skip validation and return all files
        use_cache: If True, use cached validation results
        cache_dir: Directory to store cache files
        verbose: If True, print progress and statistics
        num_workers: Number of parallel workers (None = auto-detect)

    Returns:
        List of valid tar file paths
    """
    if skip_validation:
        if verbose:
            print("Skipping tar validation (--skip_tar_validation flag set)")
        return tar_files

    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate cache filename based on dataset directory
    if tar_files:
        dataset_dir = os.path.dirname(tar_files[0])
        # Create a hash of the dataset directory for unique cache file
        dir_hash = hashlib.md5(dataset_dir.encode()).hexdigest()[:8]
        cache_file = cache_path / f"valid_tars_{dir_hash}.json"
    else:
        cache_file = cache_path / "valid_tars.json"

    # Load cached validation results if available
    cached_valid = {}
    cached_invalid = {}

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                cached_valid = {
                    item["path"]: item for item in cache_data.get("valid", [])
                }
                cached_invalid = {
                    item["path"]: item for item in cache_data.get("invalid", [])
                }
                if verbose:
                    cache_date = datetime.fromtimestamp(cache_data.get("timestamp", 0))
                    print(
                        f"Loaded validation cache from {cache_date.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    print(f"  Cached valid files: {len(cached_valid)}")
                    print(f"  Cached invalid files: {len(cached_invalid)}")
        except Exception as e:
            if verbose:
                print(f"Could not load cache file: {e}")
            cached_valid = {}
            cached_invalid = {}

    # Separate files into cached and uncached
    valid_tars = []
    corrupted_tars = []
    files_to_validate = []

    for tar_path in tar_files:
        file_info = get_tar_file_info(tar_path)
        if not file_info:
            corrupted_tars.append(tar_path)
            continue

        # Check if we have cached validation for this file
        if tar_path in cached_valid:
            cached_info = cached_valid[tar_path]
            # Check if file hasn't changed (same size and modification time)
            if (
                cached_info.get("size") == file_info["size"]
                and cached_info.get("mtime") == file_info["mtime"]
            ):
                valid_tars.append(tar_path)
                continue
        elif tar_path in cached_invalid:
            cached_info = cached_invalid[tar_path]
            # Check if file hasn't changed
            if (
                cached_info.get("size") == file_info["size"]
                and cached_info.get("mtime") == file_info["mtime"]
            ):
                corrupted_tars.append(tar_path)
                continue

        # File needs validation
        files_to_validate.append(tar_path)

    # Validate uncached files
    newly_validated = []
    newly_corrupted = []

    if files_to_validate:
        # Determine number of workers
        if num_workers is None:
            # Auto-detect: use all CPUs minus 1, minimum 1
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            # Use specified number, ensure it's at least 1
            num_workers = max(1, num_workers)

        if verbose:
            print(f"\nValidating {len(files_to_validate)} uncached tar files...")
            print(f"  (Skipping {len(valid_tars)} already validated files from cache)")
            print(f"  Using {num_workers} parallel workers for validation")

        # Use ProcessPoolExecutor for parallel validation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all validation tasks
            future_to_tar = {
                executor.submit(validate_single_tar, tar_path): tar_path
                for tar_path in files_to_validate
            }

            # Process results as they complete
            if verbose:
                # Use tqdm for progress bar
                pbar = tqdm(total=len(files_to_validate), desc="Validating tar files")

            for future in as_completed(future_to_tar):
                tar_path, is_valid, error_msg = future.result()

                if is_valid:
                    valid_tars.append(tar_path)
                    newly_validated.append(tar_path)
                else:
                    corrupted_tars.append(tar_path)
                    newly_corrupted.append(tar_path)
                    if verbose and error_msg:
                        # Only show errors for first few files to avoid spam
                        if len(newly_corrupted) <= 5:
                            tqdm.write(f"  ✗ {os.path.basename(tar_path)}: {error_msg}")

                if verbose:
                    pbar.update(1)

            if verbose:
                pbar.close()

    # Update cache with new validation results
    if use_cache and (newly_validated or newly_corrupted):
        # Update cached valid files
        for tar_path in newly_validated:
            file_info = get_tar_file_info(tar_path)
            if file_info:
                cached_valid[tar_path] = file_info

        # Update cached invalid files
        for tar_path in newly_corrupted:
            file_info = get_tar_file_info(tar_path)
            if file_info:
                cached_invalid[tar_path] = file_info

        # Save updated cache
        cache_data = {
            "timestamp": datetime.now().timestamp(),
            "total_validated": len(valid_tars) + len(corrupted_tars),
            "valid": list(cached_valid.values()),
            "invalid": list(cached_invalid.values()),
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            if verbose:
                print(f"Updated validation cache: {cache_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save cache file: {e}")

    if verbose:
        print("\nValidation complete:")
        print(f"  ✓ Valid tar files: {len(valid_tars)}")
        print(f"  ✗ Corrupted/incomplete tar files: {len(corrupted_tars)}")
        if files_to_validate:
            print(f"  📝 Newly validated: {len(newly_validated)}")
            print(f"  📝 Newly identified as corrupted: {len(newly_corrupted)}")

        if corrupted_tars and len(corrupted_tars) <= 10:
            print("\nSkipped tar files:")
            for tar in corrupted_tars:
                print(f"    - {os.path.basename(tar)}")
        elif corrupted_tars:
            print(
                f"\nSkipped {len(corrupted_tars)} corrupted tar files (too many to list)"
            )
            print(
                f"  First few: {', '.join([os.path.basename(t) for t in corrupted_tars[:5]])}"
            )

    if not valid_tars:
        raise ValueError(
            "No valid tar files found! All tar files appear to be corrupted or incomplete."
        )

    return valid_tars


def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-4,  # GLIDE paper value
    adam_weight_decay=0.0,  # GLIDE paper value
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,  # Deprecated, use precision parameter
    precision="fp32",  # "fp32", "fp16", "bf16"
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    wandb_project_name="glide_finetune",
    activation_checkpointing=False,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    sample_interval=500,
    test_prompt="a group of skiers are preparing to ski down a mountain.",
    sample_bs=1,
    sample_gs=8.0,
    prompt_file="eval_captions.txt",
    sample_batch_size=8,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    dataset_name="laion",
    enable_upsample=False,
    upsample_factor=4,
    eval_sr_base_images="data/images/base_64x64",
    use_sr_eval=False,
    sr_model_path=None,
    use_lora=False,
    lora_rank=4,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_mode="attention",
    lora_save_steps=1000,
    lora_resume="",
    save_checkpoint_interval=5000,
    gradient_accumulation_steps=1,
    random_hflip=False,
    ema_rate=0.9999,  # GLIDE paper value for EMA decay
    reinit_transformer=False,
    random_init=False,
    eval_interval=5000,
    reference_stats="",
    captions_jsonl_path=None,
    latent_mode=False,
    vae_model="stabilityai/sd-vae-ft-mse",
    clip_model_name="ViT-L-14",
    clip_pretrained="laion2b_s32b_b82k",
    init_from_pixel="",
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Handle legacy use_fp16 parameter
    if use_fp16:
        precision = "fp16"

    # Latent mode overrides
    if latent_mode:
        side_x, side_y = 32, 32
        print("Latent mode: diffusion on 32x32 latents (256x256 pixel output)")

    # Start wandb logging
    wandb_run = wandb_setup(
        batch_size=batch_size,
        side_x=side_x,
        side_y=side_y,
        learning_rate=learning_rate,
        use_fp16=(precision == "fp16"),
        device=device,
        data_dir=data_dir,
        base_dir=checkpoints_dir,
        project_name=wandb_project_name,
    )
    print("Wandb setup.")

    # Model setup
    if resume_ckpt:
        print("=" * 60)
        print(f"  RESUMING FROM CHECKPOINT: {resume_ckpt}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  NO CHECKPOINT TO RESUME FROM — TRAINING FROM SCRATCH")
        print("=" * 60)

    if latent_mode:
        glide_model, glide_diffusion, glide_options = load_latent_model(
            glide_path=resume_ckpt,
            precision=precision,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            init_from_pixel=init_from_pixel,
        )
    elif use_lora:
        glide_model, glide_diffusion, glide_options = load_model_with_lora(
            glide_path=resume_ckpt,
            use_fp16=False,  # Use precision parameter instead
            precision=precision,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            model_type="base" if not enable_upsample else "upsample",
            use_lora=True,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_mode=lora_target_mode,
            lora_resume=lora_resume,
            device=device,
        )
    else:
        glide_model, glide_diffusion, glide_options = load_model(
            glide_path=resume_ckpt,
            use_fp16=False,  # Use precision parameter instead
            precision=precision,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            model_type="base" if not enable_upsample else "upsample",
            random_init=random_init,
        )
    # Reinitialize transformer from scratch if requested (keeps pretrained UNet)
    if reinit_transformer:
        print("Reinitializing transformer/text encoder from scratch...")
        reinit_components = [
            ("transformer", True),
            ("transformer_proj", True),
            ("token_embedding", True),
            ("final_ln", True),
        ]
        reinit_count = 0
        for name, is_module in reinit_components:
            if hasattr(glide_model, name):
                module = getattr(glide_model, name)
                for p in module.parameters():
                    if p.dim() >= 2:
                        th.nn.init.xavier_uniform_(p)
                    else:
                        th.nn.init.zeros_(p)
                    reinit_count += p.numel()
        # Reinit parameter-type embeddings
        for name in ["padding_embedding", "positional_embedding"]:
            if hasattr(glide_model, name):
                p = getattr(glide_model, name)
                if isinstance(p, th.nn.Parameter):
                    th.nn.init.normal_(p, std=0.02)
                    reinit_count += p.numel()
        print(f"Reinitialized {reinit_count:,} transformer parameters")

    # Create frozen VAE and CLIP encoders for latent mode
    vae = None
    clip_enc = None
    if latent_mode:
        from glide_finetune.latent_util import LatentVAE, LatentCLIP

        vae_dtype = th.bfloat16 if precision == "bf16" else th.float32
        print(f"Loading frozen VAE ({vae_model}) ...")
        vae = LatentVAE(model_name=vae_model, device=str(device), dtype=vae_dtype)
        print(f"Loading frozen CLIP ({clip_model_name}/{clip_pretrained}) ...")
        clip_enc = LatentCLIP(
            model_name=clip_model_name, pretrained=clip_pretrained, device=str(device)
        )

    glide_model.train()
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    print(f"Number of parameters: {number_of_params}")
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    print(f"Trainable parameters: {number_of_trainable_params}")

    # Load upsampler model if needed for evaluation
    upsampler_model = None
    upsampler_options = None
    if use_sr_eval and not enable_upsample:
        print("Loading upsampler model for evaluation...")

        # Use custom SR model path if provided, otherwise use default
        sr_path = sr_model_path if sr_model_path else ""
        if sr_model_path:
            print(f"  Using custom SR model: {sr_model_path}")
        else:
            print("  Using default pretrained SR model")

        upsampler_model, _, upsampler_options = load_model(
            glide_path=sr_path,
            use_fp16=use_fp16,
            model_type="upsample",
        )
        upsampler_model.eval()
        upsampler_model.to(device)
        print(
            f"Upsampler loaded with {sum(x.numel() for x in upsampler_model.parameters())} parameters"
        )

    # Data setup
    print("Loading data...")
    if use_webdataset:
        dataset = glide_wds_loader(
            urls=data_dir,
            caption_key=caption_key,
            image_key=image_key,
            enable_image=True,
            enable_text=use_captions,
            enable_upsample=enable_upsample,
            tokenizer=glide_model.tokenizer,
            base_x=side_x,
            base_y=side_y,
            uncond_p=uncond_p,
            ar_lower=0.5,
            ar_upper=2.0,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_upper=0.0,
            similarity_threshold_lower=0.5,
            words_to_skip=[],
            dataset_name=dataset_name,  # can be laion, alamy, simple, synthetic, or datacomp.
            buffer_size=args.wds_buffer_size,
            initial_prefetch=args.wds_initial_prefetch,
            debug=args.wds_debug,
            random_hflip=random_hflip,
            captions_jsonl_path=captions_jsonl_path,
            latent_mode=latent_mode,
        )
    else:
        dataset = TextImageDataset(
            folder=data_dir,
            side_x=side_x,
            side_y=side_y,
            resize_ratio=resize_ratio,
            uncond_p=uncond_p,
            shuffle=True,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            use_captions=use_captions,
            enable_glide_upsample=enable_upsample,
            upscale_factor=upsample_factor,  # TODO: make this a parameter
            random_hflip=random_hflip,
            latent_mode=latent_mode,
        )

    # Data loader setup
    collate_fn = None
    if latent_mode:
        from glide_finetune.wds_loader import latent_collate_fn

        collate_fn = latent_collate_fn

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not use_webdataset,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
    )

    # Quick test to ensure dataloader is set up (without consuming data)
    print("\nDEBUG: Dataloader setup complete")
    print(f"DEBUG: Batch size: {batch_size}")
    print(f"DEBUG: Using webdataset: {use_webdataset}")
    if use_webdataset:
        print(f"DEBUG: Dataset name: {dataset_name}")
        print(
            f"DEBUG: Number of tar files: {len(data_dir) if isinstance(data_dir, list) else 'N/A'}"
        )

    # Move model to device before creating EMA (ensures both models are on same device)
    glide_model = glide_model.to(device)

    # Optimizer setup - GLIDE paper uses AdamW with default betas (0.9, 0.999)
    optimizer = th.optim.AdamW(
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
        weight_decay=adam_weight_decay,
        fused=th.cuda.is_available() and str(device) != "cpu",
        # Using PyTorch default betas=(0.9, 0.999) as per GLIDE paper
    )

    # EMA setup - GLIDE paper uses EMA decay of 0.9999
    ema_model = None
    if ema_rate > 0:
        from glide_finetune.ema_util import SimpleEMA

        print(f"Setting up EMA with decay rate {ema_rate}")
        ema_model = SimpleEMA(glide_model, decay=ema_rate)

    # Note: Freezing is already handled in load_model/load_model_with_lora
    # No need for additional requires_grad_ modifications here

    # Training setup
    outputs_dir = "./outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    existing_runs = [
        sub_dir
        for sub_dir in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, sub_dir))
    ]
    existing_runs_int = []
    for x in existing_runs:
        try:
            existing_runs_int.append(int(x))
        except ValueError:
            print("unexpected directory naming scheme")
            # ignore
    existing_runs_int = sorted(existing_runs_int)
    next_run = 0 if len(existing_runs) == 0 else existing_runs_int[-1] + 1
    current_run_ckpt_dir = os.path.join(checkpoints_dir, str(next_run).zfill(4))

    os.makedirs(current_run_ckpt_dir, exist_ok=True)

    for epoch in trange(num_epochs):
        print(f"Starting epoch {epoch}")
        run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            ema_model=ema_model,  # Pass EMA model for updates
            upsampler_model=upsampler_model,
            upsampler_options=upsampler_options,
            use_sr_eval=use_sr_eval,
            dataloader=dataloader,
            prompt=test_prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            prompt_file=prompt_file,
            sample_batch_size=sample_batch_size,
            eval_base_sampler=args.eval_base_sampler,
            eval_sr_sampler=args.eval_sr_sampler,
            eval_base_sampler_steps=args.eval_base_sampler_steps,
            eval_sr_sampler_steps=args.eval_sr_sampler_steps,
            checkpoints_dir=current_run_ckpt_dir,
            outputs_dir=outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            sample_interval=sample_interval,
            epoch=epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_upsample=enable_upsample,
            use_lora=use_lora,
            lora_save_steps=lora_save_steps,
            save_checkpoint_interval=save_checkpoint_interval,
            eval_interval=eval_interval,
            reference_stats=reference_stats,
            latent_mode=latent_mode,
            vae=vae,
            clip_encoder=clip_enc,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4
    )  # GLIDE paper value
    parser.add_argument(
        "--adam_weight_decay", "-adam_wd", type=float, default=0.0
    )  # GLIDE paper value
    parser.add_argument(
        "--ema_rate",
        type=float,
        default=0.9999,
        help="EMA decay rate (GLIDE uses 0.9999)",
    )
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument(
        "--resize_ratio", "-crop", type=float, default=0.8, help="Crop ratio"
    )
    parser.add_argument(
        "--random_hflip",
        action="store_true",
        help="Apply random horizontal flip augmentation during training (50% probability)",
    )
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--train_upsample",
        "-upsample",
        action="store_true",
        help="Train the upsampling type of the model instead of the base model.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/"
    )
    parser.add_argument(
        "--use_fp16",
        "-fp16",
        action="store_true",
        help="[Deprecated] Use --precision fp16 instead",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for training: fp32 (default), fp16 (unstable), bf16 (recommended for mixed precision)",
    )
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument(
        "--sample_interval",
        "-sample_freq",
        type=int,
        default=500,
        help="Frequency of sampling images for evaluation (defaults to 500)",
    )
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument(
        "--reinit_transformer",
        action="store_true",
        help="Reinitialize transformer/text encoder from scratch (use with --freeze_diffusion to train only text encoder)",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Skip loading any pretrained weights, train from random initialization",
    )
    parser.add_argument(
        "--wandb_project_name",
        "-wname",
        type=str,
        default="glide_finetune",
        help="Project name for wandb logging",
    )
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument(
        "--gradient_accumulation_steps",
        "-grad_acc",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)",
    )
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="a group of skiers are preparing to ski down a mountain.",
    )
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=4.0,
        help="Guidance scale used during model eval, not training.",
    )
    parser.add_argument(
        "--use_webdataset",
        "-wds",
        action="store_true",
        help="Enables webdataset (tar) loading",
    )
    parser.add_argument(
        "--wds_image_key",
        "-wds_img",
        type=str,
        default="jpg",
        help="A 'key' e.g. 'jpg' used to access the image in the webdataset",
    )
    parser.add_argument(
        "--wds_caption_key",
        "-wds_cap",
        type=str,
        default="txt",
        help="A 'key' e.g. 'txt' used to access the caption in the webdataset",
    )
    parser.add_argument(
        "--wds_dataset_name",
        "-wds_name",
        type=str,
        default="laion",
        help="Name of the webdataset to use (laion, alamy, simple, synthetic, datacomp-synthetic, or datacomp-real)",
    )
    parser.add_argument(
        "--wds_captions_jsonl",
        type=str,
        default=None,
        help="Path to external JSONL captions file (required for datacomp-synthetic dataset)",
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor",
        "-upscale",
        type=int,
        default=4,
        help="Upscale factor for training the upsampling model only",
    )
    parser.add_argument(
        "--eval_sr_base_images",
        type=str,
        default="data/images/base_64x64",
        help="Directory containing base 64x64 images for SR evaluation during training",
    )
    parser.add_argument(
        "--use_sr_eval",
        action="store_true",
        help="Use full pipeline (base + superres) for evaluation sampling during training. Generates both 64x64 and 256x256 images.",
    )
    parser.add_argument(
        "--sr_model_path",
        type=str,
        default=None,
        help="Path to the super-resolution model checkpoint. If not provided, will use default pretrained model.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="eval_captions.txt",
        help="Path to file containing prompts to randomly sample from during evaluation (one per line)",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Number of prompts to randomly sample and generate images for at each sample interval",
    )
    parser.add_argument(
        "--eval_base_sampler",
        type=str,
        default="euler",
        choices=["standard", "euler", "euler_a", "dpm++"],
        help="Sampler to use for base model evaluation (standard=PLMS/DDIM, euler, euler_a=ancestral, dpm++)",
    )
    parser.add_argument(
        "--eval_sr_sampler",
        type=str,
        default="euler",
        choices=["standard", "euler", "euler_a", "dpm++"],
        help="Sampler to use for super-resolution evaluation (standard=PLMS, euler, euler_a=ancestral, dpm++)",
    )
    parser.add_argument(
        "--eval_base_sampler_steps",
        type=int,
        default=30,
        help="Number of diffusion steps for base model evaluation (default: 30)",
    )
    parser.add_argument(
        "--eval_sr_sampler_steps",
        type=int,
        default=17,
        help="Number of diffusion steps for super-resolution evaluation (default: 17)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers for parallel data loading (default: 4)",
    )
    parser.add_argument(
        "--wds_buffer_size",
        type=int,
        default=1000,
        help="WebDataset shuffle buffer size (default: 1000)",
    )
    parser.add_argument(
        "--wds_initial_prefetch",
        type=int,
        default=10,
        help="WebDataset initial prefetch size (default: 10)",
    )
    parser.add_argument(
        "--wds_debug",
        action="store_true",
        help="Enable debug printing for WebDataset loading",
    )
    parser.add_argument(
        "--skip_tar_validation",
        action="store_true",
        help="Skip validation of tar files (use if you know all tar files are valid)",
    )
    parser.add_argument(
        "--no_cache_validation",
        action="store_true",
        help="Don't use cached validation results (force re-validation of all tar files)",
    )
    parser.add_argument(
        "--clear_validation_cache",
        action="store_true",
        help="Clear the validation cache before starting",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=None,
        help="Number of parallel workers for tar validation (default: auto-detect based on CPU count)",
    )

    # LoRA configuration arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA decomposition (default: 4)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA scaling parameter (default: 32)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout for LoRA layers (default: 0.1)",
    )
    parser.add_argument(
        "--lora_target_mode",
        type=str,
        default="attention",
        choices=["attention", "mlp", "all", "minimal"],
        help="Which modules to apply LoRA to (default: attention)",
    )
    parser.add_argument(
        "--lora_save_steps",
        type=int,
        default=1000,
        help="Save LoRA adapter every N steps (default: 1000)",
    )
    parser.add_argument(
        "--lora_resume", type=str, default="", help="Path to resume LoRA adapter from"
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=5000,
        help="Save full model checkpoint every N steps (default: 5000)",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="Compute FID/KID metrics every N steps (default: 5000, 0 to disable)",
    )
    parser.add_argument(
        "--reference_stats",
        type=str,
        default="",
        help="Path to pre-computed reference stats .pt file for FID/KID evaluation",
    )

    # Latent diffusion mode arguments
    parser.add_argument(
        "--latent_mode",
        action="store_true",
        help="Enable latent diffusion mode (32x32 latent space via frozen VAE, 256x256 pixel output)",
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="HuggingFace model name for the frozen VAE (latent mode only)",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-L-14",
        help="OpenCLIP model name for frozen CLIP encoder (latent mode only)",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="OpenCLIP pretrained weights name (latent mode only)",
    )
    parser.add_argument(
        "--init_from_pixel",
        type=str,
        default="",
        help="Path to a pixel-space GLIDE checkpoint for weight transfer to latent model",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()

    # Check for deprecated test_prompt argument
    if args.test_prompt != "a group of skiers are preparing to ski down a mountain.":
        print("ERROR: --test_prompt is deprecated.")
        print(
            "Please use --prompt_file instead to specify a file containing prompts (one per line)."
        )
        print("Example: --prompt_file prompts.txt")
        exit(1)

    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.benchmark = args.cudnn_benchmark

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    if args.use_webdataset:
        # webdataset uses tars - handle glob patterns and braceexpand
        # First expand any brace patterns like {00000..00115}
        expanded_patterns = list(braceexpand(args.data_dir))

        # Then apply glob to each expanded pattern
        data_dir = []
        for pattern in expanded_patterns:
            # If pattern contains wildcards, expand them
            if "*" in pattern or "?" in pattern or "[" in pattern:
                data_dir.extend(glob(pattern))
            # If it's a directory, add *.tar to it
            elif os.path.isdir(pattern):
                data_dir.extend(glob(os.path.join(pattern, "*.tar")))
            # Otherwise assume it's a specific file path
            else:
                if os.path.exists(pattern):
                    data_dir.append(pattern)

        # Sort for consistent ordering
        data_dir = sorted(data_dir)

        if not data_dir:
            raise ValueError(f"No tar files found matching pattern: {args.data_dir}")

        print(f"Found {len(data_dir)} tar files")

        # Clear cache if requested
        if args.clear_validation_cache:
            cache_dir = Path("./cache")
            if data_dir:
                dataset_dir = os.path.dirname(data_dir[0])
                dir_hash = hashlib.md5(dataset_dir.encode()).hexdigest()[:8]
                cache_file = cache_dir / f"valid_tars_{dir_hash}.json"
                if cache_file.exists():
                    cache_file.unlink()
                    print(f"Cleared validation cache: {cache_file}")

        # Validate tar files to exclude corrupted/incomplete ones
        data_dir = validate_tar_files(
            data_dir,
            skip_validation=args.skip_tar_validation,
            use_cache=not args.no_cache_validation,
            verbose=True,
            num_workers=args.validation_workers,
        )

        print(f"Using {len(data_dir)} valid tar files for training")
    else:
        data_dir = args.data_dir

    run_glide_finetune(
        data_dir=data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adam_weight_decay=args.adam_weight_decay,
        side_x=args.side_x,
        side_y=args.side_y,
        resize_ratio=args.resize_ratio,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        use_fp16=args.use_fp16,
        precision=args.precision,
        device=device,
        log_frequency=args.log_frequency,
        sample_interval=args.sample_interval,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        wandb_project_name=args.wandb_project_name,
        activation_checkpointing=args.activation_checkpointing,
        use_captions=args.use_captions,
        num_epochs=args.epochs,
        test_prompt=args.test_prompt,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        use_webdataset=args.use_webdataset,
        image_key=args.wds_image_key,
        caption_key=args.wds_caption_key,
        dataset_name=args.wds_dataset_name,
        enable_upsample=args.train_upsample,
        upsample_factor=args.upscale_factor,
        eval_sr_base_images=args.eval_sr_base_images,
        use_sr_eval=args.use_sr_eval,
        sr_model_path=args.sr_model_path,
        prompt_file=args.prompt_file,
        sample_batch_size=args.sample_batch_size,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_mode=args.lora_target_mode,
        lora_save_steps=args.lora_save_steps,
        lora_resume=args.lora_resume,
        save_checkpoint_interval=args.save_checkpoint_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        random_hflip=args.random_hflip,
        ema_rate=args.ema_rate,
        reinit_transformer=args.reinit_transformer,
        random_init=args.random_init,
        eval_interval=args.eval_interval,
        reference_stats=args.reference_stats,
        captions_jsonl_path=args.wds_captions_jsonl,
        latent_mode=args.latent_mode,
        vae_model=args.vae_model,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        init_from_pixel=args.init_from_pixel,
    )
