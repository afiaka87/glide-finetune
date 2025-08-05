import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch as th
from tqdm import trange

from glide_finetune.checkpoint_utils import CheckpointManager
from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset
from glide_finetune.optimizer_util import create_optimizer
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader

# Constants
UNCOND_LENGTH = 0


def load_eval_prompts(filepath):
    """Load and validate evaluation prompts from file.

    Args:
        filepath: Path to file containing line-separated prompts

    Returns:
        List of prompts if valid, None if file not provided

    Raises:
        ValueError: If prompt count is not a power of 2 or exceeds 32
    """
    if filepath is None:
        return None

    with open(filepath, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Check if count is power of 2 and <= 32
    valid_counts = [2, 4, 8, 16, 32]
    if len(prompts) not in valid_counts:
        valid_counts_str = ", ".join(map(str, valid_counts))
        raise ValueError(
            f"Evaluation prompts file must contain exactly {valid_counts_str} prompts. "
            f"Found {len(prompts)} prompts."
        )

    return prompts


def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    adam_weight_decay=0.0,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,  # Tends to cause issues, not sure why as paper states fp16 stable
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    project_name="glide_finetune",
    activation_checkpointing=False,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    sample_interval=1000,
    test_prompt="a group of skiers are preparing to ski down a mountain.",
    sample_bs=1,
    sample_gs=8.0,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    enable_upsample=False,
    upsample_factor=4,
    image_to_upsample="low_res_face.png",
    use_8bit_adam=False,
    test_run=0,
    wds_dataset_name="laion",
    sampler_name="plms",
    test_steps=100,
    laion_no_filter=False,
    warmup_steps=0,
    warmup_type="linear",
    eval_prompts_file=None,
    torch_compile=False,
    compile_mode="default",
    use_esrgan=False,
    esrgan_cache_dir="./esrgan_models",
    # CLIP adapter parameters
    use_clip=False,
    clip_model_name="ViT-L/14",
    clip_gate_init=1e-4,  # Small non-zero to ensure gradient flow
    adapter_warmup_steps=10000,
    adapter_lr=1e-5,
    adapter_wd=1e-2,
    adapter_beta2=0.98,
    adapter_training_phase="adapter_only",
    use_lora=False,
    lora_rank=32,
    adapter_dropout=0.1,
    stability_threshold=10.0,
    clip_cache_embeddings=False,
    use_clip_cache=False,
    clip_cache_dir=None,
    adapter_grad_clip=1.0,
    main_grad_clip=1.0,
    dry_run_interval=0,
    dry_run_samples=5,
    kl_loss_interval=100,
    kl_loss_weight=0.01,
    early_stop_threshold=0.1,
    early_stop_patience=1000,
    baseline_eval_interval=500,
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Start wandb logging (disabled for test runs)
    if test_run > 0:
        print("Test run mode enabled - disabling wandb logging")

        # Create a mock wandb run object that does nothing
        class MockWandbRun:
            def log(self, *args, **kwargs):
                pass

        wandb_run = MockWandbRun()
    else:
        wandb_run = wandb_setup(
            batch_size=batch_size,
            side_x=side_x,
            side_y=side_y,
            learning_rate=learning_rate,
            use_fp16=use_fp16,
            device=device,
            data_dir=data_dir,
            base_dir=checkpoints_dir,
            project_name=project_name,
        )
        print("Wandb setup.")

    # Model setup
    # Use resume checkpoint if provided, otherwise use OpenAI checkpoint
    initial_checkpoint = resume_ckpt if resume_ckpt else None

    if use_clip:
        # Load CLIP-enabled model
        from glide_finetune.adapters import load_glide_model_with_clip

        glide_model, glide_diffusion, glide_options = load_glide_model_with_clip(
            glide_path=initial_checkpoint,
            use_fp16=use_fp16,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            model_type="base" if not enable_upsample else "upsample",
            clip_model_name=clip_model_name,
            use_clip=True,
            clip_gate_init=clip_gate_init,
            adapter_dropout=adapter_dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )
        print(f"Loaded CLIP-enabled model with {clip_model_name}")
    else:
        # Load standard GLIDE model
        glide_model, glide_diffusion, glide_options = load_model(
            glide_path=initial_checkpoint,
            use_fp16=use_fp16,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            model_type="base" if not enable_upsample else "upsample",
            torch_compile=torch_compile,
            compile_mode=compile_mode,
        )
    glide_model.train()
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    print(f"Number of parameters: {number_of_params}")

    # Move model to device before creating optimizer
    glide_model.to(device)

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
            dataset_name=wds_dataset_name,
            laion_no_filter=laion_no_filter,
            use_clip_cache=use_clip_cache,
            clip_cache_dir=clip_cache_dir,
            clip_model_name=clip_model_name if use_clip else None,
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
            use_clip_cache=use_clip_cache,
            clip_model_name=clip_model_name if use_clip else None,
        )

    print(f"[Main] Dataset created: {dataset}")

    # Data loader setup
    print("[Main] Creating DataLoader...")
    if use_webdataset:
        # WebDataset needs to be batched before DataLoader
        print(f"[Main] Batching WebDataset with batch_size={batch_size}")
        
        # Custom collation function for WebDataset batching
        def custom_webdataset_collate(samples):
            """Custom collation that properly handles CLIP embeddings."""
            if not samples:
                return []
            
            # Debug logging disabled for performance
            
            # Transpose list of tuples to tuple of lists
            transposed = list(zip(*samples))
            
            # Stack tensors for tokens, masks, and images
            tokens = th.stack(transposed[0])
            masks = th.stack(transposed[1])
            images = th.stack(transposed[2])
            
            # Handle CLIP embeddings (4th element if present)
            if len(transposed) > 3:
                clip_embeddings = transposed[3]
                # Check if all are None
                if all(x is None for x in clip_embeddings):
                    return (tokens, masks, images, None)
                # Stack non-None embeddings (all tensors)
                elif all(x is not None and isinstance(x, th.Tensor) for x in clip_embeddings):
                    clip_tensor = th.stack(clip_embeddings)
                    return (tokens, masks, images, clip_tensor)
                # Stack non-None embeddings (all numpy arrays)
                elif all(x is not None and isinstance(x, np.ndarray) for x in clip_embeddings):
                    # Convert all numpy arrays to tensors
                    device = tokens.device
                    clip_tensors = [th.from_numpy(x).to(device=device, dtype=th.float32) for x in clip_embeddings]
                    clip_tensor = th.stack(clip_tensors)
                    return (tokens, masks, images, clip_tensor)
                else:
                    # Mixed None and tensor - this is actually common with cache misses
                    # Create a zero tensor for missing embeddings
                    print(f"WARNING: Mixed None and tensor CLIP embeddings in batch (cache misses)")
                    # Find the first non-None tensor to get the shape
                    clip_dim = None
                    for clip in clip_embeddings:
                        if clip is not None:
                            if isinstance(clip, th.Tensor):
                                clip_dim = clip.shape[-1]
                                device = clip.device
                                dtype = clip.dtype
                                break
                            elif isinstance(clip, np.ndarray):
                                clip_dim = clip.shape[-1]
                                device = tokens.device  # Use same device as tokens
                                dtype = th.float32
                                break
                    
                    if clip_dim is not None:
                        # Replace None with zero tensors
                        fixed_embeddings = []
                        for clip in clip_embeddings:
                            if clip is None:
                                # Create zero tensor with same shape as others
                                fixed_embeddings.append(th.zeros(clip_dim, device=device, dtype=dtype))
                            elif isinstance(clip, np.ndarray):
                                # Convert numpy array to tensor
                                fixed_embeddings.append(th.from_numpy(clip).to(device=device, dtype=dtype))
                            else:
                                fixed_embeddings.append(clip)
                        clip_tensor = th.stack(fixed_embeddings)
                        return (tokens, masks, images, clip_tensor)
                    else:
                        # All must be None (shouldn't happen based on our check)
                        return (tokens, masks, images, None)
            
            return (tokens, masks, images)
        
        dataset = dataset.batched(batch_size, collation_fn=custom_webdataset_collate)

        # Define collate function to properly stack batched samples
        def collate_webdataset_batch(batch):
            # Since we're using WebDataset's batched() with custom collation,
            # the data should already be properly batched.
            # DataLoader with batch_size=None should just pass through the batch.
            
            # Debug logging disabled for performance
            
            # Check if it's a single-element list containing our batch
            if isinstance(batch, list) and len(batch) == 1:
                inner = batch[0]
                # Debug logging disabled
                if isinstance(inner, (list, tuple)) and len(inner) in [3, 4]:
                    # Check if this looks like our expected batch format
                    if all(isinstance(x, (th.Tensor, type(None))) for x in inner[:3]):
                        # Verify first 3 are tensors
                        if all(isinstance(inner[i], th.Tensor) for i in range(3)):
                            return inner
            
            # If the batch itself is a tuple of the right length, return it directly
            if isinstance(batch, tuple) and len(batch) in [3, 4]:
                if all(isinstance(x, (th.Tensor, type(None))) for x in batch[:3]):
                    if all(isinstance(batch[i], th.Tensor) for i in range(3)):
                        # Batch is already in correct format
                        return batch
            
            # Otherwise, something unexpected happened
            print("ERROR: Unexpected batch format in DataLoader")
            print(f"  batch type: {type(batch)}")
            print(f"  batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                print(f"  first element type: {type(batch[0])}")
                if hasattr(batch[0], 'shape'):
                    print(f"  first element shape: {batch[0].shape}")
            raise ValueError("Unexpected batch format - WebDataset batching may have failed")

        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=None,  # WebDataset handles batching
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
            collate_fn=collate_webdataset_batch,
        )
    else:
        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )
    print("[Main] DataLoader created")

    # Optimizer setup
    if use_clip:
        # Create separate optimizers for CLIP adapter training
        from glide_finetune.adapters import create_clip_adapter_optimizer

        optimizer, optimizer_info = create_clip_adapter_optimizer(
            glide_model,
            adapter_lr=adapter_lr,
            adapter_wd=adapter_wd,
            adapter_beta2=adapter_beta2,
            main_lr=learning_rate if adapter_training_phase == "full" else None,
            main_wd=adam_weight_decay,
            train_phases=adapter_training_phase,
        )

        print(
            f"Created CLIP adapter optimizer with training phase: {adapter_training_phase}"
        )
        for name, count in optimizer_info["param_counts"].items():
            print(f"  {name} parameters: {count:,}")

        # Print trainable parameters after freezing
        trainable_params = sum(
            p.numel() for p in glide_model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters after freezing: {trainable_params:,}")
    else:
        # Standard optimizer
        optimizer = create_optimizer(
            params=[x for x in glide_model.parameters() if x.requires_grad],
            learning_rate=learning_rate,
            weight_decay=adam_weight_decay,
            use_8bit=use_8bit_adam,
        )

    # Setup CLIP adapter trainer if using CLIP
    clip_trainer = None
    if use_clip:
        from glide_finetune.adapters import ClipAdapterTrainer

        clip_trainer = ClipAdapterTrainer(
            model=glide_model,
            diffusion=glide_diffusion,
            optimizer=optimizer,
            warmup_steps=adapter_warmup_steps,
            stability_threshold=stability_threshold,
            checkpoint_dir=os.path.join(checkpoints_dir, "clip_adapter_checkpoints"),
            adapter_grad_clip=adapter_grad_clip,
            main_grad_clip=main_grad_clip,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            baseline_eval_interval=baseline_eval_interval,
        )

        # Note: dry run parameters are handled separately in training loop
        print(f"Created CLIP adapter trainer with {adapter_warmup_steps} warmup steps")

    # Training setup - create run-specific output directory
    training_base_dir = "./outputs/training"
    os.makedirs(training_base_dir, exist_ok=True)

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
    next_run = 0 if len(existing_runs_int) == 0 else existing_runs_int[-1] + 1
    run_id = str(next_run).zfill(4)
    current_run_ckpt_dir = os.path.join(checkpoints_dir, run_id)
    current_run_outputs_dir = os.path.join(training_base_dir, run_id)

    os.makedirs(current_run_ckpt_dir, exist_ok=True)
    os.makedirs(current_run_outputs_dir, exist_ok=True)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(current_run_ckpt_dir)

    # Initialize training state
    start_epoch = 0
    global_step_counter = 0

    # Resume from checkpoint if provided
    if resume_ckpt:
        print(f"\nResuming from checkpoint: {resume_ckpt}")
        resume_state = checkpoint_manager.load_checkpoint(
            checkpoint_path=resume_ckpt,
            model=glide_model,
            optimizer=optimizer,
        )

        if resume_state["has_optimizer_state"] and resume_state["has_metadata"]:
            # Full resume - continue from next epoch to keep things simple
            # TODO: Add support for resuming within an epoch
            start_epoch = resume_state["epoch"] + 1  # Start from next epoch
            global_step_counter = resume_state["global_step"] + 1
            warmup_steps = resume_state.get("warmup_steps", warmup_steps)
            warmup_type = resume_state.get("warmup_type", warmup_type)
            learning_rate = resume_state.get("base_lr", learning_rate)
            resume_msg = (
                f"Resuming training from epoch {start_epoch} "
                f"(continuing after completed epoch {resume_state['epoch']})"
            )
            print(resume_msg)
            
            # Check if training will actually happen
            if start_epoch >= num_epochs:
                print("=" * 50)
                print(f"WARNING: No training will occur!")
                print(f"Checkpoint epoch: {resume_state['epoch']} (completed)")
                print(f"Resume epoch: {start_epoch}")
                print(f"Total epochs requested: {num_epochs}")
                print(f"Training would run from epoch {start_epoch} to {num_epochs-1}")
                print(f"Solution: Increase --epochs to at least {start_epoch + 1}")
                print("=" * 50)
                sys.exit(1)
            else:
                print(f"Training will run for epochs {start_epoch} to {num_epochs-1}")
                print(f"Total epochs to train: {num_epochs - start_epoch}")
        else:
            # Model-only checkpoint - start fresh training
            print("Model weights loaded, starting fresh training")

    # Load evaluation prompts if provided
    eval_prompts = load_eval_prompts(eval_prompts_file)
    if eval_prompts:
        print(f"Loaded {len(eval_prompts)} evaluation prompts from {eval_prompts_file}")

    # Check for conflicting options
    if eval_prompts and len(test_prompt) > UNCOND_LENGTH:
        print("Error: Both --test_prompt and --eval_prompts_file were specified.")
        print("Please use only one of these options:")
        print("  --test_prompt: For evaluating with a single prompt")
        print("  --eval_prompts_file: For evaluating with multiple prompts in a grid")
        sys.exit(1)

    # Calculate steps per epoch for warmup
    # WebDataset doesn't have a length, so we'll track steps during training
    steps_per_epoch = None
    if not use_webdataset:
        steps_per_epoch = len(dataloader)

    # Pre-training summary
    print("=" * 50)
    print("Training Configuration Summary:")
    print(f"Start epoch: {start_epoch}")
    print(f"End epoch: {num_epochs - 1}")
    print(f"Total epochs to train: {num_epochs - start_epoch}")
    print(f"Global step counter: {global_step_counter}")
    print("=" * 50)
    
    for epoch in trange(start_epoch, num_epochs):
        print(f"Starting epoch {epoch}")

        steps_taken = run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=test_prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            checkpoints_dir=current_run_ckpt_dir,
            outputs_dir=current_run_outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            sample_interval=sample_interval,
            epoch=epoch,
            gradient_accumualation_steps=args.gradient_accumulation_steps,
            train_upsample=enable_upsample,
            early_stop=test_run,
            clip_trainer=clip_trainer,
            sampler_name=sampler_name,
            test_steps=test_steps,
            warmup_steps=warmup_steps,
            warmup_type=warmup_type,
            base_lr=learning_rate,
            epoch_offset=global_step_counter
            if use_webdataset
            else epoch * (steps_per_epoch or 0),
            batch_size=batch_size,
            checkpoint_manager=checkpoint_manager,
            eval_prompts=eval_prompts,
            use_esrgan=use_esrgan,
            esrgan_cache_dir=esrgan_cache_dir,
            kl_loss_interval=kl_loss_interval,
            kl_loss_weight=kl_loss_weight,
        )

        # Update global step counter for WebDataset
        if use_webdataset:
            global_step_counter += steps_taken


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler (0 = no warmup)",
    )
    parser.add_argument(
        "--warmup_type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Type of warmup schedule",
    )
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument(
        "--resize_ratio", "-crop", type=float, default=0.8, help="Crop ratio"
    )
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using empty/unconditional token instead of caption. "
        "OpenAI used 0.2 for their finetune.",
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
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1000,
        help="How often to generate sample images (default: 1000 steps)",
    )
    parser.add_argument(
        "--use_esrgan",
        action="store_true",
        help="Use ESRGAN to upsample training samples from 64x64 to 256x256",
    )
    parser.add_argument(
        "--esrgan_cache_dir",
        type=str,
        default="./esrgan_models",
        help="Directory to cache ESRGAN model weights",
    )
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="",
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        default=None,
        help=(
            "File with line-separated prompts for evaluation "
            "(must have 2,4,8,16, or 32 lines)"
        ),
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
        help="Name of webdataset to use (laion, alamy, or webdataset for no filtering)",
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
        "--image_to_upsample", "-lowres", type=str, default="low_res_face.png"
    )
    parser.add_argument(
        "--use_8bit_adam",
        "-8bit",
        action="store_true",
        help="Use 8-bit AdamW optimizer to save memory (requires bitsandbytes)",
    )
    parser.add_argument(
        "--use_tf32",
        "-tf32",
        action="store_true",
        help="Enable TF32 on Ampere GPUs for faster training with reduced precision",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile for optimized model execution (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode: default/reduce-overhead/max-autotune",
    )
    parser.add_argument(
        "--test_run",
        type=int,
        default=0,
        help="Stop training after this many steps (0 = disabled). Useful for testing.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="plms",
        choices=["plms", "ddim", "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras"],
        help="Sampler to use for generating test images during training. "
        "Options: "
        "plms (default) - stable, original GLIDE sampler; "
        "ddim - deterministic when eta=0, good for reproducibility; "
        "euler - fast first-order solver, good quality; "
        "euler_a - euler with added noise, more variation but non-convergent; "
        "dpm++_2m - second-order solver, good quality/speed balance; "
        "dpm++_2m_karras - dpm++_2m with improved schedule for low step counts",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=100,
        help="Number of sampling steps for test image generation (default: 100)",
    )
    parser.add_argument(
        "--laion_no_filter",
        action="store_true",
        help="Skip LAION metadata filtering (faster loading, no metadata requirements)",
    )

    # CLIP Adapter arguments
    parser.add_argument(
        "--use_clip",
        action="store_true",
        help="Enable CLIP adapter for dual text conditioning",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-L/14",
        choices=[
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "ViT-L/14@336px",
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "RN50x64",
        ],
        help="CLIP model architecture to use (default: ViT-L/14)",
    )
    parser.add_argument(
        "--clip_gate_init",
        type=float,
        default=1e-4,
        help="Initial value for CLIP adapter gates (1e-4 = tiny CLIP influence for gradient flow)",
    )
    parser.add_argument(
        "--adapter_warmup_steps",
        type=int,
        default=10000,
        help="Number of steps to gradually increase CLIP adapter influence from 0 to 0.5",
    )
    parser.add_argument(
        "--adapter_lr",
        type=float,
        default=1e-5,
        help="Learning rate for CLIP adapter components (default: 1e-5, 100x smaller than main LR)",
    )
    parser.add_argument(
        "--adapter_wd",
        type=float,
        default=1e-2,
        help="Weight decay for CLIP adapter components",
    )
    parser.add_argument(
        "--adapter_beta2",
        type=float,
        default=0.98,
        help="Adam beta2 for adapter optimizer (default: 0.98 for better stability)",
    )
    parser.add_argument(
        "--adapter_training_phase",
        type=str,
        default="adapter_only",
        choices=["adapter_only", "adapter_gates", "full"],
        help="Training phase: adapter_only (safest), adapter_gates, or full fine-tuning",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA instead of full MLP in CLIP adapter (more parameter efficient)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="Rank for LoRA decomposition in adapter",
    )
    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.1,
        help="Dropout rate in CLIP adapter MLP",
    )
    parser.add_argument(
        "--stability_threshold",
        type=float,
        default=10.0,
        help="Loss spike threshold for automatic checkpoint rollback",
    )
    parser.add_argument(
        "--clip_cache_embeddings",
        action="store_true",
        help="Cache CLIP embeddings to speed up training",
    )
    parser.add_argument(
        "--use_clip_cache",
        action="store_true",
        help="Use pre-computed CLIP embeddings from cache during training",
    )
    parser.add_argument(
        "--clip_cache_dir",
        type=str,
        default=None,
        help="Directory containing pre-computed CLIP embeddings (defaults to data_dir/clip_cache)",
    )
    parser.add_argument(
        "--adapter_grad_clip",
        type=float,
        default=1.0,
        help="Max gradient norm for CLIP adapter parameters (default: 1.0)",
    )
    parser.add_argument(
        "--main_grad_clip",
        type=float,
        default=1.0,
        help="Max gradient norm for main model parameters (default: 1.0)",
    )
    parser.add_argument(
        "--dry_run_interval",
        type=int,
        default=0,
        help="Run dry-run test every N steps to verify adapter doesn't affect outputs (0 = disabled)",
    )
    parser.add_argument(
        "--dry_run_samples",
        type=int,
        default=5,
        help="Number of samples to test in each dry-run (default: 5)",
    )
    parser.add_argument(
        "--kl_loss_interval",
        type=int,
        default=100,
        help="Compute KL divergence loss every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--kl_loss_weight",
        type=float,
        default=0.01,
        help="Weight for KL divergence regularization loss (default: 0.01)",
    )
    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=0.1,
        help="Max allowed degradation in pretrained performance before early stopping (0.1 = 10 percent)",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=1000,
        help="Steps to wait before early stopping after degradation detected",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps to simulate larger batch sizes (default: 1)",
    )
    parser.add_argument(
        "--baseline_eval_interval",
        type=int,
        default=500,
        help="How often to evaluate pretrained performance for early stopping",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.benchmark = args.cudnn_benchmark

    # Enable TF32 on Ampere GPUs
    if args.use_tf32:
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matrix multiplications and cuDNN operations")

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    # Validate CLIP cache settings if specified
    if args.use_clip_cache and args.clip_cache_dir:
        clip_cache_path = Path(args.clip_cache_dir)
        if not clip_cache_path.exists():
            print(
                f"\nError: CLIP cache directory '{args.clip_cache_dir}' does not exist.\n"
                f"Please either:\n"
                f"1. Run the precompute scripts first to generate the cache:\n"
                f"   uv run python scripts/precompute_clip_webdataset_embeddings.py \\\n"
                f"     --tar_urls '{args.data_dir}/*.tar' \\\n"
                f"     --cache_dir '{args.clip_cache_dir}' \\\n"
                f"     --clip_model_name '{args.clip_model_name}'\n"
                f"2. Disable CLIP cache by removing --use_clip_cache flag\n"
                f"3. Use a different cache directory with --clip_cache_dir\n"
            )
            sys.exit(1)
        
        # Check for model-specific directory
        model_dir = clip_cache_path / args.clip_model_name.replace("/", "-")
        if not model_dir.exists():
            available_models = [d.name for d in clip_cache_path.iterdir() if d.is_dir()]
            print(
                f"\nError: No cache found for CLIP model '{args.clip_model_name}' in '{args.clip_cache_dir}'.\n"
                f"Available models: {available_models}\n"
                f"Please run precompute scripts with --clip_model_name '{args.clip_model_name}'\n"
            )
            sys.exit(1)

    if args.use_webdataset:
        # webdataset uses tars
        data_dir = glob(os.path.join(args.data_dir, "*.tar"))
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
        device=device,
        log_frequency=args.log_frequency,
        sample_interval=args.sample_interval,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,
        use_captions=args.use_captions,
        num_epochs=args.epochs,
        test_prompt=args.test_prompt,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        use_webdataset=args.use_webdataset,
        image_key=args.wds_image_key,
        caption_key=args.wds_caption_key,
        enable_upsample=args.train_upsample,
        upsample_factor=args.upscale_factor,
        image_to_upsample=args.image_to_upsample,
        use_8bit_adam=args.use_8bit_adam,
        test_run=args.test_run,
        wds_dataset_name=args.wds_dataset_name,
        sampler_name=args.sampler,
        test_steps=args.test_steps,
        laion_no_filter=args.laion_no_filter,
        warmup_steps=args.warmup_steps,
        warmup_type=args.warmup_type,
        eval_prompts_file=args.eval_prompts_file,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        use_esrgan=args.use_esrgan,
        esrgan_cache_dir=args.esrgan_cache_dir,
        # CLIP adapter parameters
        use_clip=args.use_clip,
        clip_model_name=args.clip_model_name,
        clip_gate_init=args.clip_gate_init,
        adapter_warmup_steps=args.adapter_warmup_steps,
        adapter_lr=args.adapter_lr,
        adapter_wd=args.adapter_wd,
        adapter_beta2=args.adapter_beta2,
        adapter_training_phase=args.adapter_training_phase,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        adapter_dropout=args.adapter_dropout,
        stability_threshold=args.stability_threshold,
        clip_cache_embeddings=args.clip_cache_embeddings,
        use_clip_cache=args.use_clip_cache,
        clip_cache_dir=args.clip_cache_dir,
        adapter_grad_clip=args.adapter_grad_clip,
        main_grad_clip=args.main_grad_clip,
        dry_run_interval=args.dry_run_interval,
        dry_run_samples=args.dry_run_samples,
        kl_loss_interval=args.kl_loss_interval,
        kl_loss_weight=args.kl_loss_weight,
        early_stop_threshold=args.early_stop_threshold,
        early_stop_patience=args.early_stop_patience,
        baseline_eval_interval=args.baseline_eval_interval,
    )
