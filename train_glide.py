import argparse
import os
import sys
from glob import glob

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
    early_stop=0,
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
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Start wandb logging (disabled for early_stop test runs)
    if early_stop > 0:
        print("Early stopping enabled - disabling wandb logging")

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
    # First load the base model or from OpenAI checkpoint
    initial_checkpoint = (
        "" if resume_ckpt else ""
    )  # Use OpenAI checkpoint if not resuming
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
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    print(f"Trainable parameters: {number_of_trainable_params}")

    # Move model to device before creating optimizer
    glide_model.to(device)

    # Data setup
    print("Loading data...")
    wds_stats = None
    if use_webdataset:
        dataset, wds_stats = glide_wds_loader(
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
        )

    print(f"[Main] Dataset created: {dataset}")

    # Data loader setup
    print("[Main] Creating DataLoader...")
    if use_webdataset:
        # WebDataset needs to be batched before DataLoader
        print(f"[Main] Batching WebDataset with batch_size={batch_size}")
        dataset = dataset.batched(batch_size)
        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=None,  # WebDataset handles batching
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
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
    optimizer = create_optimizer(
        params=[x for x in glide_model.parameters() if x.requires_grad],
        learning_rate=learning_rate,
        weight_decay=adam_weight_decay,
        use_8bit=use_8bit_adam,
    )

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
    next_run = 0 if len(existing_runs) == 0 else existing_runs_int[-1] + 1
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
            gradient_accumualation_steps=1,
            train_upsample=enable_upsample,
            early_stop=early_stop,
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
            wds_stats=wds_stats if use_webdataset else None,
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
        "--early_stop",
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
        early_stop=args.early_stop,
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
    )
