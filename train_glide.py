import argparse
from glob import glob
import os
from braceexpand import braceexpand

import numpy as np
import torch as th
import torchvision.transforms as T
from tqdm import trange

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model, load_model_with_lora
from glide_finetune.loader import TextImageDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader


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
    use_fp16=False,  # Tends to cause issues,not sure why as the paper states fp16 is stable.
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
    sample_captions_file="eval_captions.txt",
    num_captions_sample=1,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    dataset_name="laion",
    enable_upsample=False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    use_sr_eval=False,
    use_lora=False,
    lora_rank=4,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_mode="attention",
    lora_save_steps=1000,
    lora_resume="",
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Start wandb logging
    wandb_run = wandb_setup(
        batch_size=batch_size,
        side_x=side_x,
        side_y=side_y,
        learning_rate=learning_rate,
        use_fp16=use_fp16,
        device=device,
        data_dir=data_dir,
        base_dir=checkpoints_dir,
        project_name=wandb_project_name,
    )
    print("Wandb setup.")

    # Model setup
    if use_lora:
        glide_model, glide_diffusion, glide_options = load_model_with_lora(
            glide_path=resume_ckpt,
            use_fp16=use_fp16,
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
            use_fp16=use_fp16,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            model_type="base" if not enable_upsample else "upsample",
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
        upsampler_model, _, upsampler_options = load_model(
            glide_path="",  # Use pretrained OpenAI checkpoint
            use_fp16=use_fp16,
            model_type="upsample",
        )
        upsampler_model.eval()
        upsampler_model.to(device)
        print(f"Upsampler loaded with {sum(x.numel() for x in upsampler_model.parameters())} parameters")

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
            dataset_name=dataset_name,  # can be laion, alamy, or simple.
            buffer_size=args.wds_buffer_size,
            initial_prefetch=args.wds_initial_prefetch,
            debug=args.wds_debug,
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

    # Data loader setup
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not use_webdataset,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Quick test to ensure dataloader is set up (without consuming data)
    print("\nDEBUG: Dataloader setup complete")
    print(f"DEBUG: Batch size: {batch_size}")
    print(f"DEBUG: Using webdataset: {use_webdataset}")
    if use_webdataset:
        print(f"DEBUG: Dataset name: {dataset_name}")
        print(f"DEBUG: Number of tar files: {len(data_dir) if isinstance(data_dir, list) else 'N/A'}")

    # Optimizer setup
    optimizer = th.optim.AdamW(
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
        weight_decay=adam_weight_decay,
    )

    if not freeze_transformer: # if we want to train the transformer, we need to backpropagate through the diffusion model.
        glide_model.out.requires_grad_(True)
        glide_model.input_blocks.requires_grad_(True)
        glide_model.middle_block.requires_grad_(True)
        glide_model.output_blocks.requires_grad_(True)


    # Training setup
    outputs_dir = "./outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    existing_runs = [ sub_dir for sub_dir in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, sub_dir))]
    existing_runs_int = []
    for x in existing_runs:
        try:
            existing_runs_int.append(int(x))
        except:
            print("unexpected directory naming scheme")
            #ignore
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
            upsampler_model=upsampler_model,
            upsampler_options=upsampler_options,
            use_sr_eval=use_sr_eval,
            dataloader=dataloader,
            prompt=test_prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            sample_captions_file=sample_captions_file,
            num_captions_sample=num_captions_sample,
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
            gradient_accumualation_steps=1,
            train_upsample=enable_upsample,
            use_lora=use_lora,
            lora_save_steps=lora_save_steps,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
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
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--sample_interval", "-sample_freq", type=int, default=500, 
                        help="Frequency of sampling images for evaluation (defaults to 500)")
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--wandb_project_name", "-wname", type=str, default="glide_finetune", 
                        help="Project name for wandb logging")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
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
        help="Name of the webdataset to use (laion or alamy)",
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor", "-upscale", type=int, default=4, help="Upscale factor for training the upsampling model only"
    )
    parser.add_argument("--image_to_upsample", "-lowres", type=str, default="low_res_face.png")
    parser.add_argument(
        "--use_sr_eval",
        action="store_true",
        help="Use full pipeline (base + superres) for evaluation sampling during training"
    )
    parser.add_argument(
        "--sample_captions_file",
        type=str,
        default="eval_captions.txt",
        help="Path to file containing captions to randomly sample from during evaluation"
    )
    parser.add_argument(
        "--num_captions_sample",
        type=int,
        default=1,
        help="Number of captions to sample and generate images for (should be power of 2 for grid)"
    )
    parser.add_argument(
        "--eval_base_sampler",
        type=str,
        default="euler",
        choices=["standard", "euler", "euler_a", "dpm++"],
        help="Sampler to use for base model evaluation (standard=PLMS/DDIM, euler, euler_a=ancestral, dpm++)"
    )
    parser.add_argument(
        "--eval_sr_sampler",
        type=str,
        default="euler",
        choices=["standard", "euler", "euler_a", "dpm++"],
        help="Sampler to use for super-resolution evaluation (standard=PLMS, euler, euler_a=ancestral, dpm++)"
    )
    parser.add_argument(
        "--eval_base_sampler_steps",
        type=int,
        default=30,
        help="Number of diffusion steps for base model evaluation (default: 30)"
    )
    parser.add_argument(
        "--eval_sr_sampler_steps",
        type=int,
        default=17,
        help="Number of diffusion steps for super-resolution evaluation (default: 17)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers for parallel data loading (default: 4)"
    )
    parser.add_argument(
        "--wds_buffer_size",
        type=int,
        default=1000,
        help="WebDataset shuffle buffer size (default: 1000)"
    )
    parser.add_argument(
        "--wds_initial_prefetch",
        type=int,
        default=10,
        help="WebDataset initial prefetch size (default: 10)"
    )
    parser.add_argument(
        "--wds_debug",
        action="store_true",
        help="Enable debug printing for WebDataset loading"
    )
    
    # LoRA configuration arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA decomposition (default: 4)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA scaling parameter (default: 32)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout for LoRA layers (default: 0.1)"
    )
    parser.add_argument(
        "--lora_target_mode",
        type=str,
        default="attention",
        choices=["attention", "mlp", "all", "minimal"],
        help="Which modules to apply LoRA to (default: attention)"
    )
    parser.add_argument(
        "--lora_save_steps",
        type=int,
        default=1000,
        help="Save LoRA adapter every N steps (default: 1000)"
    )
    parser.add_argument(
        "--lora_resume",
        type=str,
        default="",
        help="Path to resume LoRA adapter from"
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
            if '*' in pattern or '?' in pattern or '[' in pattern:
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
        image_to_upsample=args.image_to_upsample,
        use_sr_eval=args.use_sr_eval,
        sample_captions_file=args.sample_captions_file,
        num_captions_sample=args.num_captions_sample,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_mode=args.lora_target_mode,
        lora_save_steps=args.lora_save_steps,
        lora_resume=args.lora_resume,
    )
