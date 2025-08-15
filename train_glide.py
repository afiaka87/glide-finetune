import argparse
import os

import numpy as np
import torch as th
from tqdm import trange

# Enable TF32 for faster training on Ampere GPUs (A100, etc.)
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader
from glide_finetune.checkpoint_manager import CheckpointManager


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
    project_name="glide_finetune",
    activation_checkpointing=False,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,  # console logging frequency
    sample_frequency=500,  # image generation frequency
    save_frequency=1000,  # checkpoint save frequency
    test_prompt="a group of skiers are preparing to ski down a mountain.",
    sample_bs=1,
    sample_gs=8.0,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    enable_upsample=False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    wds_dataset_name="laion",
    sampler="plms",
    num_steps=None,
    eta=0.0,
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Determine the run directory
    existing_runs = [sub_dir for sub_dir in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, sub_dir))]
    existing_runs_int = []
    for x in existing_runs:
        try:
            existing_runs_int.append(int(x))
        except:
            pass
    existing_runs_int = sorted(existing_runs_int)
    next_run = 0 if len(existing_runs) == 0 else existing_runs_int[-1] + 1
    current_run_ckpt_dir = os.path.join(checkpoints_dir, str(next_run).zfill(4))
    os.makedirs(current_run_ckpt_dir, exist_ok=True)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(current_run_ckpt_dir, save_frequency=save_frequency)

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
        project_name=project_name,
    )
    print("Wandb setup.")

    # Determine if resume_ckpt is a pretrained model or training checkpoint
    is_pretrained_checkpoint = False
    if resume_ckpt:
        from pathlib import Path
        resume_path = Path(resume_ckpt)
        # Check if it's a single .pt file that's not in our checkpoint directory structure
        if resume_path.suffix == '.pt' and not resume_path.parent.name.isdigit():
            # Likely a pretrained model (not from our training runs)
            is_pretrained_checkpoint = True
    
    # Model setup
    if is_pretrained_checkpoint:
        # Load pretrained model directly
        print(f"Loading pretrained model from {resume_ckpt}")
        glide_model, glide_diffusion, glide_options = load_model(
            glide_path=resume_ckpt,
            use_fp16=use_fp16,
            freeze_transformer=freeze_transformer,
            freeze_diffusion=freeze_diffusion,
            activation_checkpointing=activation_checkpointing,
            model_type="base" if not enable_upsample else "upsample",
        )
    else:
        # Load base model, we'll restore training checkpoint separately
        glide_model, glide_diffusion, glide_options = load_model(
            glide_path="",  # Start with OpenAI's base model
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
            ar_lower=0.5,
            ar_upper=2.0,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_upper=0.0,
            similarity_threshold_lower=0.5,
            words_to_skip=[],
            dataset_name=wds_dataset_name,  # can be laion, alamy, synthetic.
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
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Optimizer setup
    optimizer = th.optim.AdamW(
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
        weight_decay=adam_weight_decay,
    )
    
    # Load checkpoint if resuming from training checkpoint
    start_epoch = 0
    global_step = 0
    if resume_ckpt and not is_pretrained_checkpoint:
        start_epoch, global_step = checkpoint_manager.load_checkpoint(
            resume_ckpt,
            glide_model,
            optimizer
        )
        if global_step > 0:
            # Clean up interrupted files if we successfully resumed
            checkpoint_manager.cleanup_interrupted_files()

    if not freeze_transformer: # if we want to train the transformer, we need to backpropagate through the diffusion model.
        glide_model.out.requires_grad_(True)
        glide_model.input_blocks.requires_grad_(True)
        glide_model.middle_block.requires_grad_(True)
        glide_model.output_blocks.requires_grad_(True)


    # Training setup
    outputs_dir = "./outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    for epoch in trange(start_epoch, num_epochs):
        print(f"Starting epoch {epoch}")
        global_step = run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=test_prompt,  # Keep for backwards compatibility
            test_prompts=None,  # Will use default prompts
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            checkpoints_dir=current_run_ckpt_dir,
            outputs_dir=outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            sample_frequency=sample_frequency,
            checkpoint_manager=checkpoint_manager,
            epoch=epoch,
            global_step=global_step,
            gradient_accumualation_steps=1,
            train_upsample=enable_upsample,
            sampler=sampler,
            num_steps=num_steps,
            eta=eta,
        )
        
        # Check if training was interrupted
        if checkpoint_manager.interrupted:
            checkpoint_manager.save_checkpoint(
                glide_model,
                optimizer,
                epoch,
                global_step,
                is_interrupted=True
            )
            print("\n✅ Interrupted checkpoint saved. Exiting...")
            break


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
    parser.add_argument("--log_frequency", "-freq", type=int, default=100, help="Console logging frequency (loss, metrics)")
    parser.add_argument("--sample_frequency", "-sample_freq", type=int, default=500, help="Image generation frequency")
    parser.add_argument("--save_frequency", "-save_freq", type=int, default=1000, help="Checkpoint save frequency in steps")
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
        "--sampler",
        "-sampler",
        type=str,
        default="plms",
        choices=["plms", "ddim", "euler", "euler_a", "dpm++"],
        help="Sampling method to use"
    )
    parser.add_argument(
        "--num_steps",
        "-steps",
        type=int,
        default=None,
        help="Number of sampling steps (overrides prediction_respacing)"
    )
    parser.add_argument(
        "--eta",
        "-eta",
        type=float,
        default=0.0,
        help="Eta parameter for DDIM and Euler Ancestral sampling"
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
        # webdataset uses tars - data_dir should be a braceexpand pattern or list of tar files
        data_dir = args.data_dir
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
        sample_frequency=args.sample_frequency,
        save_frequency=args.save_frequency,
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
        wds_dataset_name=args.wds_dataset_name,
        sampler=args.sampler,
        num_steps=args.num_steps,
        eta=args.eta,
    )
