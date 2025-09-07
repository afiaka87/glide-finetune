#!/usr/bin/env python3
"""
GLIDE Evaluation CLI Tool

Generates images from text prompts using GLIDE base and super-resolution models.
Outputs image-text pairs in COCO-style format with Rich terminal UI.
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import wandb
from PIL import Image
from rich.console import Console

from glide_finetune.glide_util import load_model, sample_with_superres
from glide_finetune.cli_utils import (
    get_device,
    validate_prompt_file,
    validate_prompts_string,
    setup_output_directory,
    save_metadata,
    save_checkpoint,
    load_checkpoint,
    should_skip_prompt,
    create_progress_bars,
    create_device_info_panel,
    create_config_panel,
    create_batch_status_panel,
    create_summary_table,
    create_reranking_progress_panel,
    save_image_text_pair,
    save_ranked_images,
    create_image_grid,
    display_output_tree,
)
from glide_finetune.clip_rerank import CLIPReranker


def get_compiled_model_path(original_path: str) -> Path:
    """
    Get the path for a compiled model based on the original model path.

    Args:
        original_path: Path to the original model

    Returns:
        Path where compiled model should be saved/loaded
    """
    path = Path(original_path)
    # If the model already has .compiled.pt extension, return as-is
    if path.name.endswith(".compiled.pt"):
        return path

    # Otherwise, create the compiled version path
    stem = path.stem  # filename without extension
    compiled_name = f"{stem}.compiled.pt"
    return path.parent / compiled_name


def load_or_compile_model(
    model_path: str,
    model_type: str,
    use_fp16: bool,
    use_torch_compile: bool,
    device: str,
    console: Console,
):
    """
    Load a model and optionally compile it with torch.compile.
    Uses memoization to avoid recompiling already compiled models.

    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model ("base" or "upsample")
        use_fp16: Whether to use FP16
        use_torch_compile: Whether to compile the model
        device: Device to load model on
        console: Rich console for output

    Returns:
        Tuple of (model, diffusion, options)
    """
    # Check if we should use compiled version
    if use_torch_compile:
        compiled_path = get_compiled_model_path(model_path)

        # If the original path already ends with .compiled.pt, load and compile
        if model_path.endswith(".compiled.pt"):
            console.print(
                f"[cyan]Loading and compiling {model_type} model from {Path(model_path).name}[/cyan]"
            )
            model, diffusion, options = load_model(
                glide_path=model_path,
                use_fp16=use_fp16,
                model_type=model_type,
            )
            model.to(device)
            model.eval()
            # Apply torch.compile
            try:
                compiled_model = torch.compile(model, mode="reduce-overhead")
                return compiled_model, diffusion, options
            except Exception as e:
                console.print(
                    f"[yellow]Warning: torch.compile failed, using uncompiled model: {e}[/yellow]"
                )
                return model, diffusion, options

        # Check if compiled checkpoint exists (contains original weights)
        if compiled_path.exists():
            console.print(
                f"[green]Found cached {model_type} model weights at {compiled_path.name}[/green]"
            )
            # Load from the checkpoint and apply torch.compile
            model, diffusion, options = load_model(
                glide_path=str(compiled_path),
                use_fp16=use_fp16,
                model_type=model_type,
            )
            model.to(device)
            model.eval()

            console.print(
                f"[cyan]Compiling {model_type} model (using cached compilation)...[/cyan]"
            )
            try:
                compiled_model = torch.compile(model, mode="reduce-overhead")
                return compiled_model, diffusion, options
            except Exception as e:
                console.print(
                    f"[yellow]Warning: torch.compile failed, using uncompiled model: {e}[/yellow]"
                )
                return model, diffusion, options

        # Load original model, compile it, and save checkpoint for future use
        console.print(
            f"[yellow]Loading {model_type} model for first-time compilation...[/yellow]"
        )
        model, diffusion, options = load_model(
            glide_path=model_path,
            use_fp16=use_fp16,
            model_type=model_type,
        )
        model.to(device)
        model.eval()

        # Get the original state dict before compilation
        original_state_dict = model.state_dict()

        # Compile the model
        console.print(
            f"[yellow]Compiling {model_type} model (first run, will be slower)...[/yellow]"
        )
        try:
            # Use torch.compile with mode='reduce-overhead' for inference
            compiled_model = torch.compile(model, mode="reduce-overhead")

            # Save the ORIGINAL state dict (not the compiled one) for future loading
            console.print(
                f"[yellow]Caching model weights to {compiled_path.name} for future runs...[/yellow]"
            )
            compiled_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the original state dict so it can be loaded and recompiled later
            torch.save(original_state_dict, compiled_path)

            console.print(
                "[green]✓ Model weights cached for fast compilation on future runs[/green]"
            )

            return compiled_model, diffusion, options

        except Exception as e:
            console.print(
                f"[red]Warning: torch.compile failed for {model_type} model: {e}[/red]"
            )
            console.print("[yellow]Falling back to uncompiled model[/yellow]")
            return model, diffusion, options

    else:
        # Normal loading without compilation
        model, diffusion, options = load_model(
            glide_path=model_path,
            use_fp16=use_fp16,
            model_type=model_type,
        )
        model.to(device)
        model.eval()
        return model, diffusion, options


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GLIDE Evaluation Tool - Generate images from text prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt_file",
        type=str,
        help="Path to file with line-separated prompts (must be power of 2, max 1024)",
    )
    input_group.add_argument(
        "--prompts",
        type=str,
        help="Pipe-separated prompts, e.g., 'cat|dog|bird|fish' (must be power of 2)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model checkpoint (64x64)",
    )
    parser.add_argument(
        "--sr_model",
        type=str,
        required=True,
        help="Path to super-resolution model checkpoint (256x256)",
    )

    # Generation arguments
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "euler_a", "dpm++", "plms", "ddim"],
        help="Sampling method (default: euler)",
    )
    parser.add_argument(
        "--base_steps",
        type=int,
        default=30,
        help="Number of sampling steps for base model (default: 30)",
    )
    parser.add_argument(
        "--sr_steps",
        type=int,
        default=30,
        help="Number of sampling steps for SR model (default: 30)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (default: 4.0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for unique-pair processing (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="glide-outputs",
        help="Base output directory (default: glide-outputs)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save a grid image of all outputs",
    )

    # Performance arguments
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use BF16 mixed precision",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use torch.compile for optimized inference (saves compiled models for reuse)",
    )

    # Execution modes
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without generating images",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if output files already exist",
    )

    # W&B integration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="glide-finetune-eval",
        help="W&B project name (default: glide-finetune-eval)",
    )

    # CLIP re-ranking arguments
    clip_group = parser.add_argument_group("CLIP Re-ranking")
    clip_group.add_argument(
        "--use_clip_rerank",
        action="store_true",
        help="Enable CLIP re-ranking of generated images",
    )
    clip_group.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        help="CLIP model to use for re-ranking (default: ViT-L/14)",
    )
    clip_group.add_argument(
        "--clip_candidates",
        type=int,
        default=32,
        help="Number of candidates to generate per prompt (default: 32)",
    )
    clip_group.add_argument(
        "--clip_top_k",
        type=int,
        default=8,
        help="Number of top images to keep after re-ranking (default: 8)",
    )
    clip_group.add_argument(
        "--clip_batch_size",
        type=int,
        default=16,
        help="Batch size for CLIP scoring (default: 16)",
    )

    return parser.parse_args()


def setup_wandb(args, prompts: List[str], output_dir: Path):
    """Initialize W&B logging."""
    config = {
        "base_model": args.base_model,
        "sr_model": args.sr_model,
        "sampler": args.sampler,
        "base_steps": args.base_steps,
        "sr_steps": args.sr_steps,
        "cfg": args.cfg,
        "batch_size": args.batch_size,
        "device": get_device(args.device),
        "seed": args.seed,
        "num_prompts": len(prompts),
        "output_format": args.output_format,
        "use_fp16": args.use_fp16,
        "use_bf16": args.use_bf16,
        "use_torch_compile": args.use_torch_compile,
    }

    run_name = f"eval_{output_dir.name}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=config,
        tags=["evaluation", args.sampler],
    )

    # Log prompts as a table
    prompt_table = wandb.Table(columns=["index", "prompt"])
    for i, prompt in enumerate(prompts):
        prompt_table.add_data(i, prompt)
    wandb.log({"prompts": prompt_table})


def load_models_with_progress(
    base_model_path: str,
    sr_model_path: str,
    use_fp16: bool,
    use_bf16: bool,
    use_torch_compile: bool,
    device: str,
    console: Console,
):
    """Load models with progress display and optional compilation."""
    with create_progress_bars() as progress:
        # Add loading tasks
        load_task = progress.add_task("[cyan]Loading models...", total=2)

        # Load base model (with compilation if requested)
        progress.update(load_task, description="[cyan]Loading/compiling base model...")
        base_model, base_diffusion, base_options = load_or_compile_model(
            model_path=base_model_path,
            model_type="base",
            use_fp16=use_fp16,
            use_torch_compile=use_torch_compile,
            device=device,
            console=console,
        )
        progress.update(load_task, advance=1)

        # Load SR model (with compilation if requested)
        progress.update(load_task, description="[cyan]Loading/compiling SR model...")
        sr_model, sr_diffusion, sr_options = load_or_compile_model(
            model_path=sr_model_path,
            model_type="upsample",
            use_fp16=use_fp16,
            use_torch_compile=use_torch_compile,
            device=device,
            console=console,
        )
        progress.update(load_task, advance=1)

        # Convert to BF16 if requested
        if use_bf16:
            console.print("[yellow]Converting models to BF16...[/yellow]")
            base_model = base_model.to(dtype=torch.bfloat16)
            sr_model = sr_model.to(dtype=torch.bfloat16)

    return (base_model, base_diffusion, base_options), (
        sr_model,
        sr_diffusion,
        sr_options,
    )


def process_batch(
    prompts: List[str],
    base_model_tuple,
    sr_model_tuple,
    args,
    device: str,
    progress,
    batch_task_id,
) -> List[torch.Tensor]:
    """
    Process a batch of prompts.

    Returns:
        List of generated image tensors
    """
    base_model, base_diffusion, base_options = base_model_tuple
    sr_model, sr_diffusion, sr_options = sr_model_tuple

    # Update progress description
    progress.update(
        batch_task_id, description=f"[green]Generating {len(prompts)} images..."
    )

    # Generate images using sample_with_superres
    # Note: The function handles batching internally
    samples = []

    for prompt in prompts:
        # Generate single image per prompt (unique-pair batch)
        sample = sample_with_superres(
            base_model,
            base_options,
            sr_model,
            sr_options,
            prompt=prompt,
            batch_size=1,  # One image per prompt
            guidance_scale=args.cfg,
            device=device,
            base_respacing=str(args.base_steps),
            upsampler_respacing=str(args.sr_steps),
            sampler=args.sampler,
        )
        samples.append(sample[0])  # Extract single sample
        progress.update(batch_task_id, advance=1)

    return samples


def process_prompt_with_reranking(
    prompt: str,
    prompt_idx: int,
    base_model_tuple,
    sr_model_tuple,
    args,
    device: str,
    output_dir: Path,
    console: Console,
) -> Dict[str, Any]:
    """
    Process a single prompt with CLIP re-ranking.

    Returns:
        Dictionary with results and metadata
    """
    base_model, base_diffusion, base_options = base_model_tuple
    sr_model, sr_diffusion, sr_options = sr_model_tuple

    # Phase 1: Generate candidates
    console.print(
        create_reranking_progress_panel(
            prompt, args.clip_candidates, args.clip_top_k, args.clip_model, "generating"
        )
    )

    candidates = []
    with create_progress_bars() as progress:
        gen_task = progress.add_task(
            f"[cyan]Generating {args.clip_candidates} candidates...",
            total=args.clip_candidates,
        )

        for i in range(args.clip_candidates):
            sample = sample_with_superres(
                base_model,
                base_options,
                sr_model,
                sr_options,
                prompt=prompt,
                batch_size=1,
                guidance_scale=args.cfg,
                device=device,
                base_respacing=str(args.base_steps),
                upsampler_respacing=str(args.sr_steps),
                sampler=args.sampler,
            )
            candidates.append(sample[0])
            progress.update(gen_task, advance=1)

    # Phase 2: Unload GLIDE models and load CLIP
    console.print(
        create_reranking_progress_panel(
            prompt, args.clip_candidates, args.clip_top_k, args.clip_model, "ranking"
        )
    )

    # Move GLIDE models to CPU to free GPU memory
    console.print("[yellow]Offloading GLIDE models to free GPU memory...[/yellow]")
    base_model.cpu()
    sr_model.cpu()
    torch.cuda.empty_cache()

    # Load CLIP and perform re-ranking
    clip_ranker = CLIPReranker(
        model_name=args.clip_model,
        device=device,
        use_fp16=args.use_fp16,
        console=console,
    )

    with clip_ranker:
        # Get top-k indices and scores
        top_indices, top_scores = clip_ranker.rerank_images(
            candidates,
            prompt,
            top_k=args.clip_top_k,
            batch_size=args.clip_batch_size,
            return_scores=True,
        )

    # Get the top images
    top_images = [candidates[idx] for idx in top_indices]

    # Phase 3: Save ranked images
    console.print(
        create_reranking_progress_panel(
            prompt, args.clip_candidates, args.clip_top_k, args.clip_model, "saving"
        )
    )

    results = save_ranked_images(
        top_images,
        prompt,
        top_scores,
        output_dir,
        prompt_idx,
        args.output_format,
    )

    # Reload GLIDE models if needed (will be done in main loop)

    return {
        "prompt": prompt,
        "prompt_idx": prompt_idx,
        "num_candidates": args.clip_candidates,
        "num_selected": len(top_images),
        "clip_scores": top_scores.tolist(),
        "results": results,
    }


def main():
    """Main execution function."""
    console = Console()

    # Parse arguments
    args = parse_arguments()

    # Validate prompts
    try:
        if args.prompt_file:
            prompts = validate_prompt_file(args.prompt_file)
            console.print(f"[green]✓[/green] Loaded {len(prompts)} prompts from file")
        else:
            prompts = validate_prompts_string(args.prompts)
            console.print(
                f"[green]✓[/green] Parsed {len(prompts)} prompts from command line"
            )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Check for mutually exclusive options
    if args.use_fp16 and args.use_bf16:
        console.print("[red]Error:[/red] Cannot use both FP16 and BF16")
        sys.exit(1)

    # Setup device
    device = get_device(args.device)

    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    console.print(f"[green]✓[/green] Output directory: {output_dir}")

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        console.print(f"[green]✓[/green] Random seed set to {args.seed}")

    # Display configuration
    console.print("\n")
    console.print(create_config_panel(args))
    console.print(create_device_info_panel(device))

    # Dry run mode
    if args.dry_run:
        console.print("\n[yellow]DRY RUN MODE - No images will be generated[/yellow]")
        console.print(f"Would process {len(prompts)} prompts")
        console.print(f"Would save to: {output_dir}")
        return

    # Save metadata
    config = vars(args).copy()
    config["device"] = device
    save_metadata(output_dir, config, prompts)

    # Initialize W&B
    setup_wandb(args, prompts, output_dir)

    # Load models
    console.print("\n[cyan]Loading models...[/cyan]")
    base_tuple, sr_tuple = load_models_with_progress(
        args.base_model,
        args.sr_model,
        args.use_fp16,
        args.use_bf16,
        args.use_torch_compile,
        device,
        console,
    )

    # Check for resume
    checkpoint = None
    start_idx = 0
    if args.resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            start_idx = checkpoint.get("last_completed_idx", 0) + 1
            console.print(f"[yellow]Resuming from prompt {start_idx}[/yellow]")

    # Process prompts
    console.print("\n[cyan]Starting generation...[/cyan]\n")

    results = []
    all_images = []
    start_time = time.time()

    # CLIP re-ranking mode - process each prompt individually
    if args.use_clip_rerank:
        console.print("[bold cyan]CLIP Re-ranking Mode[/bold cyan]")
        console.print(f"  • Generating {args.clip_candidates} candidates per prompt")
        console.print(f"  • Selecting top {args.clip_top_k} using {args.clip_model}\n")

        for prompt_idx, prompt in enumerate(prompts):
            if prompt_idx < start_idx:
                continue

            console.print(
                f"\n[bold]Processing prompt {prompt_idx + 1}/{len(prompts)}[/bold]"
            )

            # Reload GLIDE models if they were offloaded
            if prompt_idx > start_idx:
                console.print("[cyan]Reloading GLIDE models...[/cyan]")
                base_tuple[0].to(device)
                sr_tuple[0].to(device)

            try:
                # Process with re-ranking
                result = process_prompt_with_reranking(
                    prompt,
                    prompt_idx,
                    base_tuple,
                    sr_tuple,
                    args,
                    device,
                    output_dir,
                    console,
                )

                results.append(
                    {
                        "index": prompt_idx,
                        "prompt": prompt,
                        "status": "success",
                        "num_candidates": result["num_candidates"],
                        "num_selected": result["num_selected"],
                        "clip_scores": result["clip_scores"],
                    }
                )

                # Log to W&B
                for rank, img_result in enumerate(result["results"], 1):
                    wandb.log(
                        {
                            f"image_{prompt_idx}_rank_{rank}": wandb.Image(
                                output_dir
                                / f"prompt_{prompt_idx:03d}"
                                / img_result["filename"],
                                caption=f"{prompt} (CLIP: {img_result['clip_score']:.3f})",
                            ),
                            "prompt_idx": prompt_idx,
                            "clip_score": img_result["clip_score"],
                        }
                    )

                # Save checkpoint
                save_checkpoint(
                    output_dir,
                    {
                        "last_completed_idx": prompt_idx,
                        "timestamp": time.time(),
                    },
                )

            except Exception as e:
                console.print(f"[red]Error processing prompt {prompt_idx}: {e}[/red]")
                results.append(
                    {
                        "index": prompt_idx,
                        "prompt": prompt,
                        "status": "error",
                    }
                )

        # Ensure models are back on GPU at the end
        base_tuple[0].to(device)
        sr_tuple[0].to(device)

    else:
        # Original batch processing mode
        total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

        with create_progress_bars() as progress:
            # Main progress bar
            main_task = progress.add_task(
                "[bold blue]Overall Progress", total=len(prompts)
            )

            # Batch progress bar
            batch_task = progress.add_task(
                "[green]Current Batch", total=args.batch_size
            )

            # Process in batches
            for batch_idx in range(start_idx, len(prompts), args.batch_size):
                batch_prompts = prompts[batch_idx : batch_idx + args.batch_size]
                current_batch_num = (batch_idx // args.batch_size) + 1

                # Reset batch progress
                progress.reset(batch_task, total=len(batch_prompts))

                # Skip existing if requested
                if args.skip_existing:
                    skip_count = sum(
                        1
                        for i in range(len(batch_prompts))
                        if should_skip_prompt(
                            batch_idx + i,
                            output_dir,
                            args.output_format,
                            args.skip_existing,
                        )
                    )
                    if skip_count == len(batch_prompts):
                        console.print(
                            f"[yellow]Skipping batch {current_batch_num} (all exist)[/yellow]"
                        )
                        progress.update(main_task, advance=len(batch_prompts))
                        continue

                # Update status
                elapsed = time.time() - start_time
                status_panel = create_batch_status_panel(
                    current_batch_num,
                    total_batches,
                    batch_prompts,
                    batch_idx,
                    len(prompts),
                    elapsed,
                )
                console.print(status_panel)

                # Process batch
                try:
                    batch_start = time.time()
                    samples = process_batch(
                        batch_prompts,
                        base_tuple,
                        sr_tuple,
                        args,
                        device,
                        progress,
                        batch_task,
                    )
                    batch_time = time.time() - batch_start

                    # Save outputs
                    for i, (sample, prompt) in enumerate(zip(samples, batch_prompts)):
                        idx = batch_idx + i

                        # Check if should skip
                        if should_skip_prompt(
                            idx, output_dir, args.output_format, args.skip_existing
                        ):
                            results.append(
                                {
                                    "index": idx,
                                    "prompt": prompt,
                                    "filename": f"prompt_{idx:03d}.{args.output_format}",
                                    "status": "skipped",
                                    "time": 0,
                                }
                            )
                            continue

                        # Save image-text pair
                        img_path, txt_path = save_image_text_pair(
                            sample,
                            prompt,
                            output_dir,
                            idx,
                            args.output_format,
                        )

                        # Load image for grid
                        if args.save_grid:
                            all_images.append(Image.open(img_path))

                        # Log to W&B
                        wandb.log(
                            {
                                f"image_{idx}": wandb.Image(img_path, caption=prompt),
                                "batch_idx": batch_idx,
                                "batch_time": batch_time,
                            }
                        )

                        results.append(
                            {
                                "index": idx,
                                "prompt": prompt,
                                "filename": Path(img_path).name,
                                "status": "success",
                                "time": batch_time / len(batch_prompts),
                            }
                        )

                    # Update main progress
                    progress.update(main_task, advance=len(batch_prompts))

                    # Save checkpoint
                    save_checkpoint(
                        output_dir,
                        {
                            "last_completed_idx": batch_idx + len(batch_prompts) - 1,
                            "timestamp": time.time(),
                        },
                    )

                except Exception as e:
                    console.print(f"[red]Error in batch {current_batch_num}: {e}[/red]")
                    for prompt in batch_prompts:
                        results.append(
                            {
                                "index": batch_idx,
                                "prompt": prompt,
                                "filename": "error",
                                "status": "error",
                                "time": 0,
                            }
                        )
                    progress.update(main_task, advance=len(batch_prompts))
                    continue

                # Clear CUDA cache periodically
                if device == "cuda" and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

    # Create and save grid if requested
    if args.save_grid and all_images:
        console.print("\n[cyan]Creating grid image...[/cyan]")
        grid = create_image_grid(all_images)
        grid_path = output_dir / f"grid.{args.output_format}"
        grid.save(grid_path, quality=95 if args.output_format == "jpg" else None)

        # Log grid to W&B
        wandb.log({"grid": wandb.Image(grid_path)})
        console.print(f"[green]✓[/green] Grid saved to {grid_path}")

    # Display summary
    console.print("\n")
    console.print(create_summary_table(results))

    # Display output tree
    display_output_tree(output_dir, console)

    # Final stats
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    console.print("\n[bold green]Generation complete![/bold green]")
    console.print(f"  • Generated: {successful}/{len(prompts)} images")
    console.print(f"  • Total time: {total_time:.1f}s")
    console.print(f"  • Average: {total_time / successful:.2f}s per image")
    console.print(f"  • Output: {output_dir}")

    # Close W&B
    wandb.finish()


if __name__ == "__main__":
    main()
