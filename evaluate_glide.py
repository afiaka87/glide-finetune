#!/usr/bin/env python3
"""
GLIDE Evaluation CLI Tool

Generates images from text prompts using GLIDE base and super-resolution models.
Outputs image-text pairs in COCO-style format with Rich terminal UI.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import wandb
from PIL import Image
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from glide_finetune.glide_util import load_model, sample_with_superres
from glide_finetune.train_util import pred_to_pil
from glide_finetune.cli_utils import (
    get_device,
    validate_prompt_file,
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
    save_image_text_pair,
    create_image_grid,
    display_output_tree,
)


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
    if path.name.endswith('.compiled.pt'):
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
        if model_path.endswith('.compiled.pt'):
            console.print(f"[cyan]Loading and compiling {model_type} model from {Path(model_path).name}[/cyan]")
            model, diffusion, options = load_model(
                glide_path=model_path,
                use_fp16=use_fp16,
                model_type=model_type,
            )
            model.to(device)
            model.eval()
            # Apply torch.compile
            try:
                compiled_model = torch.compile(model, mode='reduce-overhead')
                return compiled_model, diffusion, options
            except Exception as e:
                console.print(f"[yellow]Warning: torch.compile failed, using uncompiled model: {e}[/yellow]")
                return model, diffusion, options
        
        # Check if compiled checkpoint exists (contains original weights)
        if compiled_path.exists():
            console.print(f"[green]Found cached {model_type} model weights at {compiled_path.name}[/green]")
            # Load from the checkpoint and apply torch.compile
            model, diffusion, options = load_model(
                glide_path=str(compiled_path),
                use_fp16=use_fp16,
                model_type=model_type,
            )
            model.to(device)
            model.eval()
            
            console.print(f"[cyan]Compiling {model_type} model (using cached compilation)...[/cyan]")
            try:
                compiled_model = torch.compile(model, mode='reduce-overhead')
                return compiled_model, diffusion, options
            except Exception as e:
                console.print(f"[yellow]Warning: torch.compile failed, using uncompiled model: {e}[/yellow]")
                return model, diffusion, options
        
        # Load original model, compile it, and save checkpoint for future use
        console.print(f"[yellow]Loading {model_type} model for first-time compilation...[/yellow]")
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
        console.print(f"[yellow]Compiling {model_type} model (first run, will be slower)...[/yellow]")
        try:
            # Use torch.compile with mode='reduce-overhead' for inference
            compiled_model = torch.compile(model, mode='reduce-overhead')
            
            # Save the ORIGINAL state dict (not the compiled one) for future loading
            console.print(f"[yellow]Caching model weights to {compiled_path.name} for future runs...[/yellow]")
            compiled_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the original state dict so it can be loaded and recompiled later
            torch.save(original_state_dict, compiled_path)
            
            console.print(f"[green]✓ Model weights cached for fast compilation on future runs[/green]")
            
            return compiled_model, diffusion, options
            
        except Exception as e:
            console.print(f"[red]Warning: torch.compile failed for {model_type} model: {e}[/red]")
            console.print(f"[yellow]Falling back to uncompiled model[/yellow]")
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
    
    # Required arguments
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to file with line-separated prompts (must be power of 2, max 1024)",
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
    
    return (base_model, base_diffusion, base_options), (sr_model, sr_diffusion, sr_options)


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
    progress.update(batch_task_id, description=f"[green]Generating {len(prompts)} images...")
    
    # Generate images using sample_with_superres
    # Note: The function handles batching internally
    samples = []
    
    for prompt in prompts:
        # Generate single image per prompt (unique-pair batch)
        sample = sample_with_superres(
            base_model, base_options,
            sr_model, sr_options,
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


def main():
    """Main execution function."""
    console = Console()
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate prompts
    try:
        prompts = validate_prompt_file(args.prompt_file)
        console.print(f"[green]✓[/green] Loaded {len(prompts)} prompts")
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
    config['device'] = device
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
            start_idx = checkpoint.get('last_completed_idx', 0) + 1
            console.print(f"[yellow]Resuming from prompt {start_idx}[/yellow]")
    
    # Process prompts
    console.print("\n[cyan]Starting generation...[/cyan]\n")
    
    results = []
    all_images = []
    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    start_time = time.time()
    
    with create_progress_bars() as progress:
        # Main progress bar
        main_task = progress.add_task(
            "[bold blue]Overall Progress",
            total=len(prompts)
        )
        
        # Batch progress bar
        batch_task = progress.add_task(
            "[green]Current Batch",
            total=args.batch_size
        )
        
        # Process in batches
        for batch_idx in range(start_idx, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + args.batch_size]
            current_batch_num = (batch_idx // args.batch_size) + 1
            
            # Reset batch progress
            progress.reset(batch_task, total=len(batch_prompts))
            
            # Skip existing if requested
            if args.skip_existing:
                skip_count = sum(
                    1 for i in range(len(batch_prompts))
                    if should_skip_prompt(
                        batch_idx + i, output_dir, 
                        args.output_format, args.skip_existing
                    )
                )
                if skip_count == len(batch_prompts):
                    console.print(f"[yellow]Skipping batch {current_batch_num} (all exist)[/yellow]")
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
                elapsed
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
                    if should_skip_prompt(idx, output_dir, args.output_format, args.skip_existing):
                        results.append({
                            'index': idx,
                            'prompt': prompt,
                            'filename': f"prompt_{idx:03d}.{args.output_format}",
                            'status': 'skipped',
                            'time': 0,
                        })
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
                    wandb.log({
                        f"image_{idx}": wandb.Image(img_path, caption=prompt),
                        "batch_idx": batch_idx,
                        "batch_time": batch_time,
                    })
                    
                    results.append({
                        'index': idx,
                        'prompt': prompt,
                        'filename': Path(img_path).name,
                        'status': 'success',
                        'time': batch_time / len(batch_prompts),
                    })
                
                # Update main progress
                progress.update(main_task, advance=len(batch_prompts))
                
                # Save checkpoint
                save_checkpoint(output_dir, {
                    'last_completed_idx': batch_idx + len(batch_prompts) - 1,
                    'timestamp': time.time(),
                })
                
            except Exception as e:
                console.print(f"[red]Error in batch {current_batch_num}: {e}[/red]")
                for prompt in batch_prompts:
                    results.append({
                        'index': batch_idx,
                        'prompt': prompt,
                        'filename': 'error',
                        'status': 'error',
                        'time': 0,
                    })
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
    successful = sum(1 for r in results if r['status'] == 'success')
    console.print(f"\n[bold green]Generation complete![/bold green]")
    console.print(f"  • Generated: {successful}/{len(prompts)} images")
    console.print(f"  • Total time: {total_time:.1f}s")
    console.print(f"  • Average: {total_time/successful:.2f}s per image")
    console.print(f"  • Output: {output_dir}")
    
    # Close W&B
    wandb.finish()


if __name__ == "__main__":
    main()