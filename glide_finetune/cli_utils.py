"""
CLI utilities for GLIDE evaluation with Rich UI components.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import torch
from PIL import Image
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box
from rich.text import Text


def get_device(device_arg: Optional[str] = None) -> str:
    """
    Get the device to use for computation.

    Args:
        device_arg: User-specified device or None for auto-detection

    Returns:
        Device string ("cuda" or "cpu")
    """
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_prompts_string(prompts_str: str) -> List[str]:
    """
    Validate and parse prompts from a pipe-separated string.

    Args:
        prompts_str: Pipe-separated string of prompts

    Returns:
        List of prompts

    Raises:
        ValueError: If the number of prompts is not a power of 2
    """
    # Split by pipe and strip whitespace
    prompts = [p.strip() for p in prompts_str.split("|") if p.strip()]

    # Check power of 2
    n = len(prompts)
    if n == 0:
        raise ValueError("No prompts provided")
    if n > 1024:
        raise ValueError(f"Too many prompts: {n} (max 1024)")
    if not (n & (n - 1)) == 0:
        raise ValueError(
            f"Number of prompts must be a power of 2 (1, 2, 4, 8, 16, ...), got {n}"
        )

    return prompts


def validate_prompt_file(prompt_file: str) -> List[str]:
    """
    Validate that the prompt file contains a power-of-2 number of prompts.

    Args:
        prompt_file: Path to the prompt file

    Returns:
        List of prompts

    Raises:
        ValueError: If prompt count is not a power of 2 or exceeds 1024
    """
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    count = len(prompts)
    valid_counts = [
        2**i for i in range(11)
    ]  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

    if count not in valid_counts:
        raise ValueError(
            f"Prompt file must contain a power of 2 prompts (1-1024). "
            f"Found {count} prompts. Valid counts: {valid_counts}"
        )

    return prompts


def setup_output_directory(base_dir: str = "glide-outputs") -> Path:
    """
    Create an auto-incremented output directory.

    Args:
        base_dir: Base directory name

    Returns:
        Path to the created directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Find the next available directory number
    existing_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]

    if existing_dirs:
        next_num = max(int(d.name) for d in existing_dirs) + 1
    else:
        next_num = 0

    # Create directory with 5-digit padding
    output_dir = base_path / f"{next_num:05d}"
    output_dir.mkdir(exist_ok=True)

    return output_dir


def save_metadata(output_dir: Path, config: Dict[str, Any], prompts: List[str]):
    """
    Save generation metadata to JSON file.

    Args:
        output_dir: Output directory
        config: Configuration dictionary
        prompts: List of prompts
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "prompts_count": len(prompts),
        "prompts": prompts,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_checkpoint(output_dir: Path, state: Dict[str, Any]):
    """
    Save checkpoint for resume functionality.

    Args:
        output_dir: Output directory
        state: State dictionary with progress information
    """
    checkpoint_path = output_dir / "checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint if it exists.

    Args:
        output_dir: Output directory

    Returns:
        Checkpoint state or None
    """
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
            return data  # type: ignore
    return None


def should_skip_prompt(
    prompt_idx: int, output_dir: Path, output_format: str, skip_existing: bool
) -> bool:
    """
    Check if a prompt should be skipped.

    Args:
        prompt_idx: Index of the prompt
        output_dir: Output directory
        output_format: Image format (jpg/png)
        skip_existing: Whether to skip existing files

    Returns:
        True if prompt should be skipped
    """
    if not skip_existing:
        return False

    output_path = output_dir / f"prompt_{prompt_idx:03d}.{output_format}"
    return output_path.exists()


def create_progress_bars() -> Progress:
    """
    Create Rich progress bars for the application.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )


def create_device_info_panel(device: str) -> Panel:
    """
    Create a panel showing device information.

    Args:
        device: Device string

    Returns:
        Rich Panel with device info
    """
    info_lines = [f"[bold cyan]Device:[/bold cyan] {device}"]

    if device == "cuda":
        # Add GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info_lines.append(f"[bold cyan]GPU:[/bold cyan] {gpu_name}")
        info_lines.append(f"[bold cyan]VRAM:[/bold cyan] {gpu_memory:.1f} GB")

        # Current memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            info_lines.append(
                f"[bold cyan]Memory Used:[/bold cyan] {allocated:.1f}/{reserved:.1f} GB"
            )

    return Panel(
        "\n".join(info_lines),
        title="Device Information",
        border_style="blue",
        box=box.ROUNDED,
    )


def create_config_panel(args) -> Panel:
    """
    Create a panel showing configuration.

    Args:
        args: Parsed arguments

    Returns:
        Rich Panel with configuration
    """
    config_lines = [
        f"[bold]Base Model:[/bold] {Path(args.base_model).name}",
        f"[bold]SR Model:[/bold] {Path(args.sr_model).name}",
        f"[bold]Sampler:[/bold] {args.sampler}",
        f"[bold]Base Steps:[/bold] {args.base_steps}",
        f"[bold]SR Steps:[/bold] {args.sr_steps}",
        f"[bold]CFG Scale:[/bold] {args.cfg}",
        f"[bold]Batch Size:[/bold] {args.batch_size}",
        f"[bold]Output Format:[/bold] {args.output_format}",
    ]

    if args.seed is not None:
        config_lines.append(f"[bold]Seed:[/bold] {args.seed}")

    if args.use_fp16:
        config_lines.append("[bold]Precision:[/bold] FP16")
    elif args.use_bf16:
        config_lines.append("[bold]Precision:[/bold] BF16")
    else:
        config_lines.append("[bold]Precision:[/bold] FP32")

    if hasattr(args, "use_torch_compile") and args.use_torch_compile:
        config_lines.append("[bold]Torch Compile:[/bold] ✓ Enabled")

    return Panel(
        "\n".join(config_lines),
        title="Configuration",
        border_style="green",
        box=box.ROUNDED,
    )


def create_batch_status_panel(
    current_batch: int,
    total_batches: int,
    current_prompts: List[str],
    images_completed: int,
    total_images: int,
    elapsed_time: float,
) -> Panel:
    """
    Create a status panel for batch processing.

    Args:
        current_batch: Current batch number
        total_batches: Total number of batches
        current_prompts: Prompts in current batch
        images_completed: Number of images completed
        total_images: Total number of images
        elapsed_time: Elapsed time in seconds

    Returns:
        Rich Panel with status
    """
    # Calculate throughput
    throughput = images_completed / elapsed_time if elapsed_time > 0 else 0

    # Truncate prompts for display
    prompt_display = []
    for i, prompt in enumerate(current_prompts[:3]):  # Show max 3 prompts
        truncated = prompt[:50] + "..." if len(prompt) > 50 else prompt
        prompt_display.append(f"  {i + 1}. {truncated}")
    if len(current_prompts) > 3:
        prompt_display.append(f"  ... and {len(current_prompts) - 3} more")

    status_lines = [
        f"[bold yellow]Batch:[/bold yellow] {current_batch}/{total_batches}",
        f"[bold yellow]Images:[/bold yellow] {images_completed}/{total_images}",
        f"[bold yellow]Throughput:[/bold yellow] {throughput:.2f} img/sec",
        "",
        "[bold yellow]Current Prompts:[/bold yellow]",
    ] + prompt_display

    return Panel(
        "\n".join(status_lines),
        title="Processing Status",
        border_style="yellow",
        box=box.ROUNDED,
    )


def create_summary_table(results: List[Dict[str, Any]]) -> Table:
    """
    Create a summary table of generation results.

    Args:
        results: List of result dictionaries

    Returns:
        Rich Table with results
    """
    table = Table(
        title="Generation Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Index", style="dim", width=8)
    table.add_column("Prompt", width=50)
    table.add_column("File", style="cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("Status", justify="center")

    for result in results:
        # Truncate prompt for display
        prompt = result["prompt"]
        truncated = prompt[:47] + "..." if len(prompt) > 50 else prompt

        # Status with color
        if result["status"] == "success":
            status = "[green]✓[/green]"
        elif result["status"] == "skipped":
            status = "[yellow]→[/yellow]"
        else:
            status = "[red]✗[/red]"

        table.add_row(
            str(result["index"]),
            truncated,
            result["filename"],
            f"{result.get('time', 0):.2f}",
            status,
        )

    return table


def create_reranking_progress_panel(
    current_prompt: str,
    num_candidates: int,
    top_k: int,
    clip_model: str,
    phase: str = "generating",
) -> Panel:
    """
    Create a progress panel for CLIP re-ranking.

    Args:
        current_prompt: Current prompt being processed
        num_candidates: Number of candidates to generate
        top_k: Number of top images to keep
        clip_model: CLIP model being used
        phase: Current phase ("generating", "ranking", "saving")

    Returns:
        Rich Panel with re-ranking status
    """
    content = Table(show_header=False, box=None, padding=(0, 1))

    # Add status rows
    content.add_row(
        "[bold]Prompt:[/bold]",
        Text(
            current_prompt[:80] + "..." if len(current_prompt) > 80 else current_prompt
        ),
    )
    content.add_row("[bold]CLIP Model:[/bold]", clip_model)
    content.add_row("[bold]Candidates:[/bold]", f"{num_candidates} images")
    content.add_row("[bold]Selecting:[/bold]", f"Top {top_k} images")

    # Phase indicator
    phase_text = {
        "generating": "[yellow]⚡ Generating candidates...[/yellow]",
        "ranking": "[cyan]🎯 Re-ranking with CLIP...[/cyan]",
        "saving": "[green]💾 Saving best images...[/green]",
    }
    content.add_row("[bold]Status:[/bold]", phase_text.get(phase, phase))

    return Panel(
        content,
        title="[bold blue]CLIP Re-ranking[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
    )


def save_image_text_pair(
    image_tensor: torch.Tensor,
    prompt: str,
    output_dir: Path,
    index: int,
    output_format: str = "jpg",
) -> Tuple[str, str]:
    """
    Save an image-text pair in COCO-style format.

    Args:
        image_tensor: Image tensor from model
        prompt: Text prompt
        output_dir: Output directory
        index: Prompt index
        output_format: Image format (jpg/png)

    Returns:
        Tuple of (image_path, text_path)
    """
    # Convert tensor to PIL image
    image_tensor_uint8 = ((image_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    image_np = image_tensor_uint8.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(image_np)

    # Save files with consistent naming
    base_name = f"prompt_{index:03d}"
    image_path = output_dir / f"{base_name}.{output_format}"
    text_path = output_dir / f"{base_name}.txt"

    # Save image
    if output_format == "jpg":
        image.save(image_path, quality=95)
    else:
        image.save(image_path)

    # Save text
    with open(text_path, "w") as f:
        f.write(prompt)

    return str(image_path), str(text_path)


def save_ranked_images(
    images: List[torch.Tensor],
    prompt: str,
    scores: torch.Tensor,
    output_dir: Path,
    prompt_idx: int,
    output_format: str = "jpg",
) -> List[Dict[str, Any]]:
    """
    Save ranked images with CLIP scores and metadata.

    Args:
        images: List of image tensors (already ranked)
        prompt: Text prompt
        scores: CLIP scores for the images
        output_dir: Output directory
        prompt_idx: Prompt index
        output_format: Image format

    Returns:
        List of metadata dictionaries for each saved image
    """
    # Create prompt-specific directory
    prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Save each ranked image
    for rank, (image_tensor, score) in enumerate(zip(images, scores), 1):
        # Convert tensor to PIL image
        image_tensor_uint8 = ((image_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        image_np = image_tensor_uint8.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image_np)

        # Save with rank in filename
        image_path = prompt_dir / f"image_rank_{rank:03d}.{output_format}"
        if output_format == "jpg":
            image.save(image_path, quality=95)
        else:
            image.save(image_path)

        # Add to results
        results.append(
            {
                "rank": rank,
                "filename": image_path.name,
                "clip_score": float(score),
                "prompt": prompt,
            }
        )

    # Save metadata
    metadata_path = prompt_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "num_images": len(images),
                "images": results,
            },
            f,
            indent=2,
        )

    # Save prompt text file
    prompt_path = prompt_dir / "prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt)

    return results


def create_image_grid(
    images: List[Image.Image], grid_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of PIL images
        grid_size: Optional (rows, cols) tuple, auto-calculated if None

    Returns:
        Grid image
    """
    n = len(images)

    if grid_size is None:
        # Auto-calculate grid size (n is guaranteed to be power of 2)
        if n == 1:
            rows, cols = 1, 1
        else:
            # For power of 2, create square grid or closest rectangle
            sqrt_n = int(math.sqrt(n))
            if sqrt_n * sqrt_n == n:
                rows, cols = sqrt_n, sqrt_n
            else:
                # Find best rectangle
                cols = sqrt_n * 2
                rows = n // cols
    else:
        rows, cols = grid_size

    # Get image dimensions (assume all same size)
    img_width, img_height = images[0].size

    # Create grid
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid = Image.new("RGB", (grid_width, grid_height))

    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    return grid


def display_output_tree(output_dir: Path, console: Console):
    """
    Display the output directory structure as a tree.

    Args:
        output_dir: Output directory
        console: Rich console
    """
    tree = Tree(f"[bold cyan]{output_dir.name}[/bold cyan]")

    # Count files by type
    txt_files = list(output_dir.glob("*.txt"))
    img_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))

    # Add summary
    tree.add(f"[green]{len(txt_files)} text files[/green]")
    tree.add(f"[green]{len(img_files)} image files[/green]")

    if (output_dir / "grid.jpg").exists() or (output_dir / "grid.png").exists():
        tree.add("[yellow]Grid image created[/yellow]")

    if (output_dir / "metadata.json").exists():
        tree.add("[blue]Metadata saved[/blue]")

    console.print(Panel(tree, title="Output Directory", border_style="cyan"))
