#!/usr/bin/env python3
"""
GLIDE Gradio Web Interface

Interactive web application for text-to-image generation using GLIDE models.
"""

import os
import time
import random
from typing import Optional, Tuple, List, Any
import gc

import gradio as gr
import numpy as np
import torch
from PIL import Image

from glide_finetune.glide_util import load_model, sample_with_superres


# Global variables for model caching
cached_models = {
    "base": None,
    "sr": None,
    "current_paths": {"base": None, "sr": None},
    "device": None,
    "use_fp16": None,
    "use_torch_compile": None,
}


def get_device() -> str:
    """Auto-detect the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_memory_info() -> str:
    """Get GPU memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU Memory: {allocated:.1f}/{total:.1f} GB used"
    return "Running on CPU"


def load_models_if_needed(
    base_model_path: str,
    sr_model_path: str,
    device: str,
    use_fp16: bool,
    use_torch_compile: bool,
    progress=gr.Progress(),
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Load models only if paths have changed or models aren't loaded.
    Returns: (base_model, base_diffusion, base_options, sr_model, sr_diffusion, sr_options)
    """
    global cached_models

    # Check if we need to reload models
    need_reload = (
        cached_models["base"] is None
        or cached_models["sr"] is None
        or cached_models["current_paths"]["base"] != base_model_path
        or cached_models["current_paths"]["sr"] != sr_model_path
        or cached_models["device"] != device
        or cached_models["use_fp16"] != use_fp16
        or cached_models["use_torch_compile"] != use_torch_compile
    )

    if not need_reload:
        return (
            cached_models["base"]["model"],
            cached_models["base"]["diffusion"],
            cached_models["base"]["options"],
            cached_models["sr"]["model"],
            cached_models["sr"]["diffusion"],
            cached_models["sr"]["options"],
        )

    # Clear previous models if loaded
    if cached_models["base"] is not None:
        del cached_models["base"]
    if cached_models["sr"] is not None:
        del cached_models["sr"]
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    progress(0.0, desc="Loading base model...")

    # Load base model
    base_model, base_diffusion, base_options = load_model(
        glide_path=base_model_path,
        use_fp16=use_fp16,
        model_type="base",
    )
    base_model.to(device)
    base_model.eval()

    # Apply torch.compile if requested
    if use_torch_compile:
        try:
            progress(0.25, desc="Compiling base model...")
            base_model = torch.compile(base_model, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: torch.compile failed for base model: {e}")

    progress(0.5, desc="Loading SR model...")

    # Load SR model
    sr_model, sr_diffusion, sr_options = load_model(
        glide_path=sr_model_path,
        use_fp16=use_fp16,
        model_type="upsample",
    )
    sr_model.to(device)
    sr_model.eval()

    # Apply torch.compile if requested
    if use_torch_compile:
        try:
            progress(0.75, desc="Compiling SR model...")
            sr_model = torch.compile(sr_model, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: torch.compile failed for SR model: {e}")

    # Cache the models
    cached_models["base"] = {
        "model": base_model,
        "diffusion": base_diffusion,
        "options": base_options,
    }
    cached_models["sr"] = {
        "model": sr_model,
        "diffusion": sr_diffusion,
        "options": sr_options,
    }
    cached_models["current_paths"]["base"] = base_model_path
    cached_models["current_paths"]["sr"] = sr_model_path
    cached_models["device"] = device
    cached_models["use_fp16"] = use_fp16
    cached_models["use_torch_compile"] = use_torch_compile

    progress(1.0, desc="Models loaded!")

    return base_model, base_diffusion, base_options, sr_model, sr_diffusion, sr_options


def generate_images(
    prompt: str,
    batch_size: int,
    sampler: str,
    base_steps: int,
    sr_steps: int,
    cfg_scale: float,
    seed: Optional[int],
    base_model_path: str,
    sr_model_path: str,
    use_fp16: bool,
    use_torch_compile: bool,
    progress=gr.Progress(),
) -> Tuple[List[Image.Image], str, str]:
    """
    Generate images from a text prompt.

    Returns:
        Tuple of (images, status_message, memory_info)
    """
    if not prompt:
        return [], "Please enter a prompt", get_gpu_memory_info()

    # Set device
    device = get_device()

    # Set random seed if provided
    if seed is not None and seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    try:
        # Load models if needed
        progress(0.0, desc="Loading models...")
        base_model, base_diffusion, base_options, sr_model, sr_diffusion, sr_options = (
            load_models_if_needed(
                base_model_path,
                sr_model_path,
                device,
                use_fp16,
                use_torch_compile,
                progress=progress,
            )
        )

        # Generate images
        start_time = time.time()
        progress(0.5, desc=f"Generating {batch_size} image(s)...")

        # Generate using sample_with_superres
        samples = sample_with_superres(
            base_model,
            base_options,
            sr_model,
            sr_options,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=cfg_scale,
            device=device,
            base_respacing=str(base_steps),
            upsampler_respacing=str(sr_steps),
            sampler=sampler,
        )

        progress(0.9, desc="Converting to images...")

        # Convert tensors to PIL images
        images = []
        # samples is a tensor of shape [batch_size, 3, 256, 256]
        for i in range(samples.shape[0]):
            sample = samples[i]
            # Convert from [-1, 1] to [0, 255]
            img_np = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            img_np = img_np.permute(1, 2, 0).cpu().numpy()
            images.append(Image.fromarray(img_np))

        elapsed = time.time() - start_time
        status = f"✓ Generated {batch_size} image(s) in {elapsed:.1f}s ({elapsed / batch_size:.1f}s per image)"

        # Clear CUDA cache after generation
        if device == "cuda":
            torch.cuda.empty_cache()

        progress(1.0, desc="Done!")

        return images, status, get_gpu_memory_info()

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return [], error_msg, get_gpu_memory_info()


def create_interface():
    """Create the Gradio interface."""

    # Example prompts
    example_prompts = [
        "a majestic mountain landscape at sunset",
        "a futuristic cityscape with flying cars",
        "a cute robot playing chess in a garden",
        "an oil painting of a sailing ship in a storm",
        "a cozy cabin in the woods during autumn",
        "a steampunk airship floating above clouds",
        "a magical forest with glowing mushrooms",
        "a serene Japanese garden with cherry blossoms",
    ]

    with gr.Blocks(title="GLIDE Text-to-Image Generator") as app:
        gr.Markdown(
            """
            # 🎨 GLIDE Text-to-Image Generator
            
            Generate high-quality images from text descriptions using GLIDE models.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description...",
                    lines=3,
                    value=random.choice(example_prompts),
                )

                with gr.Accordion("Generation Settings", open=True):
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=1,
                        step=1,
                        label="Batch Size",
                        info="Number of images to generate",
                    )

                    sampler = gr.Dropdown(
                        choices=["euler", "euler_a", "dpm++", "plms", "ddim"],
                        value="euler",
                        label="Sampler",
                        info="Sampling algorithm to use",
                    )

                    with gr.Row():
                        base_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=1,
                            label="Base Model Steps",
                            info="Steps for 64x64 generation",
                        )
                        sr_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=27,
                            step=1,
                            label="SR Model Steps",
                            info="Steps for 256x256 upsampling",
                        )

                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="CFG Scale",
                        info="Classifier-free guidance strength",
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="Random seed (-1 for random)",
                    )

                with gr.Accordion("Model Settings", open=False):
                    base_model_path = gr.Textbox(
                        label="Base Model Path",
                        value="checkpoints/0056/glide-ft-0x60000.pt",
                        info="Path to base model checkpoint",
                    )

                    sr_model_path = gr.Textbox(
                        label="SR Model Path",
                        value="glide_model_cache/upsample.pt",
                        info="Path to super-resolution model",
                    )

                    with gr.Row():
                        use_fp16 = gr.Checkbox(
                            label="Use FP16",
                            value=False,
                            info="Use half precision for memory savings",
                        )
                        use_torch_compile = gr.Checkbox(
                            label="Use torch.compile",
                            value=False,
                            info="Compile models for faster inference",
                        )

                generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")

                # Example prompts
                gr.Examples(
                    examples=[[p] for p in example_prompts],
                    inputs=prompt_input,
                    label="Example Prompts",
                )

            with gr.Column(scale=2):
                # Output section
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                )

                with gr.Row():
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Ready to generate...",
                    )
                    memory_text = gr.Textbox(
                        label="Memory Usage",
                        interactive=False,
                        value=get_gpu_memory_info(),
                    )

        # Wire up the generation
        generate_btn.click(
            fn=generate_images,
            inputs=[
                prompt_input,
                batch_size,
                sampler,
                base_steps,
                sr_steps,
                cfg_scale,
                seed,
                base_model_path,
                sr_model_path,
                use_fp16,
                use_torch_compile,
            ],
            outputs=[output_gallery, status_text, memory_text],
        )

        # Add footer
        gr.Markdown(
            """
            ---
            ### Tips:
            - **Batch Size**: Generate multiple variations at once
            - **Sampler**: Euler is fast and reliable, DPM++ can be good with fewer steps
            - **CFG Scale**: Higher values follow the prompt more closely (3-5 is usually good)
            - **Seed**: Use the same seed to reproduce results
            - **torch.compile**: Enable for faster generation after initial compilation
            """
        )

    return app


if __name__ == "__main__":
    # Check for share option from environment or command line
    import sys

    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    share = share or "--share" in sys.argv

    # Get port from environment
    port = int(os.environ.get("PORT", "7860"))

    # Create and launch the app
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )
