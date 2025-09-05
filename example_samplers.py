#!/usr/bin/env python3
"""
Example script showing how to use the new sampling methods with GLIDE.

This demonstrates using Euler, Euler Ancestral, and DPM++ samplers
for text-to-image generation.
"""

import torch
from glide_finetune.glide_util import load_model, sample
import matplotlib.pyplot as plt
from pathlib import Path


def generate_with_samplers(prompt: str, output_dir: str = "examples"):
    """Generate images using different samplers and save them."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    print("Loading GLIDE model...")
    model, diffusion, options = load_model(
        model_type="base",
        use_fp16=False,  # FP16 can cause gradient issues
    )
    model.to(device)
    model.eval()
    
    # Configuration
    samplers_config = [
        {"name": "plms", "eta": 0.0, "steps": "50", "description": "PLMS (default)"},
        {"name": "ddim", "eta": 0.0, "steps": "50", "description": "DDIM (deterministic)"},
        {"name": "ddim", "eta": 1.0, "steps": "50", "description": "DDIM (stochastic)"},
        {"name": "euler", "eta": 0.0, "steps": "50", "description": "Euler"},
        {"name": "euler_a", "eta": 1.0, "steps": "50", "description": "Euler Ancestral"},
        {"name": "dpm++", "eta": 0.0, "steps": "25", "description": "DPM++ (25 steps)"},
    ]
    
    results = []
    
    for config in samplers_config:
        print(f"\nGenerating with {config['description']}...")
        
        # Generate sample
        samples = sample(
            glide_model=model,
            glide_options=options,
            side_x=64,
            side_y=64,
            prompt=prompt,
            batch_size=1,
            guidance_scale=3.0,
            device=device,
            prediction_respacing=config["steps"],
            sampler=config["name"],
            sampler_eta=config["eta"],
            dpm_order=2 if config["name"] == "dpm++" else 2,
        )
        
        # Convert to displayable format
        samples_np = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        samples_np = samples_np.permute(0, 2, 3, 1).cpu().numpy()[0]
        
        results.append({
            "image": samples_np,
            "title": config["description"],
        })
        
        # Save individual image
        filename = f"{config['description'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.imsave(output_path / filename, samples_np)
        print(f"Saved to {output_path / filename}")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sampler Comparison: "{prompt}"', fontsize=16)
    
    for idx, (ax, result) in enumerate(zip(axes.flat, results)):
        ax.imshow(result["image"])
        ax.set_title(result["title"])
        ax.axis("off")
    
    plt.tight_layout()
    comparison_path = output_path / "sampler_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    print(f"\nComparison saved to {comparison_path}")
    plt.show()


def main():
    """Main function to run the example."""
    
    # Example prompts to try
    prompts = [
        "a serene mountain landscape at sunset",
        "a cute robot playing chess",
        "abstract colorful geometric patterns",
        "a medieval castle in the clouds",
    ]
    
    print("GLIDE Enhanced Samplers Example")
    print("=" * 50)
    print("\nAvailable prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"{i}. {p}")
    
    # Get user choice
    choice = input("\nEnter prompt number (or custom prompt): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(prompts):
        prompt = prompts[int(choice) - 1]
    else:
        prompt = choice if choice else prompts[0]
    
    print(f"\nGenerating images for: '{prompt}'")
    generate_with_samplers(prompt)
    
    print("\nDone! Check the 'examples' directory for outputs.")


if __name__ == "__main__":
    main()