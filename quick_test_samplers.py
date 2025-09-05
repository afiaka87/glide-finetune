#!/usr/bin/env python3
"""Quick test to verify the samplers are working."""

import sys
import torch
from glide_text2im.model_creation import create_gaussian_diffusion, model_and_diffusion_defaults
from glide_finetune.enhanced_samplers import enhance_diffusion

def test_basic_functionality():
    """Test that we can enhance a diffusion instance and the methods exist."""
    print("Testing basic sampler functionality...")
    
    # Create a minimal diffusion instance
    options = model_and_diffusion_defaults()
    diffusion = create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing="50",  # Use fewer steps for testing
    )
    
    # Enhance with new samplers
    enhance_diffusion(diffusion)
    
    # Check that new methods exist
    assert hasattr(diffusion, "euler_sample_loop"), "Euler sampler not added"
    assert hasattr(diffusion, "euler_ancestral_sample_loop"), "Euler Ancestral sampler not added"
    assert hasattr(diffusion, "dpm_solver_sample_loop"), "DPM++ sampler not added"
    
    print("✓ All samplers successfully added to diffusion instance")
    
    # Test that we can call them (with dummy model)
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            # Return model output with variance prediction
            # GLIDE models return (B, C*2, H, W) where C*2 = 3*2 = 6
            B, C, H, W = x.shape
            return torch.randn(B, C * 2, H, W, device=x.device)
    
    model = DummyModel()
    shape = (1, 3, 64, 64)
    device = "cpu"
    
    print("Testing sampler calls...")
    
    # Test Euler
    try:
        with torch.no_grad():
            samples = diffusion.euler_sample_loop(
                model,
                shape,
                device=device,
                progress=False,
            )
        print("✓ Euler sampler works")
    except Exception as e:
        import traceback
        print(f"✗ Euler sampler failed: {e}")
        traceback.print_exc()
        return False
    
    # Test Euler Ancestral
    try:
        with torch.no_grad():
            samples = diffusion.euler_ancestral_sample_loop(
                model,
                shape,
                device=device,
                progress=False,
                eta=1.0,
            )
        print("✓ Euler Ancestral sampler works")
    except Exception as e:
        print(f"✗ Euler Ancestral sampler failed: {e}")
        return False
    
    # Test DPM++
    try:
        with torch.no_grad():
            samples = diffusion.dpm_solver_sample_loop(
                model,
                shape,
                device=device,
                progress=False,
                order=2,
            )
        print("✓ DPM++ sampler works")
    except Exception as e:
        print(f"✗ DPM++ sampler failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)