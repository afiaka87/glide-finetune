# Enhanced Sampling Methods for GLIDE

This document describes the enhanced sampling methods (Euler, Euler Ancestral, and DPM++) that have been integrated into the GLIDE finetuning codebase.

## Overview

The original GLIDE implementation includes PLMS and DDIM samplers. We've extended this with three additional state-of-the-art sampling methods that offer different trade-offs between speed, quality, and stochasticity.

## Available Samplers

### 1. PLMS (Pseudo Linear Multi-Step)
- **Type**: Deterministic
- **Speed**: Moderate
- **Quality**: Good
- **Best for**: General use, balanced performance
- **Steps**: 50-100 typical

### 2. DDIM (Denoising Diffusion Implicit Models)
- **Type**: Deterministic or Stochastic (via eta)
- **Speed**: Moderate  
- **Quality**: Good
- **Best for**: Flexible deterministic/stochastic control
- **Steps**: 50-100 typical
- **Parameters**: 
  - `eta`: 0.0 (deterministic) to 1.0 (fully stochastic)

### 3. Euler
- **Type**: Deterministic
- **Speed**: Fast
- **Quality**: Good with sufficient steps
- **Best for**: Fast deterministic generation
- **Steps**: 50-100 typical
- **Notes**: Simple first-order ODE solver

### 4. Euler Ancestral
- **Type**: Stochastic
- **Speed**: Fast
- **Quality**: Good, more diverse outputs
- **Best for**: Diverse/creative outputs
- **Steps**: 50-100 typical
- **Parameters**:
  - `eta`: Controls noise level (typically 1.0)

### 5. DPM++ (DPM-Solver++)
- **Type**: Deterministic
- **Speed**: Very Fast
- **Quality**: Excellent with fewer steps
- **Best for**: High quality with minimal steps
- **Steps**: 20-30 typical
- **Parameters**:
  - `order`: 1 or 2 (2 is recommended)

## Usage Examples

### Basic Usage

```python
from glide_finetune.glide_util import load_model, sample

# Load model
model, diffusion, options = load_model(model_type="base")
model.to("cuda")

# Generate with Euler
sample_euler = sample(
    model, options, 64, 64,
    prompt="a beautiful landscape",
    sampler="euler",
    prediction_respacing="50",
    guidance_scale=3.0,
)

# Generate with DPM++ (fewer steps)
sample_dpm = sample(
    model, options, 64, 64,
    prompt="a beautiful landscape", 
    sampler="dpm++",
    prediction_respacing="25",
    dpm_order=2,
)

# Generate with Euler Ancestral (stochastic)
sample_euler_a = sample(
    model, options, 64, 64,
    prompt="a beautiful landscape",
    sampler="euler_a",
    sampler_eta=1.0,
)
```

### Testing All Samplers

```bash
# Run the test script to compare all samplers
uv run python test_samplers.py \
    --prompt "your prompt here" \
    --steps 50 \
    --guidance-scale 3.0

# Run the example script for interactive testing
uv run python example_samplers.py
```

## Implementation Details

### Architecture

The enhanced samplers are implemented as extensions to the `GaussianDiffusion` class using a monkey-patching approach. This ensures:

1. **Compatibility**: Existing code continues to work unchanged
2. **Modularity**: Samplers can be added/removed independently
3. **Clean Interface**: Same API as existing samplers

### Key Components

- `enhanced_samplers.py`: Core implementation of new sampling algorithms
- `enhance_diffusion()`: Function to add samplers to a diffusion instance
- Type hints and comprehensive documentation throughout

### Design Principles

1. **Immutability**: Samplers don't modify internal state
2. **Type Safety**: Full type hints for all methods
3. **Documentation**: Extensive docstrings explaining each algorithm
4. **Compatibility**: Seamless integration with existing GLIDE code

## Performance Comparison

| Sampler | Relative Speed | Steps Needed | Deterministic | Quality |
|---------|---------------|--------------|---------------|---------|
| PLMS    | 1.0x (baseline) | 50-100 | Yes | Good |
| DDIM    | ~1.0x | 50-100 | Optional | Good |
| Euler   | ~1.2x | 50-100 | Yes | Good |
| Euler A | ~1.2x | 50-100 | No | Good |
| DPM++   | ~2-3x | 20-30 | Yes | Excellent |

## Recommendations

- **For Speed**: Use DPM++ with 20-30 steps
- **For Determinism**: Use Euler or DDIM with eta=0
- **For Diversity**: Use Euler Ancestral with eta=1.0
- **For Quality**: Use DPM++ with order=2
- **For Compatibility**: Continue using PLMS (default)

## Technical Notes

### Classifier-Free Guidance

All samplers support classifier-free guidance (CFG) through the standard model wrapper approach used by GLIDE.

### Memory Usage

The new samplers have minimal memory overhead:
- No additional model parameters
- Only temporary tensors during sampling
- DPM++ stores one previous epsilon (negligible)

### Numerical Stability

- All samplers use FP32 for critical computations
- Proper clamping and normalization applied
- Tested with various guidance scales

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `glide_finetune/enhanced_samplers.py` exists
2. **AttributeError**: Call `enhance_diffusion()` before using new samplers
3. **CUDA OOM**: Reduce batch size or use CPU for testing
4. **Poor Quality**: Adjust steps, guidance scale, or try different sampler

### Validation

Run the test script to verify installation:
```bash
uv run python quick_test_samplers.py
```

## Future Enhancements

Potential additions:
- PNDM (Pseudo Numerical Diffusion Model)
- SDE variants of existing solvers
- Adaptive step size methods
- Higher-order DPM-Solver variants

## References

- [Euler Methods](https://arxiv.org/abs/2206.00364) - EDM paper
- [DPM-Solver++](https://arxiv.org/abs/2211.01095) - Original paper
- [DDIM](https://arxiv.org/abs/2010.02502) - Original paper
- [PLMS](https://arxiv.org/abs/2202.09778) - Original paper