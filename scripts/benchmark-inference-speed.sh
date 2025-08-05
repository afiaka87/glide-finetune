#!/bin/bash
# Benchmark script to compare inference speeds with different optimizations

echo "🏁 Benchmarking inference speed with different optimizations..."
echo "Generating 4 images with 2 CLIP models for each test"
echo

# Test 1: Baseline (no optimizations)
echo "Test 1: Baseline (no optimizations)"
time uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --prompt "a red sports car on a mountain road" \
    --num_samples 4 \
    --sampler euler \
    --steps 30 \
    --clip_model ViT-L/14 ViT-B/32 \
    --output_dir outputs/benchmark/baseline

echo -e "\n---\n"

# Test 2: With AMP
echo "Test 2: With AMP (Automatic Mixed Precision)"
time uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --prompt "a red sports car on a mountain road" \
    --num_samples 4 \
    --sampler euler \
    --steps 30 \
    --clip_model ViT-L/14 ViT-B/32 \
    --amp \
    --output_dir outputs/benchmark/amp

echo -e "\n---\n"

# Test 3: With torch.compile
echo "Test 3: With torch.compile (first run will be slower)"
time uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --prompt "a red sports car on a mountain road" \
    --num_samples 4 \
    --sampler euler \
    --steps 30 \
    --clip_model ViT-L/14 ViT-B/32 \
    --compile \
    --output_dir outputs/benchmark/compiled

echo -e "\n---\n"

# Test 4: All optimizations
echo "Test 4: All optimizations (AMP + torch.compile + larger batch)"
time uv run python inference_clip_rerank.py \
    --checkpoint synthetic-1m-dalle-high-quality.pt \
    --prompt "a red sports car on a mountain road" \
    --num_samples 4 \
    --sampler euler \
    --steps 30 \
    --clip_model ViT-L/14 ViT-B/32 \
    --amp \
    --compile \
    --compile_mode reduce-overhead \
    --batch_size 4 \
    --output_dir outputs/benchmark/turbo

echo -e "\n✅ Benchmark complete! Check timing results above."