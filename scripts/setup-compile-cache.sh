#!/bin/bash
# Script to set up persistent torch.compile cache

echo "Setting up torch.compile cache for faster subsequent runs..."

# Create cache directory
CACHE_DIR="$HOME/.cache/torch_compile/glide_finetune"
mkdir -p "$CACHE_DIR"

# Export environment variables for the current session
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCH_COMPILE_DEBUG=0

echo "Cache directory created at: $CACHE_DIR"
echo
echo "To make this permanent, add these lines to your ~/.bashrc or ~/.zshrc:"
echo
echo "# Torch compile cache for GLIDE"
echo "export TORCHINDUCTOR_CACHE_DIR=\"$CACHE_DIR\""
echo "export TORCHINDUCTOR_FX_GRAPH_CACHE=1"
echo
echo "Then run: source ~/.bashrc"
echo
echo "You can also use the --compile_cache_dir flag to specify a custom cache location."