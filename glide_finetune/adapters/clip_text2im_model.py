"""
Extended Text2ImUNet model with CLIP adapter support.

This module provides a modified version of GLIDE's Text2ImUNet that can
optionally use CLIP features alongside the original text conditioning.
"""

from typing import Any, Dict, List, Optional

import torch
from glide_text2im.nn import timestep_embedding
from glide_text2im.text2im_model import Text2ImUNet
from glide_text2im.unet import AttentionBlock, TimestepBlock, TimestepEmbedSequential

from .clip_adapter import ClipAdapter, ClipTextEncoder
from .dual_attention import (
    DualAttentionBlock,
    DualConditioningAdapter,
    replace_attention_blocks,
)


class ClipText2ImUNet(Text2ImUNet):
    """
    Extended Text2ImUNet that supports dual conditioning with CLIP features.

    This model preserves all original GLIDE functionality while adding optional
    CLIP conditioning through adapters and dual attention mechanisms.
    """

    def __init__(
        self,
        *args,
        clip_model_name: Optional[str] = None,
        use_clip: bool = False,
        clip_gate_init: float = 0.0,
        adapter_hidden_dim: Optional[int] = None,
        adapter_dropout: float = 0.1,
        use_lora: bool = False,
        lora_rank: int = 32,
        freeze_glide_encoder: bool = True,
        **kwargs,
    ):
        """
        Args:
            *args: Arguments for parent Text2ImUNet
            clip_model_name: Name of CLIP model to use (e.g., 'ViT-L/14')
            use_clip: Whether to enable CLIP conditioning
            clip_gate_init: Initial value for CLIP gates (0.0 for stability)
            adapter_hidden_dim: Hidden dimension for adapter MLP
            adapter_dropout: Dropout rate in adapter
            use_lora: Use LoRA instead of full MLP in adapter
            lora_rank: Rank for LoRA decomposition
            freeze_glide_encoder: Whether to freeze GLIDE's text encoder
            **kwargs: Additional arguments for parent
        """
        super().__init__(*args, **kwargs)

        self.use_clip = use_clip
        self.clip_model_name = clip_model_name
        self.freeze_glide_encoder = freeze_glide_encoder

        if use_clip and clip_model_name:
            # Initialize CLIP components
            self._setup_clip_conditioning(
                clip_model_name=clip_model_name,
                clip_gate_init=clip_gate_init,
                adapter_hidden_dim=adapter_hidden_dim,
                adapter_dropout=adapter_dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
            )

            # Optionally freeze GLIDE's text encoder
            if freeze_glide_encoder:
                self._freeze_text_encoder()

    def _setup_clip_conditioning(
        self,
        clip_model_name: str,
        clip_gate_init: float,
        adapter_hidden_dim: Optional[int],
        adapter_dropout: float,
        use_lora: bool,
        lora_rank: int,
    ):
        """Set up CLIP model, adapter, and dual attention."""
        from .clip_adapter import create_clip_adapter_config, load_clip_model

        # Load CLIP model (will be moved to correct device later)
        self.clip_model, _ = load_clip_model(clip_model_name, device="cpu")
        self.clip_encoder = ClipTextEncoder(self.clip_model, device="cpu")

        # Get CLIP dimension
        clip_config = create_clip_adapter_config(
            clip_model_name=clip_model_name,
            glide_xf_width=self.xf_width,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            gate_init=clip_gate_init,
        )

        # Create adapter
        self.clip_adapter = ClipAdapter(**clip_config)

        # Create conditioning adapter
        self.conditioning_adapter = DualConditioningAdapter(
            text_dim=self.xf_width,
            clip_dim=clip_config["input_dim"],
            output_dim=self.xf_width,
            sequence_length=self.text_ctx,
            dropout=adapter_dropout,
        )

        # Don't replace attention blocks here - let load_glide_model_with_clip handle it
        # This allows pretrained weights to load properly first
        self._clip_gate_init = clip_gate_init
        self._attention_blocks_replaced = False
    
    def replace_attention_blocks_after_load(self):
        """Replace attention blocks with dual attention blocks after loading weights."""
        if self._attention_blocks_replaced:
            print("Attention blocks already replaced, skipping")
            return
            
        num_replaced = replace_attention_blocks(
            self,
            clip_channels=self.xf_width,
            clip_gate_init=self._clip_gate_init,
        )
        print(f"Replaced {num_replaced} attention blocks with dual attention")
        self._attention_blocks_replaced = True
        
        # Sanity check: Verify all cross-attention blocks are now DualAttentionBlock
        self._verify_attention_replacement()

    def _verify_attention_replacement(self):
        """
        Verify that all AttentionBlocks have been replaced with DualAttentionBlocks.
        This is a critical sanity check for CLIP integration.
        """
        from glide_text2im.unet import AttentionBlock

        attention_blocks = []
        dual_blocks = []

        # Walk through all modules and check attention blocks
        for name, module in self.named_modules():
            if isinstance(module, AttentionBlock):
                if isinstance(module, DualAttentionBlock):
                    dual_blocks.append(name)
                else:
                    attention_blocks.append(name)

        # Report findings
        total_blocks = len(attention_blocks) + len(dual_blocks)
        print(f"[Attention Check] Found {total_blocks} attention blocks:")
        print(f"  - {len(dual_blocks)} DualAttentionBlocks (✓)")
        print(f"  - {len(attention_blocks)} regular AttentionBlocks (⚠️)")

        # Assert that all are converted
        if attention_blocks:
            print("[Attention Check] ERROR: Found unconverted AttentionBlocks:")
            for block_name in attention_blocks[:5]:  # Show first 5
                print(f"    - {block_name}")
            if len(attention_blocks) > 5:
                print(f"    ... and {len(attention_blocks) - 5} more")

            raise RuntimeError(
                f"Found {len(attention_blocks)} AttentionBlocks that were not "
                f"converted to DualAttentionBlocks. This indicates a problem "
                f"with the replacement process."
            )

        print(
            f"[Attention Check] ✓ All {len(dual_blocks)} attention blocks "
            f"successfully converted to DualAttentionBlock"
        )

    def _freeze_text_encoder(self):
        """Freeze GLIDE's text encoder components."""
        # Freeze transformer
        if hasattr(self, "transformer"):
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Freeze embeddings
        if hasattr(self, "token_embedding"):
            self.token_embedding.requires_grad_(False)
        if hasattr(self, "positional_embedding"):
            self.positional_embedding.requires_grad = False
        if hasattr(self, "padding_embedding"):
            self.padding_embedding.requires_grad = False

        # Freeze transformer projection and final LN
        if hasattr(self, "transformer_proj"):
            self.transformer_proj.requires_grad_(False)
        if hasattr(self, "final_ln"):
            self.final_ln.requires_grad_(False)

    def get_clip_text_emb(self, text_prompts: list) -> torch.Tensor:
        """
        Get CLIP text embeddings for a batch of prompts.

        Args:
            text_prompts: List of text prompts

        Returns:
            CLIP embeddings [batch_size, clip_dim]
        """
        if not self.use_clip:
            raise ValueError("CLIP is not enabled for this model")

        return self.clip_encoder.encode_text(text_prompts)

    def forward(
        self,
        x,
        timesteps,
        tokens=None,
        mask=None,
        clip_text_prompts=None,
        clip_embeddings=None,
        use_clip_override=None,
        dry_run=False,
    ):
        """
        Forward pass with optional CLIP conditioning.

        Args:
            x: Input tensor
            timesteps: Diffusion timesteps
            tokens: Text tokens from GLIDE tokenizer
            mask: Token mask
            clip_text_prompts: Optional text prompts for CLIP encoding
            clip_embeddings: Pre-computed CLIP embeddings (alternative to prompts)
            use_clip_override: Override self.use_clip setting
            dry_run: If True, compute CLIP features but don't use them (for testing)

        Returns:
            Model output
        """
        # Determine whether to use CLIP
        use_clip = use_clip_override if use_clip_override is not None else self.use_clip

        # In dry run mode, compute CLIP features but don't use them
        # Note: use_clip_for_forward removed as it was unused

        # Get base embeddings
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Get text embeddings (always needed for now)
        if self.xf_width and tokens is not None:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None

        # Get CLIP embeddings if using CLIP
        clip_out = None
        if use_clip and (clip_text_prompts is not None or clip_embeddings is not None):
            # Get CLIP embeddings
            if clip_embeddings is None:
                clip_embeddings = self.get_clip_text_emb(clip_text_prompts)

            # Ensure correct dtype
            clip_embeddings = clip_embeddings.to(self.dtype)

            # Pass through adapter
            adapted_clip = self.clip_adapter(clip_embeddings)

            # Convert to sequence format and ensure correct dtype
            _, clip_out = self.conditioning_adapter(xf_out, adapted_clip)
            clip_out = clip_out.to(self.dtype)

            # Debug prints disabled for production
            # Debug prints disabled for production
            # print(f"DEBUG forward: use_clip={use_clip}, clip_out shape={
            #     clip_out.shape if clip_out is not None else None}")
            # print(f"DEBUG forward: adapter gate={
            #     self.clip_adapter.get_gate_value():.4f}")

            # In dry run mode, set clip_out to None so it's not used
            if dry_run:
                clip_out = None

        # Pass through UNet with dual conditioning
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = self._forward_timestep_embed_sequential(
                module, h, emb, xf_out, clip_out
            )
            hs.append(h)

        h = self._forward_timestep_embed_sequential(
            self.middle_block, h, emb, xf_out, clip_out
        )

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._forward_timestep_embed_sequential(
                module, h, emb, xf_out, clip_out
            )

        h = h.type(x.dtype)
        h = self.out(h)
        return h

    def get_adapter_mlp_params(self):
        """Get only the MLP adapter parameters (for adapter_only phase)."""
        params: List[torch.nn.Parameter] = []
        if hasattr(self, "clip_adapter"):
            params.extend(self.clip_adapter.parameters())
        if hasattr(self, "conditioning_adapter"):
            params.extend(self.conditioning_adapter.parameters())
        return params

    def get_attention_gate_params(self):
        """Get only the attention gate parameters."""
        gate_params = []
        for module in self.modules():
            if hasattr(module, "clip_gate"):
                gate_params.append(module.clip_gate)
        return gate_params

    def get_clip_kv_params(self):
        """Get only the CLIP K/V projection parameters."""
        kv_params = []
        for module in self.modules():
            if hasattr(module, "clip_kv") and hasattr(module.clip_kv, "parameters"):
                kv_params.extend(list(module.clip_kv.parameters()))
            if hasattr(module, "clip_attention") and hasattr(
                module.clip_attention, "parameters"
            ):
                kv_params.extend(list(module.clip_attention.parameters()))
        return kv_params

    def get_adapter_params(self):
        """Get ALL CLIP adapter parameters (legacy method for compatibility)."""
        params = []
        params.extend(self.get_adapter_mlp_params())
        params.extend(self.get_attention_gate_params())
        params.extend(self.get_clip_kv_params())
        return params

    def set_adapter_gate_schedule(self, current_step: int, warmup_steps: int):
        """
        Set adapter and attention gates based on warmup schedule.

        Args:
            current_step: Current training step
            warmup_steps: Number of warmup steps
        """
        if warmup_steps <= 0:
            progress = 1.0
        else:
            progress = min(1.0, current_step / warmup_steps)

        # Gradual increase from 0 to 0.5
        target_gate = 0.5 * progress

        # Update adapter gate
        if hasattr(self, "clip_adapter"):
            self.clip_adapter.set_gate_value(target_gate)

        # Update attention gates
        for module in self.modules():
            if hasattr(module, "set_clip_gate_value") and callable(
                getattr(module, "set_clip_gate_value")
            ):
                module.set_clip_gate_value(target_gate)

    def get_stability_metrics(self) -> Dict[str, float]:
        """Get metrics for monitoring training stability."""
        metrics = {}

        # Adapter gate value
        if hasattr(self, "clip_adapter"):
            metrics["adapter_gate"] = self.clip_adapter.get_gate_value()

        # Attention gate values
        gate_values = []
        for i, module in enumerate(self.modules()):
            if hasattr(module, "get_clip_gate_value") and callable(
                getattr(module, "get_clip_gate_value")
            ):
                value = module.get_clip_gate_value()
                gate_values.append(value)
                metrics[f"attention_gate_{i}"] = value

        if gate_values:
            metrics["attention_gate_mean"] = sum(gate_values) / len(gate_values)

        return metrics

    def to(self, *args, **kwargs):
        """Override to method to also move CLIP components."""
        # Call parent to() method
        result = super().to(*args, **kwargs)

        # Move CLIP components if they exist
        if hasattr(self, "clip_model") and self.clip_model is not None:
            self.clip_model = self.clip_model.to(*args, **kwargs)

        if hasattr(self, "clip_encoder") and self.clip_encoder is not None:
            self.clip_encoder.device = str(next(self.parameters()).device)

        return result

    def convert_to_fp16(self):
        """Convert model to fp16."""
        super().convert_to_fp16()

        # Convert CLIP adapter components
        if hasattr(self, "clip_adapter"):
            self.clip_adapter.half()
        if hasattr(self, "conditioning_adapter"):
            self.conditioning_adapter.half()

        # Convert any dual attention blocks
        for module in self.modules():
            if hasattr(module, "clip_kv"):
                module.clip_kv.half()
            if hasattr(module, "clip_attention"):
                module.clip_attention.half()

    def dry_run_test(
        self,
        x,
        timesteps,
        tokens=None,
        mask=None,
        clip_text_prompts=None,
        clip_embeddings=None,
        return_metrics=True,
    ) -> Dict[str, Any]:
        """
        Perform a dry run test to verify CLIP adapter doesn't affect outputs.

        This runs the model twice:
        1. With CLIP features computed but not used (dry_run=True)
        2. Without any CLIP computation (baseline)

        Args:
            x: Input tensor
            timesteps: Diffusion timesteps
            tokens: Text tokens from GLIDE tokenizer
            mask: Token mask
            clip_text_prompts: Optional text prompts for CLIP encoding
            clip_embeddings: Pre-computed CLIP embeddings
            return_metrics: Whether to compute detailed metrics

        Returns:
            Dict containing:
                - output_diff_max: Maximum absolute difference in outputs
                - output_diff_mean: Mean absolute difference in outputs
                - outputs_identical: Whether outputs are exactly identical
                - clip_embeddings_computed: Whether CLIP embeddings were computed
                - adapter_gate_value: Current adapter gate value
                - dry_run_output: Output with dry_run=True
                - baseline_output: Output without CLIP
        """
        # Store original use_clip setting
        original_use_clip = self.use_clip

        results = {}

        # Run with dry_run=True (compute CLIP but don't use it)
        if self.use_clip and (
            clip_text_prompts is not None or clip_embeddings is not None
        ):
            with torch.no_grad():
                dry_run_output = self.forward(
                    x=x,
                    timesteps=timesteps,
                    tokens=tokens,
                    mask=mask,
                    clip_text_prompts=clip_text_prompts,
                    clip_embeddings=clip_embeddings,
                    dry_run=True,
                )
                results["clip_embeddings_computed"] = True
        else:
            # No CLIP features to compute
            dry_run_output = None
            results["clip_embeddings_computed"] = False

        # Run baseline (no CLIP at all)
        self.use_clip = False
        with torch.no_grad():
            baseline_output = self.forward(
                x=x,
                timesteps=timesteps,
                tokens=tokens,
                mask=mask,
            )

        # Restore original setting
        self.use_clip = original_use_clip

        # Store outputs
        results["dry_run_output"] = dry_run_output
        results["baseline_output"] = baseline_output

        # Compute metrics if requested
        if return_metrics and dry_run_output is not None:
            diff = torch.abs(dry_run_output - baseline_output)
            results["output_diff_max"] = float(diff.max().item())
            results["output_diff_mean"] = float(diff.mean().item())
            results["outputs_identical"] = bool(
                torch.allclose(dry_run_output, baseline_output, rtol=1e-5, atol=1e-8)
            )

            # Add adapter metrics
            if hasattr(self, "clip_adapter"):
                results["adapter_gate_value"] = float(
                    self.clip_adapter.get_gate_value()
                )

        return results

    def _forward_timestep_embed_sequential(
        self, module, x, emb, encoder_out=None, clip_encoder_out=None
    ):
        """
        Forward through a TimestepEmbedSequential module with dual conditioning support.

        This extends the standard TimestepEmbedSequential forward to also pass
        clip_encoder_out to DualAttentionBlock layers.
        """
        if isinstance(module, TimestepEmbedSequential):
            # Manually iterate through layers to pass clip_encoder_out
            for layer in module:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                elif isinstance(layer, DualAttentionBlock):
                    # Enable debugging on DualAttentionBlock
                    if hasattr(self, "_debug_clip") and self._debug_clip:
                        layer._debug = True
                    x = layer(x, encoder_out, clip_encoder_out)
                    if hasattr(self, "_debug_clip") and self._debug_clip:
                        layer._debug = False
                elif isinstance(layer, AttentionBlock):
                    x = layer(x, encoder_out)
                else:
                    x = layer(x)
            return x
        else:
            # Not a TimestepEmbedSequential, use standard forward
            return module(x, emb, encoder_out)
