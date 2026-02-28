"""
JiT (Just image Transformer) — ViT-based diffusion model with adaLN-Zero.

Architecture from "Back to Basics: Let Denoising Generative Models Denoise"
(Li & He, 2025). Uses rectified flow (x-prediction + v-loss) instead of DDPM
epsilon-prediction.

Key design choices:
- Patchify input images into token sequences
- adaLN-Zero conditioning (timestep + pooled text → shift/scale/gate)
- Cross-attention for per-token text conditioning
- Zero-initialized output layers for identity initialization
- Text encoder: GLIDE (512-dim, 76M) or OpenCLIP ViT-L/14 (768-dim, 124M)
"""

from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from glide_text2im.nn import timestep_embedding
from glide_text2im.xf import (
    LayerNorm,
    Transformer,
    convert_module_to_bf16,
    convert_module_to_f16,
)


# ---------------------------------------------------------------------------
# CLIP tokenizer wrapper
# ---------------------------------------------------------------------------


class CLIPTokenizerWrapper:
    """Wraps OpenCLIP tokenizer to match GLIDE BPE tokenizer interface."""

    def __init__(self, model_name="ViT-L-14"):
        import open_clip

        self._tokenizer = open_clip.get_tokenizer(model_name)
        # Access the internal SimpleTokenizer for encode()
        self._inner = (
            self._tokenizer.tokenizer
            if hasattr(self._tokenizer, "tokenizer")
            else self._tokenizer
        )
        self.n_vocab = 49408

    def encode(self, text: str) -> list[int]:
        """Tokenize text to token IDs (without SOT/EOT)."""
        return self._inner.encode(text)

    def padded_tokens_and_mask(self, tokens: list[int], text_ctx: int):
        """Pad/truncate tokens and create attention mask."""
        SOT, EOT = 49406, 49407
        tokens = [SOT] + tokens[: text_ctx - 2] + [EOT]
        mask = [True] * len(tokens)
        padding = text_ctx - len(tokens)
        tokens = tokens + [0] * padding
        mask = mask + [False] * padding
        return tokens, mask


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

JIT_CONFIGS = {
    "B": dict(hidden_dim=768, depth=12, heads=12, patch_size=4, bottleneck_dim=128),
    "L": dict(hidden_dim=1024, depth=24, heads=16, patch_size=4, bottleneck_dim=256),
    "H": dict(hidden_dim=1280, depth=32, heads=16, patch_size=4, bottleneck_dim=256),
    "G": dict(hidden_dim=1536, depth=40, heads=24, patch_size=4, bottleneck_dim=256),
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Multi-head self-attention with F.scaled_dot_product_attention."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, N, H, D]
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: Q from image tokens, KV from text tokens."""

    def __init__(self, dim: int, heads: int, kv_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: th.Tensor, context: th.Tensor) -> th.Tensor:
        B, N, C = x.shape
        _, S, _ = context.shape

        q = self.q(x).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, S, 2, self.heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MLP(nn.Module):
    """Feed-forward with GELU and 4x expansion."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    3 sub-blocks: self-attention, cross-attention, MLP.
    Each gets its own (shift, scale, gate) from the conditioning vector.
    """

    def __init__(self, hidden_dim: int, heads: int, cond_dim: int, text_dim: int):
        super().__init__()
        # adaLN modulation: 9 values = 3 × (shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 9 * hidden_dim),
        )
        # Zero-init the linear output so model starts as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = SelfAttention(hidden_dim, heads)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = CrossAttention(hidden_dim, heads, kv_dim=text_dim)

        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = MLP(hidden_dim)

    def forward(self, x: th.Tensor, cond: th.Tensor, context: th.Tensor) -> th.Tensor:
        # cond: [B, cond_dim] → [B, 1, 9*hidden]
        mod = self.adaLN_modulation(cond).unsqueeze(1)
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = mod.chunk(
            9, dim=-1
        )

        # Self-attention with adaLN-Zero
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.self_attn(h)

        # Cross-attention with adaLN-Zero
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.cross_attn(h, context)

        # MLP with adaLN-Zero
        h = self.norm3(x) * (1 + scale3) + shift3
        x = x + gate3 * self.mlp(h)

        return x


class FinalLayer(nn.Module):
    """Final layer: adaLN (shift+scale only, no gate) → Linear projection.

    Both the adaLN modulation and the linear projection are zero-initialized
    so the model output starts at zero.
    """

    def __init__(self, hidden_dim: int, patch_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, patch_dim)

        # Zero-init both
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: th.Tensor, cond: th.Tensor) -> th.Tensor:
        mod = self.adaLN_modulation(cond).unsqueeze(1)
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class JiTModel(nn.Module):
    """JiT: Just image Transformer for rectified flow diffusion.

    Args:
        image_size: Input image resolution (must be square).
        patch_size: Patch size for patchification.
        in_channels: Number of input image channels.
        hidden_dim: Transformer hidden dimension.
        depth: Number of DiTBlock layers.
        heads: Number of attention heads.
        bottleneck_dim: Intermediate dim for patch embedding bottleneck.
        text_ctx: Number of text tokens.
        xf_width: Text transformer width.
        xf_layers: Text transformer depth.
        xf_heads: Text transformer attention heads.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        bottleneck_dim: int = 128,
        # Text encoder params (GLIDE defaults)
        text_ctx: int = 128,
        xf_width: int = 512,
        xf_layers: int = 16,
        xf_heads: int = 8,
        # Text encoder selection
        text_encoder: str = "glide",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.text_encoder = text_encoder

        # Patch dimensions
        self.num_patches = (image_size // patch_size) ** 2  # 256 for 64/4
        self.patch_dim = in_channels * patch_size * patch_size  # 48 for 3*4*4

        # Conditioning dimension: hidden_dim * 4 (matches GLIDE convention)
        self.cond_dim = hidden_dim * 4

        # --- Patch embedding (bottleneck) ---
        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

        # --- Positional embedding ---
        self.pos_embed = nn.Parameter(th.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Time embedding ---
        # Sinusoidal → MLP, same pattern as GLIDE UNet
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        # --- Text encoder ---
        if text_encoder == "clip":
            # OpenCLIP ViT-L/14 (DataComp): 768-dim, 77 context, 49408 vocab
            import open_clip

            xf_width = 768
            text_ctx = 77
            self.text_ctx = text_ctx
            self.xf_width = xf_width

            clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
            )
            # Extract only text encoder components (drop vision tower to save ~600MB)
            self.clip_token_embedding = clip_model.token_embedding
            self.clip_positional_embedding = clip_model.positional_embedding
            self.clip_transformer = clip_model.transformer
            self.clip_ln_final = clip_model.ln_final
            self.clip_text_projection = clip_model.text_projection
            del clip_model
            self.tokenizer = CLIPTokenizerWrapper()

            # Projection layers (trainable)
            self.text_proj = nn.Linear(xf_width, self.cond_dim)
            self.text_token_proj = nn.Linear(xf_width, hidden_dim)
        else:
            # GLIDE text encoder: 512-dim, 128 context, ~16k vocab
            self.text_ctx = text_ctx
            self.xf_width = xf_width

            from glide_text2im.tokenizer.bpe import get_encoder

            self.tokenizer = get_encoder()

            self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width)
            self.positional_embedding = nn.Parameter(
                th.empty(text_ctx, xf_width, dtype=th.float32)
            )
            nn.init.normal_(self.positional_embedding, std=0.01)

            self.transformer = Transformer(text_ctx, xf_width, xf_layers, xf_heads)
            self.final_ln = LayerNorm(xf_width)

            # Projection layers (trainable)
            self.text_proj = nn.Linear(xf_width, self.cond_dim)
            self.text_token_proj = nn.Linear(xf_width, hidden_dim)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, heads, self.cond_dim, text_dim=hidden_dim)
                for _ in range(depth)
            ]
        )

        # --- Final layer ---
        self.final_layer = FinalLayer(hidden_dim, self.patch_dim, self.cond_dim)

        # --- Caching ---
        self.cache_text_emb = False
        self.cache = None

    # ----- Patchify / Unpatchify -----

    def patchify(self, x: th.Tensor) -> th.Tensor:
        """[B, C, H, W] → [B, num_patches, patch_dim]"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H/p, W/p, C, p, p]
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return x

    def unpatchify(self, x: th.Tensor) -> th.Tensor:
        """[B, num_patches, patch_dim] → [B, C, H, W]"""
        B = x.shape[0]
        p = self.patch_size
        C = self.in_channels
        h = w = self.image_size // p
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, h, p, w, p]
        x = x.reshape(B, C, self.image_size, self.image_size)
        return x

    # ----- Text encoding -----

    def get_text_emb(self, tokens: th.Tensor, mask: th.Tensor) -> dict:
        """Encode text tokens. Mirrors Text2ImUNet.get_text_emb with caching."""
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (tokens == self.cache["tokens"]).all()
            return self.cache

        if self.text_encoder == "clip":
            outputs = self._get_text_emb_clip(tokens)
        else:
            outputs = self._get_text_emb_glide(tokens)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_pooled=outputs["xf_pooled"].detach(),
                xf_tokens=outputs["xf_tokens"].detach(),
            )

        return outputs

    def _get_text_emb_clip(self, tokens: th.Tensor) -> dict:
        """Text encoding via frozen OpenCLIP ViT-L/14."""
        x = self.clip_token_embedding(tokens.long())  # [B, 77, 768]
        x = x + self.clip_positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_ln_final(x)

        # Pooled: EOT token (argmax of token IDs) with text_projection
        pooled = x[th.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
        pooled = pooled @ self.clip_text_projection

        # L2-normalize CLIP outputs — raw CLIP embeddings have norm ~20-30
        # which overwhelms the time conditioning (~7). Normalizing to unit
        # vectors lets the trainable projection layers learn the right scale.
        pooled = F.normalize(pooled, dim=-1)
        x = F.normalize(x, dim=-1)

        xf_pooled = self.text_proj(pooled)  # [B, cond_dim]
        xf_tokens = self.text_token_proj(x)  # [B, 77, hidden_dim]
        return dict(xf_pooled=xf_pooled, xf_tokens=xf_tokens)

    def _get_text_emb_glide(self, tokens: th.Tensor) -> dict:
        """Text encoding via GLIDE transformer."""
        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        xf_out = self.transformer(xf_in.to(self.dtype))
        xf_out = self.final_ln(xf_out)

        # Pooled representation from last token → adaLN conditioning
        xf_pooled = self.text_proj(xf_out[:, -1])
        # Per-token representation → cross-attention context
        xf_tokens = self.text_token_proj(xf_out)
        return dict(xf_pooled=xf_pooled, xf_tokens=xf_tokens)

    def del_cache(self):
        """Clear text embedding cache (compat with sampling code)."""
        self.cache = None

    @property
    def dtype(self) -> th.dtype:
        """Model dtype, inferred from patch embedding weights."""
        return self.patch_embed[0].weight.dtype

    # ----- Precision conversion -----

    def convert_to_fp16(self):
        """Convert model weights to float16."""
        self.patch_embed.apply(convert_module_to_f16)
        self.pos_embed.data = self.pos_embed.data.half()
        self.time_embed.apply(convert_module_to_f16)
        if self.text_encoder == "clip":
            self.clip_token_embedding.to(th.float16)
            self.clip_positional_embedding.data = self.clip_positional_embedding.data.half()
            self.clip_transformer.to(th.float16)
            self.clip_ln_final.to(th.float16)
            self.clip_text_projection.data = self.clip_text_projection.data.half()
        else:
            self.transformer.apply(convert_module_to_f16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.data = self.positional_embedding.data.half()
        self.text_proj.apply(convert_module_to_f16)
        self.text_token_proj.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.final_layer.linear.apply(convert_module_to_f16)
        self.final_layer.adaLN_modulation.apply(convert_module_to_f16)

    def convert_to_bf16(self):
        """Convert model weights to bfloat16."""
        self.patch_embed.apply(convert_module_to_bf16)
        self.pos_embed.data = self.pos_embed.data.bfloat16()
        self.time_embed.apply(convert_module_to_bf16)
        if self.text_encoder == "clip":
            self.clip_token_embedding.to(th.bfloat16)
            self.clip_positional_embedding.data = self.clip_positional_embedding.data.bfloat16()
            self.clip_transformer.to(th.bfloat16)
            self.clip_ln_final.to(th.bfloat16)
            self.clip_text_projection.data = self.clip_text_projection.data.bfloat16()
        else:
            self.transformer.apply(convert_module_to_bf16)
            self.token_embedding.to(th.bfloat16)
            self.positional_embedding.data = self.positional_embedding.data.bfloat16()
        self.text_proj.apply(convert_module_to_bf16)
        self.text_token_proj.apply(convert_module_to_bf16)
        self.blocks.apply(convert_module_to_bf16)
        self.final_layer.linear.apply(convert_module_to_bf16)
        self.final_layer.adaLN_modulation.apply(convert_module_to_bf16)

    # ----- Text encoder freezing -----

    def freeze_text_encoder(self):
        """Freeze the text encoder core, keeping bridge layers trainable.

        For CLIP: freezes clip_text_model.* (all of it).
        For GLIDE: freezes transformer, embeddings, LN.
        text_proj and text_token_proj always stay trainable.
        """
        frozen = 0
        if self.text_encoder == "clip":
            for name, param in self.named_parameters():
                if name.startswith("clip_"):
                    param.requires_grad = False
                    frozen += 1
        else:
            for name, param in self.named_parameters():
                if any(
                    name.startswith(prefix)
                    for prefix in [
                        "token_embedding",
                        "positional_embedding",
                        "transformer.",
                        "final_ln.",
                    ]
                ):
                    param.requires_grad = False
                    frozen += 1
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        total = sum(1 for p in self.parameters())
        encoder_name = "CLIP ViT-L/14" if self.text_encoder == "clip" else "GLIDE"
        print(
            f"JiT: froze {frozen} {encoder_name} text encoder params "
            f"({trainable}/{total} params still trainable)"
        )

    # ----- Weight initialization from GLIDE -----

    def init_text_from_glide(self, state_dict: dict):
        """Copy text encoder weights from a GLIDE checkpoint.

        Loads: transformer, final_ln, token_embedding, positional_embedding.
        Does NOT load: text_proj, text_token_proj (different architecture).
        Skipped entirely when using CLIP text encoder.
        """
        if self.text_encoder == "clip":
            print("JiT: skipping GLIDE text init (using CLIP text encoder)")
            return
        # Strip _orig_mod. prefix if present (torch.compile)
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }

        own = self.state_dict()
        loaded = 0
        for name in [
            "token_embedding.weight",
            "positional_embedding",
            "final_ln.weight",
            "final_ln.bias",
        ]:
            if name in state_dict and name in own:
                own[name].copy_(state_dict[name])
                loaded += 1

        # Transformer blocks
        for key in state_dict:
            if key.startswith("transformer.") and key in own:
                own[key].copy_(state_dict[key])
                loaded += 1

        print(f"JiT: loaded {loaded} text encoder tensors from GLIDE checkpoint")

    # ----- Forward pass -----

    def forward(
        self,
        z_t: th.Tensor,
        t: th.Tensor,
        tokens: Optional[th.Tensor] = None,
        mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Forward pass: predict clean image x from noisy z_t at time t.

        Args:
            z_t: Noisy image [B, 3, 64, 64], values in [-1, 1].
            t: Time values [B], in [0, 1] (0=noise, 1=clean).
            tokens: Text token IDs [B, text_ctx], optional.
            mask: Text attention mask [B, text_ctx], optional.

        Returns:
            x_pred: Predicted clean image [B, 3, 64, 64].
        """
        B = z_t.shape[0]

        # Patchify and embed
        x = self.patchify(z_t.to(self.dtype))  # [B, 256, 48]
        x = self.patch_embed(x)  # [B, 256, hidden]
        x = x + self.pos_embed

        # Time conditioning
        # Scale t from [0,1] to [0,1000] for sinusoidal embedding compatibility
        t_scaled = t.float() * 1000.0
        t_emb = timestep_embedding(t_scaled, self.hidden_dim)  # [B, hidden]
        cond = self.time_embed(t_emb.to(self.dtype))  # [B, cond_dim]

        # Text conditioning
        if tokens is not None:
            text_out = self.get_text_emb(tokens, mask)
            cond = cond + text_out["xf_pooled"].to(cond)
            context = text_out["xf_tokens"].to(x.dtype)  # [B, text_ctx, hidden]
        else:
            # Unconditional: use zero context
            context = th.zeros(B, 1, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond, context)

        # Final layer → unpatchify
        x = self.final_layer(x, cond)  # [B, 256, patch_dim]
        x = self.unpatchify(x)  # [B, 3, 64, 64]

        return x.to(z_t.dtype)
