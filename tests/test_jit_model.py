"""Unit tests for JiT (Just image Transformer) model architecture."""

import torch as th
from glide_finetune.jit_model import JiTModel, JIT_CONFIGS


def _make_jit_model(**kwargs):
    """Create a small JiT model for testing."""
    defaults = dict(
        image_size=64,
        patch_size=4,
        in_channels=3,
        hidden_dim=64,
        depth=2,
        heads=4,
        bottleneck_dim=32,
        text_ctx=128,
        xf_width=64,
        xf_layers=2,
        xf_heads=4,
    )
    defaults.update(kwargs)
    return JiTModel(**defaults)


class TestForwardPass:
    def test_output_shape(self):
        model = _make_jit_model()
        x = th.randn(2, 3, 64, 64)
        t = th.tensor([0.3, 0.7])
        tokens = th.randint(0, 100, (2, 128))
        mask = th.ones(2, 128, dtype=th.bool)

        out = model(x, t, tokens=tokens, mask=mask)
        assert out.shape == (2, 3, 64, 64)

    def test_unconditional_forward(self):
        """Forward pass without text tokens should work."""
        model = _make_jit_model()
        x = th.randn(2, 3, 64, 64)
        t = th.tensor([0.5, 0.5])

        out = model(x, t, tokens=None, mask=None)
        assert out.shape == (2, 3, 64, 64)

    def test_single_batch(self):
        model = _make_jit_model()
        x = th.randn(1, 3, 64, 64)
        t = th.tensor([0.5])
        tokens = th.randint(0, 100, (1, 128))
        mask = th.ones(1, 128, dtype=th.bool)

        out = model(x, t, tokens=tokens, mask=mask)
        assert out.shape == (1, 3, 64, 64)


class TestPatchifyUnpatchify:
    def test_roundtrip(self):
        model = _make_jit_model()
        x = th.randn(2, 3, 64, 64)
        patches = model.patchify(x)
        assert patches.shape == (2, 256, 48)  # 256 patches, 3*4*4=48 dim

        reconstructed = model.unpatchify(patches)
        assert reconstructed.shape == (2, 3, 64, 64)
        assert th.allclose(x, reconstructed)

    def test_patchify_shape(self):
        model = _make_jit_model(patch_size=8)
        x = th.randn(2, 3, 64, 64)
        patches = model.patchify(x)
        # 64/8 = 8, 8*8 = 64 patches, 3*8*8 = 192 dim
        assert patches.shape == (2, 64, 192)


class TestZeroInit:
    def test_final_layer_zero_init(self):
        """FinalLayer linear output should be zero-initialized."""
        model = _make_jit_model()
        assert th.all(model.final_layer.linear.weight == 0)
        assert th.all(model.final_layer.linear.bias == 0)

    def test_final_layer_adaLN_zero_init(self):
        """FinalLayer adaLN modulation should be zero-initialized."""
        model = _make_jit_model()
        adaLN_linear = model.final_layer.adaLN_modulation[1]
        assert th.all(adaLN_linear.weight == 0)
        assert th.all(adaLN_linear.bias == 0)

    def test_dit_block_adaLN_zero_init(self):
        """DiTBlock adaLN modulation outputs should be zero-initialized."""
        model = _make_jit_model()
        for block in model.blocks:
            adaLN_linear = block.adaLN_modulation[1]
            assert th.all(adaLN_linear.weight == 0)
            assert th.all(adaLN_linear.bias == 0)

    def test_initial_output_near_zero(self):
        """Due to zero-init, initial model output should be near zero."""
        model = _make_jit_model()
        model.eval()
        x = th.randn(1, 3, 64, 64)
        t = th.tensor([0.5])
        with th.no_grad():
            out = model(x, t)
        assert out.abs().max() < 1e-5, (
            f"Initial output should be near zero, got max {out.abs().max()}"
        )


class TestPrecisionConversion:
    def test_bf16_conversion(self):
        model = _make_jit_model()
        model.convert_to_bf16()
        assert model.dtype == th.bfloat16

    def test_bf16_forward(self):
        model = _make_jit_model()
        model.convert_to_bf16()
        x = th.randn(1, 3, 64, 64)
        t = th.tensor([0.5])
        tokens = th.randint(0, 100, (1, 128))
        mask = th.ones(1, 128, dtype=th.bool)

        out = model(x, t, tokens=tokens, mask=mask)
        assert out.shape == (1, 3, 64, 64)
        assert th.isfinite(out).all()

    def test_fp16_conversion(self):
        model = _make_jit_model()
        model.convert_to_fp16()
        assert model.dtype == th.float16


class TestTextEncoder:
    def test_text_encoding(self):
        model = _make_jit_model()
        tokens = th.randint(0, 100, (2, 128))
        mask = th.ones(2, 128, dtype=th.bool)

        out = model.get_text_emb(tokens, mask)
        assert "xf_pooled" in out
        assert "xf_tokens" in out
        # xf_pooled should be [B, cond_dim]
        assert out["xf_pooled"].shape == (2, 64 * 4)  # hidden_dim * 4
        # xf_tokens should be [B, text_ctx, hidden_dim]
        assert out["xf_tokens"].shape == (2, 128, 64)

    def test_text_caching(self):
        model = _make_jit_model()
        model.cache_text_emb = True
        tokens = th.randint(0, 100, (2, 128))
        mask = th.ones(2, 128, dtype=th.bool)

        model.get_text_emb(tokens, mask)
        model.get_text_emb(tokens, mask)
        # Second call should use cache
        assert model.cache is not None
        model.del_cache()
        assert model.cache is None

    def test_tokenizer_exists(self):
        model = _make_jit_model()
        assert hasattr(model, "tokenizer")
        assert hasattr(model.tokenizer, "encode")
        assert hasattr(model.tokenizer, "padded_tokens_and_mask")


class TestConfigs:
    def test_b_config(self):
        cfg = JIT_CONFIGS["B"]
        assert cfg["hidden_dim"] == 768
        assert cfg["depth"] == 12
        assert cfg["heads"] == 12

    def test_all_configs_present(self):
        for key in ["B", "L", "H", "G"]:
            assert key in JIT_CONFIGS
            cfg = JIT_CONFIGS[key]
            assert "hidden_dim" in cfg
            assert "depth" in cfg
            assert "heads" in cfg
            assert "patch_size" in cfg
            assert "bottleneck_dim" in cfg
