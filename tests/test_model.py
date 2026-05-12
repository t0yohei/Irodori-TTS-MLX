from __future__ import annotations

import unittest

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.config import ModelConfig
    from irodori_mlx.model import JointAttention, TextToLatentRFDiT
    from irodori_mlx.weights import assign_named_weights, rf_dit_required_keys

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only on machines without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


def to_np(value):
    return np.array(value)


class RFDiTModelTests(unittest.TestCase):
    def tiny_config(self, *, caption: bool = False) -> ModelConfig:
        return ModelConfig(
            latent_dim=4,
            latent_patch_size=1,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=1.5,
            text_vocab_size=32,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            speaker_dim=8,
            speaker_layers=1,
            speaker_heads=2,
            timestep_embed_dim=8,
            adaln_rank=2,
            norm_eps=1e-5,
            dropout=0.0,
            use_caption_condition=caption,
            caption_vocab_size=32 if caption else None,
            caption_dim=8 if caption else None,
            caption_layers=1 if caption else None,
            caption_heads=2 if caption else None,
        )

    @require_mlx
    def test_joint_attention_context_cache_matches_direct_projection(self):
        attn = JointAttention(
            dim=8,
            heads=2,
            text_ctx_dim=8,
            speaker_ctx_dim=8,
            caption_ctx_dim=None,
            norm_eps=1e-5,
        )
        x = mx.array(np.linspace(-0.2, 0.3, 1 * 3 * 8, dtype=np.float32).reshape(1, 3, 8))
        text = mx.array(np.linspace(-0.1, 0.2, 1 * 2 * 8, dtype=np.float32).reshape(1, 2, 8))
        speaker = mx.array(np.linspace(0.05, 0.25, 1 * 2 * 8, dtype=np.float32).reshape(1, 2, 8))
        text_mask = mx.array([[True, False]])
        speaker_mask = mx.array([[True, True]])
        freqs = mx.ones((3, 2), dtype=mx.complex64)

        direct = attn(
            x=x,
            text_context=text,
            text_mask=text_mask,
            speaker_context=speaker,
            speaker_mask=speaker_mask,
            caption_context=None,
            caption_mask=None,
            freqs_cis=freqs,
        )
        cache = attn.project_context_kv(text_context=text, speaker_context=speaker)
        cached = attn(
            x=x,
            text_context=text,
            text_mask=text_mask,
            speaker_context=speaker,
            speaker_mask=speaker_mask,
            caption_context=None,
            caption_mask=None,
            freqs_cis=freqs,
            context_kv=cache,
        )
        np.testing.assert_allclose(to_np(cached), to_np(direct), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_full_model_forward_runs_with_encoded_conditions_and_cache(self):
        model = TextToLatentRFDiT(self.tiny_config())
        x_t = mx.array(np.ones((1, 2, 4), dtype=np.float32))
        t = mx.array([0.5], dtype=mx.float32)
        encoded = model.encode_conditions(
            text_input_ids=mx.array([[1, 2, 0]], dtype=mx.int32),
            text_mask=mx.array([[True, True, False]]),
            ref_latent=mx.array(np.ones((1, 2, 4), dtype=np.float32)),
            ref_mask=mx.array([[True, True]]),
        )
        direct = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=encoded.text_state,
            text_mask=encoded.text_mask,
            speaker_state=encoded.speaker_state,
            speaker_mask=encoded.speaker_mask,
        )
        cache = model.build_context_kv_cache(text_state=encoded.text_state, speaker_state=encoded.speaker_state)
        cached = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=encoded.text_state,
            text_mask=encoded.text_mask,
            speaker_state=encoded.speaker_state,
            speaker_mask=encoded.speaker_mask,
            context_kv_cache=cache,
        )
        self.assertEqual(direct.shape, (1, 2, 4))
        self.assertTrue(np.isfinite(to_np(direct)).all())
        np.testing.assert_allclose(to_np(cached), to_np(direct), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_full_forward_patches_and_unpatches_latents(self):
        cfg = self.tiny_config()
        model = TextToLatentRFDiT(cfg)
        out = model(
            x_t=mx.array(np.ones((1, 2, 4), dtype=np.float32)),
            t=mx.array([0.25], dtype=mx.float32),
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=mx.array(np.ones((1, 2, 4), dtype=np.float32)),
            ref_mask=mx.array([[True, True]]),
        )
        self.assertEqual(out.shape, (1, 2, 4))
        self.assertEqual(out.dtype, mx.float32)
        self.assertTrue(np.isfinite(to_np(out)).all())

    @require_mlx
    def test_required_keys_cover_rf_dit_root_weights_and_assign(self):
        cfg = self.tiny_config()
        model = TextToLatentRFDiT(cfg)
        keys = rf_dit_required_keys(cfg)
        self.assertIn("blocks.0.attention.wk_text.weight", keys)
        self.assertIn("blocks.0.attention.wk_speaker.weight", keys)
        self.assertIn("cond_module.2.weight", keys)
        self.assertIn("cond_module.4.weight", keys)
        self.assertIn("out_proj.bias", keys)

        subset = {
            "cond_module.0.weight": mx.zeros_like(model.cond_module[0].weight),
            "cond_module.4.weight": mx.zeros_like(model.cond_module[4].weight),
            "in_proj.bias": mx.zeros_like(model.in_proj.bias),
            "out_proj.weight": mx.zeros_like(model.out_proj.weight),
        }
        report = assign_named_weights(model, subset, strict=False)
        self.assertEqual(set(report.assigned), set(subset))


if __name__ == "__main__":
    unittest.main()
