from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx import (
        ConditionEncoders,
        ModelConfig,
        ReferenceLatentEncoder,
        TextEncoder,
        assign_named_weights,
        encoder_required_keys,
        load_npz_weights,
        masked_mean_token,
    )
    from irodori_mlx.encoders import SelfAttention
    from irodori_mlx.layers import precompute_freqs_cis

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only on machines without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


def to_np(value):
    return np.array(value)


class ConfigTests(unittest.TestCase):
    def test_base_and_caption_properties(self):
        base = ModelConfig()
        self.assertTrue(base.use_speaker_condition)
        self.assertEqual(base.patched_latent_dim, 32)
        self.assertEqual(base.speaker_patched_latent_dim, 32)
        self.assertEqual(base.text_mlp_ratio_resolved, 2.6)

        caption = ModelConfig(use_caption_condition=True, caption_dim=None, caption_layers=None)
        self.assertFalse(caption.use_speaker_condition)
        self.assertEqual(caption.caption_dim_resolved, caption.text_dim)
        self.assertEqual(caption.caption_layers_resolved, caption.text_layers)

    def test_duration_predictor_config_normalizes_and_validates(self):
        cfg = ModelConfig(
            text_dim=8,
            speaker_dim=8,
            use_duration_predictor=True,
            duration_attention_heads=2,
            duration_architecture=" Token_Sum_AdaRN_Zero_No_Aux ",
            duration_speaker_fusion=" AdaRN_Zero ",
        )
        self.assertEqual(cfg.duration_architecture, "token_sum_adarn_zero_no_aux")
        self.assertEqual(cfg.duration_speaker_fusion, "adarn_zero")

        with self.assertRaisesRegex(ValueError, "duration_attention_heads"):
            ModelConfig(text_dim=10, speaker_dim=8, use_duration_predictor=True, duration_attention_heads=3)
        with self.assertRaisesRegex(ValueError, "duration_architecture"):
            ModelConfig(
                text_dim=8,
                speaker_dim=8,
                use_duration_predictor=True,
                duration_attention_heads=2,
                duration_architecture="pooled",
            )
        with self.assertRaisesRegex(ValueError, "speaker-conditioned"):
            ModelConfig(
                text_dim=8,
                use_caption_condition=True,
                caption_vocab_size=32,
                caption_dim=8,
                caption_layers=1,
                caption_heads=2,
                use_duration_predictor=True,
                duration_attention_heads=2,
            )


class EncoderRuntimeTests(unittest.TestCase):
    @require_mlx
    def test_self_attention_all_false_mask_is_finite_in_float16(self):
        attention = SelfAttention(dim=8, heads=2, norm_eps=1e-5)
        attention.set_dtype(mx.float16)
        x = mx.ones((1, 3, 8), dtype=mx.float16)
        mask = mx.array([[False, False, False]])
        freqs = precompute_freqs_cis(dim=4, end=3)

        output = attention(x, key_mask=mask, freqs_cis=freqs)

        self.assertFalse(bool(mx.any(mx.isnan(output)).item()))
        np.testing.assert_allclose(to_np(output), 0.0, atol=1e-6)

    @require_mlx
    def test_text_encoder_masks_outputs_to_zero(self):
        encoder = TextEncoder(
            vocab_size=16,
            dim=8,
            layers=1,
            heads=2,
            mlp_ratio=1.5,
            norm_eps=1e-5,
        )
        input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        mask = mx.array([[True, False, True, False]])
        output = encoder(input_ids, mask)
        self.assertEqual(output.shape, (1, 4, 8))
        np.testing.assert_allclose(to_np(output)[:, [1, 3], :], 0.0, atol=1e-6)

    @require_mlx
    def test_reference_encoder_shape_and_masking(self):
        cfg = ModelConfig(
            latent_dim=4,
            latent_patch_size=1,
            speaker_dim=12,
            speaker_layers=1,
            speaker_heads=3,
            speaker_mlp_ratio=1.5,
        )
        encoder = ReferenceLatentEncoder(cfg)
        latent = mx.array(np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4))
        mask = mx.array([[True, False, True]])
        output = encoder(latent, mask)
        self.assertEqual(output.shape, (1, 3, 12))
        np.testing.assert_allclose(to_np(output)[:, 1, :], 0.0, atol=1e-6)

    @require_mlx
    def test_masked_mean_token_handles_unconditional_rows(self):
        state = mx.array(
            [
                [[2.0, 4.0], [6.0, 8.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ],
            dtype=mx.float32,
        )
        mask = mx.array([[True, False], [False, False]])
        out_state, out_mask = masked_mean_token(state, mask)
        self.assertEqual(out_state.shape, (2, 3, 2))
        np.testing.assert_allclose(to_np(out_state)[0, 0], np.array([2.0, 4.0]), atol=1e-6)
        np.testing.assert_allclose(to_np(out_state)[1, 0], np.array([0.0, 0.0]), atol=1e-6)
        np.testing.assert_array_equal(to_np(out_mask), np.array([[True, True, False], [False, False, False]]))

    @require_mlx
    def test_condition_encoders_base_masks_and_speaker_summary(self):
        cfg = ModelConfig(
            latent_dim=4,
            text_vocab_size=32,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            text_mlp_ratio=1.5,
            speaker_dim=12,
            speaker_layers=1,
            speaker_heads=3,
            speaker_mlp_ratio=1.5,
        )
        encoders = ConditionEncoders(cfg)
        encoded = encoders(
            text_input_ids=mx.array([[1, 2, 3]], dtype=mx.int32),
            text_mask=mx.array([[True, True, False]]),
            ref_latent=mx.array(np.ones((1, 2, 4), dtype=np.float32)),
            ref_mask=mx.array([[True, False]]),
        )
        self.assertEqual(encoded.text_state.shape, (1, 3, 8))
        self.assertEqual(encoded.speaker_state.shape, (1, 3, 12))
        np.testing.assert_array_equal(to_np(encoded.speaker_mask), np.array([[True, True, False]]))
        np.testing.assert_allclose(to_np(encoded.text_state)[:, 2, :], 0.0, atol=1e-6)

    @require_mlx
    def test_condition_dropout_produces_unconditional_masks(self):
        cfg = ModelConfig(
            latent_dim=4,
            text_vocab_size=32,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            text_mlp_ratio=1.5,
            speaker_dim=12,
            speaker_layers=1,
            speaker_heads=3,
            speaker_mlp_ratio=1.5,
        )
        encoders = ConditionEncoders(cfg)
        encoded = encoders(
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=mx.array(np.ones((1, 2, 4), dtype=np.float32)),
            ref_mask=mx.array([[True, True]]),
            text_condition_dropout=mx.array([True]),
            speaker_condition_dropout=mx.array([True]),
        )
        np.testing.assert_array_equal(to_np(encoded.text_mask), np.array([[False, False]]))
        np.testing.assert_allclose(to_np(encoded.text_state), 0.0, atol=1e-6)
        np.testing.assert_array_equal(to_np(encoded.speaker_mask), np.array([[False, False, False]]))
        np.testing.assert_allclose(to_np(encoded.speaker_state), 0.0, atol=1e-6)

    @require_mlx
    def test_caption_condition_path(self):
        cfg = ModelConfig(
            use_caption_condition=True,
            text_vocab_size=32,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            text_mlp_ratio=1.5,
            caption_vocab_size=32,
            caption_dim=8,
            caption_layers=1,
            caption_heads=2,
            caption_mlp_ratio=1.5,
        )
        encoders = ConditionEncoders(cfg)
        encoded = encoders(
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, False]]),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=mx.array([[3, 4, 5]], dtype=mx.int32),
            caption_mask=mx.array([[True, True, False]]),
        )
        self.assertIsNone(encoded.speaker_state)
        self.assertEqual(encoded.caption_state.shape, (1, 3, 8))
        np.testing.assert_allclose(to_np(encoded.caption_state)[:, 2, :], 0.0, atol=1e-6)

    @require_mlx
    def test_assign_named_weights_and_npz_loader(self):
        encoder = TextEncoder(
            vocab_size=4,
            dim=4,
            layers=0,
            heads=1,
            mlp_ratio=1.0,
            norm_eps=1e-5,
        )
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.npz"
            np.savez(path, **{"text_embedding.weight": expected})
            weights = load_npz_weights(path)
        report = assign_named_weights(encoder, weights, required=("text_embedding.weight",))
        self.assertEqual(report.assigned, ("text_embedding.weight",))
        np.testing.assert_array_equal(to_np(encoder.text_embedding.weight), expected)

    def test_required_key_builder_includes_observed_encoder_keys(self):
        keys = encoder_required_keys(
            prefix="text_encoder",
            layers=1,
            dim=512,
            heads=8,
            mlp_ratio=2.6,
            has_embedding=True,
            has_input_projection=False,
        )
        self.assertIn("text_encoder.text_embedding.weight", keys)
        self.assertIn("text_encoder.blocks.0.attention.wq.weight", keys)
        self.assertIn("text_encoder.blocks.0.mlp.w3.weight", keys)


if __name__ == "__main__":
    unittest.main()
