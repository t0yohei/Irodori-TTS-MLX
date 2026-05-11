from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.layers import (
        LowRankAdaLN,
        RMSNorm,
        SwiGLU,
        apply_rotary_emb,
        get_timestep_embedding,
        patch_latents,
        patch_sequence_with_mask,
        precompute_freqs_cis,
        unpatch_latents,
    )

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only on machines without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


def to_np(value):
    return np.array(value)


class LayerFormulaTests(unittest.TestCase):
    @require_mlx
    def test_precompute_freqs_cis_matches_reference(self):
        got = to_np(precompute_freqs_cis(dim=4, end=3))
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float32) / 4.0))
        args = np.arange(3, dtype=np.float32)[:, None] * inv_freq[None, :]
        expected = np.cos(args) + 1j * np.sin(args)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_apply_rotary_emb_matches_reference(self):
        x = np.arange(1 * 3 * 1 * 4, dtype=np.float32).reshape(1, 3, 1, 4) / 10.0
        freqs = precompute_freqs_cis(dim=4, end=3)
        got = to_np(apply_rotary_emb(mx.array(x), freqs))

        freqs_np = to_np(freqs)
        pairs = x.reshape(1, 3, 1, 2, 2)
        real = pairs[..., 0]
        imag = pairs[..., 1]
        cos = freqs_np.real[None, :, None, :]
        sin = freqs_np.imag[None, :, None, :]
        expected = np.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1).reshape(x.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_timestep_embedding_matches_reference_and_dtype(self):
        timesteps = mx.array([0.0, 0.5], dtype=mx.float32)
        got = get_timestep_embedding(timesteps, dim=6)
        half = 3
        freqs = 1000.0 * np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
        args = np.array([0.0, 0.5], dtype=np.float32)[:, None] * freqs[None, :]
        expected = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        self.assertEqual(got.dtype, mx.float32)
        np.testing.assert_allclose(to_np(got), expected, rtol=1e-5, atol=1e-5)

    @require_mlx
    def test_rmsnorm_uses_fp32_stats_and_restores_dtype(self):
        layer = RMSNorm(3, eps=1e-5)
        layer.weight = mx.array([1.0, 1.5, -0.5], dtype=mx.float32)
        x = mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32)
        got = layer(x)
        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        expected = x_np / np.sqrt(np.mean(x_np * x_np, axis=-1, keepdims=True) + 1e-5)
        expected = expected * np.array([1.0, 1.5, -0.5], dtype=np.float32)
        self.assertEqual(got.dtype, mx.float32)
        np.testing.assert_allclose(to_np(got), expected, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_swiglu_with_deterministic_weights(self):
        layer = SwiGLU(dim=2, hidden_dim=3)
        layer.w1.weight = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=mx.float32)
        layer.w2.weight = mx.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 1.5]], dtype=mx.float32)
        layer.w3.weight = mx.array([[0.5, 1.0], [1.5, -0.5], [1.0, 1.0]], dtype=mx.float32)
        x = mx.array([[2.0, -1.0]], dtype=mx.float32)
        got = to_np(layer(x))

        x_np = np.array([[2.0, -1.0]], dtype=np.float32)
        w1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        w2 = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 1.5]], dtype=np.float32)
        w3 = np.array([[0.5, 1.0], [1.5, -0.5], [1.0, 1.0]], dtype=np.float32)
        h1 = x_np @ w1.T
        h3 = x_np @ w3.T
        silu = h1 / (1.0 + np.exp(-h1))
        expected = (silu * h3) @ w2.T
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_low_rank_adaln_zero_up_projection_matches_direct_formula(self):
        layer = LowRankAdaLN(model_dim=2, rank=1, eps=1e-5)
        x = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
        cond = mx.array([[[0.1, -0.2, 0.5, -0.5, 2.0, -2.0]]], dtype=mx.float32)
        got_x, got_gate = layer(x, cond)

        x_np = np.array([[[1.0, 2.0]]], dtype=np.float32)
        shift = np.array([[[0.1, -0.2]]], dtype=np.float32)
        scale = np.array([[[0.5, -0.5]]], dtype=np.float32)
        gate = np.array([[[2.0, -2.0]]], dtype=np.float32)
        norm = x_np / np.sqrt(np.mean(x_np * x_np, axis=-1, keepdims=True) + 1e-5)
        expected_x = norm * (1.0 + scale) + shift
        expected_gate = np.tanh(gate)
        np.testing.assert_allclose(to_np(got_x), expected_x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(to_np(got_gate), expected_gate, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_patch_sequence_with_mask_reduces_mask_by_all(self):
        seq = mx.array(np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3))
        mask = mx.array([[True, True, False, True], [True, True, True, True]])
        patched, patched_mask = patch_sequence_with_mask(seq, mask, patch_size=2)
        self.assertEqual(patched.shape, (2, 2, 6))
        np.testing.assert_array_equal(to_np(patched_mask), np.array([[True, False], [True, True]]))

    @require_mlx
    def test_patch_latents_roundtrip(self):
        latents = mx.array(np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2))
        patched = patch_latents(latents, patch_size=3)
        self.assertEqual(patched.shape, (1, 2, 6))
        roundtrip = unpatch_latents(patched, patch_size=3)
        np.testing.assert_array_equal(to_np(roundtrip), to_np(latents))


if __name__ == "__main__":
    unittest.main()
