from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.dacvae import (
        DACVAEDecoderBlock,
        DACVAEResidualUnit,
        DACVAESnake1d,
        DACVAEWNConv1d,
        DACVAEWNConvTranspose1d,
        SemanticDACVAEDecoder,
        SemanticDACVAEDecoderConfig,
    )

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only on machines without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


def to_np(value):
    return np.array(value)


class DACVAEDecoderModuleTests(unittest.TestCase):
    @require_mlx
    def test_snake_matches_formula(self):
        layer = DACVAESnake1d(2)
        layer.alpha = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
        x = mx.array([[[0.0, 0.5], [1.0, -1.0]]], dtype=mx.float32)

        got = to_np(layer(x))

        x_np = to_np(x)
        alpha = np.array([[[1.0, 2.0]]], dtype=np.float32)
        expected = x_np + np.sin(alpha * x_np) ** 2 / (alpha + 1e-9)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_weight_normalized_conv_can_reconstruct_direct_weight(self):
        x = mx.array(np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2) / 10.0)
        weight = mx.array(
            [
                [[0.5, -0.25], [0.25, 0.75], [-0.5, 0.125]],
                [[-0.25, 0.5], [0.5, -0.75], [0.25, 0.25]],
            ],
            dtype=mx.float32,
        )
        bias = mx.array([0.1, -0.2], dtype=mx.float32)
        direct = DACVAEWNConv1d(2, 2, 3, norm="none")
        direct.weight = weight
        direct.bias = bias

        normalized = DACVAEWNConv1d(2, 2, 3, norm="weight_norm")
        normalized.weight_v = weight
        normalized.weight_g = mx.sqrt(mx.sum(weight * weight, axis=(1, 2), keepdims=True))
        normalized.bias = bias

        np.testing.assert_allclose(to_np(normalized(x)), to_np(direct(x)), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_convtranspose_auto_mode_unpads_causal_tail(self):
        layer = DACVAEWNConvTranspose1d(
            1,
            1,
            kernel_size=4,
            stride=2,
            causal=True,
            pad_mode="auto",
            norm="none",
            bias=False,
        )
        layer.weight = mx.ones((1, 4, 1), dtype=mx.float32)
        x = mx.ones((1, 3, 1), dtype=mx.float32)

        got = layer(x)

        self.assertEqual(got.shape, (1, 6, 1))
        self.assertTrue(np.isfinite(to_np(got)).all())

    @require_mlx
    def test_residual_unit_preserves_channel_last_shape(self):
        layer = DACVAEResidualUnit(4, dilation=3)
        x = mx.ones((2, 9, 4), dtype=mx.float32)

        got = layer(x)

        self.assertEqual(got.shape, x.shape)
        self.assertTrue(np.isfinite(to_np(got)).all())

    @require_mlx
    def test_decoder_block_upsamples_main_path(self):
        block = DACVAEDecoderBlock(input_dim=12, output_dim=6, stride=2, stride_wm=2)
        x = mx.ones((1, 5, 12), dtype=mx.float32)

        got = block(x)

        self.assertEqual(got.shape, (1, 10, 6))
        self.assertTrue(np.isfinite(to_np(got)).all())

    @require_mlx
    def test_semantic_decoder_accepts_runtime_layout_and_outputs_waveform_layout(self):
        decoder = SemanticDACVAEDecoder(
            SemanticDACVAEDecoderConfig(
                latent_dim=8,
                decoder_dim=48,
                decoder_rates=(2, 2),
                wm_rates=(2, 2),
                codebook_dim=4,
                output_channels=1,
            )
        )
        latents = mx.ones((1, 4, 4), dtype=mx.float32)

        got = decoder(latents)

        self.assertEqual(got.shape, (1, 16, 1))
        self.assertTrue(np.isfinite(to_np(got)).all())

    def test_dacvae_module_import_does_not_import_torch(self):
        script = "import sys; import irodori_mlx.dacvae; print('torch' in sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
