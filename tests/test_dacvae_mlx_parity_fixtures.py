from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.runtime import DACVAEBridgeConfig, MLXDACVAEBridge

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


def require_decode_fixture_env(test_func):
    required = (
        "IRODORI_MLX_DACVAE_CODEC_NPZ",
        "IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY",
        "IRODORI_MLX_DACVAE_DECODE_AUDIO_NPY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    return unittest.skipIf(missing, "DACVAE decode parity fixture env vars not set: " + ", ".join(missing))(test_func)


def require_encode_fixture_env(test_func):
    required = (
        "IRODORI_MLX_DACVAE_CODEC_NPZ",
        "IRODORI_MLX_DACVAE_ENCODE_AUDIO_WAV",
        "IRODORI_MLX_DACVAE_ENCODE_LATENTS_NPY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    return unittest.skipIf(missing, "DACVAE encode parity fixture env vars not set: " + ", ".join(missing))(test_func)


class MLXDACVAEParityFixtureTests(unittest.TestCase):
    @require_mlx
    @require_decode_fixture_env
    def test_mlx_codec_matches_upstream_generated_decode_fixture(self):
        bridge = MLXDACVAEBridge(
            config=DACVAEBridgeConfig(
                runtime_mode="mlx",
                codec_path=os.environ["IRODORI_MLX_DACVAE_CODEC_NPZ"],
            ),
            require_encode=False,
        )

        latents = mx.array(np.load(os.environ["IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY"]).astype("float32"))
        expected_audio = np.load(os.environ["IRODORI_MLX_DACVAE_DECODE_AUDIO_NPY"]).astype("float32")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "decoded.wav"
            bridge.decode_to_wav(latents, out, max_samples=int(expected_audio.shape[-1]))
            import soundfile as sf

            decoded, sample_rate = sf.read(str(out), dtype="float32")
        self.assertEqual(sample_rate, bridge.sample_rate)
        np.testing.assert_allclose(decoded[: expected_audio.shape[-1]], expected_audio, atol=1e-3, rtol=1e-3)

    @require_mlx
    @require_encode_fixture_env
    def test_mlx_codec_matches_upstream_generated_encode_fixture(self):
        bridge = MLXDACVAEBridge(
            config=DACVAEBridgeConfig(
                runtime_mode="mlx",
                codec_path=os.environ["IRODORI_MLX_DACVAE_CODEC_NPZ"],
            )
        )

        encoded = bridge.encode_reference(
            os.environ["IRODORI_MLX_DACVAE_ENCODE_AUDIO_WAV"],
            max_seconds=None,
            normalize_db=None,
            ensure_max=False,
        )
        expected_latents = np.load(os.environ["IRODORI_MLX_DACVAE_ENCODE_LATENTS_NPY"]).astype("float32")
        np.testing.assert_allclose(np.array(encoded), expected_latents, atol=1e-3, rtol=1e-3)
