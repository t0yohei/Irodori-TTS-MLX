from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from irodori_mlx.runtime import DACVAEBridgeConfig, describe_codec_capabilities, inspect_mlx_codec_artifact


MLX_AUDIO_DACVAE_CONFIG = {
    "sample_rate": 48000,
    "encoder_rates": [2, 8, 10, 12],
    "decoder_rates": [12, 10, 8, 2],
    "n_codebooks": 16,
    "codebook_size": 1024,
    "codebook_dim": 32,
}


def _write_fixture_codec(path: Path, *, include_encode: bool = False) -> None:
    payload: dict[str, object] = {
        "sample_rate": np.array(48000),
        "hop_length": np.array(1920),
        "latent_dim": np.array(32),
        "decode_basis": np.ones((32, 1920), dtype=np.float32),
        "decode_bias": np.zeros((1920,), dtype=np.float32),
        "metadata_json": np.array(
            json.dumps(
                {
                    "source_layout": "mlx-audio dacvae/config.json + dacvae/model.safetensors",
                    "sample_rate": 48000,
                    "hop_length": 1920,
                    "latent_dim": 32,
                    "large_weight_policy": "local-only",
                }
            )
        ),
    }
    if include_encode:
        payload["encode_basis"] = np.ones((1920, 32), dtype=np.float32)
        payload["encode_bias"] = np.zeros((32,), dtype=np.float32)
    np.savez(path, **payload)


class MlxAudioDACVAEContractDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "mlx_audio_dacvae_contract.md").read_text(encoding="utf-8")

    def test_doc_pins_mlx_audio_dacvae_layout_and_config_constants(self):
        for term in (
            "dacvae/config.json",
            "dacvae/model.safetensors",
            '"sample_rate": 48000',
            '"encoder_rates": [2, 8, 10, 12]',
            '"decoder_rates": [12, 10, 8, 2]',
            '"n_codebooks": 16',
            '"codebook_size": 1024',
            '"codebook_dim": 32',
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

        self.assertEqual(MLX_AUDIO_DACVAE_CONFIG["sample_rate"], 48000)
        self.assertEqual(MLX_AUDIO_DACVAE_CONFIG["codebook_dim"], 32)
        self.assertEqual(MLX_AUDIO_DACVAE_CONFIG["encoder_rates"], [2, 8, 10, 12])

    def test_doc_records_shape_channel_and_large_weight_contract(self):
        for term in (
            "audio input/output is mono",
            "(B, T, D)",
            "(B, D, T)",
            "D == 32",
            "generated WAV output is mono",
            "sample rate, hop/step length, latent dimension",
            "Do not commit mlx-audio `dacvae/model.safetensors`",
            "without downloading multi-GiB upstream artifacts",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_recommends_existing_hosted_layout_for_issue_131(self):
        for term in (
            "For #131, the recommendation is explicit",
            "existing hosted Irodori-TTS-MLX layout",
            "do not add direct",
            "dacvae-codec.npz",
            "codec_artifact_layout.md",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_fixture_codec_contract_covers_selected_decode_only_path(self):
        with tempfile.TemporaryDirectory() as td:
            codec_path = Path(td) / "dacvae-codec.npz"
            _write_fixture_codec(codec_path, include_encode=True)

            artifact = inspect_mlx_codec_artifact(codec_path)
            self.assertEqual(artifact["sample_rate"], 48000)
            self.assertEqual(artifact["hop_length"], 1920)
            self.assertEqual(artifact["latent_dim"], 32)
            self.assertTrue(artifact["has_mlx_decode"])
            self.assertTrue(artifact["has_mlx_encode"])
            self.assertEqual(artifact["metadata"]["large_weight_policy"], "local-only")

            report = describe_codec_capabilities(
                DACVAEBridgeConfig(runtime_mode="mlx", codec_path=str(codec_path))
            )
            self.assertTrue(report["requires_codec_artifact"])
            self.assertFalse(report["mlx_decode_available"])
            self.assertFalse(report["mlx_encode_available"])
            self.assertEqual(report["decode_policy"], "mlx-artifact")
            self.assertEqual(report["reference_encode_policy"], "mlx-artifact")


if __name__ == "__main__":
    unittest.main()
