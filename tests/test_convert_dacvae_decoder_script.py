from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import scripts.convert_dacvae_decoder as convert_dacvae_decoder
from irodori_mlx.runtime import DACVAEBridgeConfig, describe_codec_capabilities, inspect_mlx_codec_artifact


class ConvertDACVAEDecoderScriptTests(unittest.TestCase):
    def _state_dict(self):
        return {
            "quantizer.out_proj.bias": np.zeros((64,), dtype=np.float32),
            "quantizer.out_proj.parametrizations.weight.original0": np.ones((64, 32, 1), dtype=np.float32),
            "decoder.final.parametrizations.weight.original0": np.ones((1, 64, 7), dtype=np.float32),
            "decoder.final.bias": np.zeros((1,), dtype=np.float32),
            "encoder.block.0.weight": np.ones((64, 1, 7), dtype=np.float32),
        }

    def _args(self, *, dry_run: bool = False):
        return SimpleNamespace(
            source_repo=convert_dacvae_decoder.DEFAULT_CODEC_REPO,
            source_revision="hf-commit-for-test",
            source_file=convert_dacvae_decoder.DEFAULT_SOURCE_FILE,
            dacvae_revision="dacvae-commit-for-test",
            converter_commit="converter-commit-for-test",
            license_review_status="pending",
            license_review_ref=None,
            dry_run=dry_run,
        )

    def test_extract_decoder_tensors_keeps_only_decode_groups(self):
        tensors = convert_dacvae_decoder.extract_decoder_tensors(self._state_dict())

        self.assertIn("quantizer.out_proj.bias", tensors)
        self.assertIn("decoder.final.bias", tensors)
        self.assertNotIn("encoder.block.0.weight", tensors)
        self.assertEqual(tensors["quantizer.out_proj.parametrizations.weight.original0"].shape, (64, 32, 1))

    def test_extract_decoder_tensors_rejects_missing_quantizer_out_projection(self):
        with self.assertRaisesRegex(convert_dacvae_decoder.DACVAEDecoderConversionError, "quantizer.out_proj"):
            convert_dacvae_decoder.extract_decoder_tensors({"decoder.final.weight": np.ones((1, 64, 7))})

    def test_convert_writes_deterministic_real_decoder_manifest_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "weights.pth"
            output = root / "dacvae-codec.npz"
            source.write_bytes(b"fake torch checkpoint")

            fake_torch = SimpleNamespace(load=lambda path, map_location=None: {"state_dict": self._state_dict()})
            with mock.patch.object(convert_dacvae_decoder, "import_torch", return_value=fake_torch):
                report = convert_dacvae_decoder.convert(source, output, self._args())

            self.assertEqual(report["artifact_kind"], "real_semantic_dacvae_decoder")
            self.assertEqual(report["sample_rate"], 48000)
            self.assertTrue(output.exists())

            artifact = inspect_mlx_codec_artifact(output)
            self.assertEqual(artifact["artifact_kind"], "real_semantic_dacvae_decoder")
            self.assertTrue(artifact["has_real_dacvae_decode"])
            self.assertFalse(artifact["has_mlx_decode"])
            self.assertEqual(artifact["real_dacvae_decode_tensor_count"], 4)

            metadata = artifact["metadata"]
            self.assertEqual(metadata["source_revision"], "hf-commit-for-test")
            self.assertEqual(metadata["dacvae_revision"], "dacvae-commit-for-test")
            self.assertEqual(metadata["license_review_status"], "pending")
            self.assertEqual(metadata["runtime_status"]["mlx_decoder_execution"], "blocked")
            self.assertEqual(
                metadata["tensor_manifest_sha256"],
                convert_dacvae_decoder.manifest_digest(metadata["tensors"]),
            )

    def test_capability_report_explains_real_decoder_execution_blocker(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "real-decoder.npz"
            metadata = {
                "artifact_format": "irodori-tts-mlx-dacvae-codec",
                "artifact_format_version": "0.2",
                "artifact_kind": "real_semantic_dacvae_decoder",
                "sample_rate": 48000,
                "hop_length": 512,
                "latent_dim": 32,
                "tensors": [{"name": "decoder.final.bias"}],
            }
            np.savez(
                path,
                sample_rate=np.array(48000),
                hop_length=np.array(512),
                latent_dim=np.array(32),
                metadata_json=np.array(json.dumps(metadata)),
                **{"dacvae_decoder/decoder.final.bias": np.zeros((1,), dtype=np.float32)},
            )

            report = describe_codec_capabilities(DACVAEBridgeConfig(runtime_mode="mlx-decode", codec_path=str(path)))

        self.assertFalse(report["mlx_decode_available"])
        self.assertIn("real Semantic-DACVAE decoder tensors", "\n".join(report["messages"]))
        self.assertIn("not yet implement", "\n".join(report["messages"]))


if __name__ == "__main__":
    unittest.main()

