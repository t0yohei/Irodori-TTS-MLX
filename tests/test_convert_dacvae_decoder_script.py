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


def executable_decoder_manifest(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    prefix = "dacvae_decoder_exec/"
    return [
        {
            "target_name": key[len(prefix) :],
            "artifact_key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "parameter_count": int(value.size),
        }
        for key, value in sorted(arrays.items())
        if key.startswith(prefix)
    ]


class ConvertDACVAEDecoderScriptTests(unittest.TestCase):
    def _state_dict(self):
        state = {
            "quantizer.out_proj.bias": np.zeros((1024,), dtype=np.float32),
            "quantizer.out_proj.parametrizations.weight.original0": np.ones((1024, 1, 1), dtype=np.float32),
            "quantizer.out_proj.parametrizations.weight.original1": np.ones((1024, 32, 1), dtype=np.float32),
            "decoder.model.0.bias": np.zeros((1536,), dtype=np.float32),
            "decoder.model.0.parametrizations.weight.original0": np.ones((1536, 1, 1), dtype=np.float32),
            "decoder.model.0.parametrizations.weight.original1": np.ones((1536, 1024, 7), dtype=np.float32),
            "decoder.wm_model.encoder_block.pre.0.alpha": np.ones((1, 96, 1), dtype=np.float32),
            "decoder.wm_model.encoder_block.pre.1.bias": np.zeros((1,), dtype=np.float32),
            "decoder.wm_model.encoder_block.pre.1.parametrizations.weight.original0": np.ones((1, 1, 1), dtype=np.float32),
            "decoder.wm_model.encoder_block.pre.1.parametrizations.weight.original1": np.ones((1, 96, 7), dtype=np.float32),
            "encoder.block.0.weight": np.ones((64, 1, 7), dtype=np.float32),
        }
        dims = [1536, 768, 384, 192, 96]
        for index, stride in enumerate((12, 10, 8, 2)):
            in_dim = dims[index]
            out_dim = dims[index + 1]
            prefix = f"decoder.model.{index + 1}.block"
            state[f"{prefix}.0.alpha"] = np.ones((1, in_dim, 1), dtype=np.float32)
            state[f"{prefix}.1.bias"] = np.zeros((out_dim,), dtype=np.float32)
            state[f"{prefix}.1.parametrizations.weight.original0"] = np.ones((in_dim, 1, 1), dtype=np.float32)
            state[f"{prefix}.1.parametrizations.weight.original1"] = np.ones((in_dim, out_dim, 2 * stride), dtype=np.float32)
            for source_block in ("4", "5", "8"):
                state[f"{prefix}.{source_block}.block.0.alpha"] = np.ones((1, out_dim, 1), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.1.bias"] = np.zeros((out_dim,), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.1.parametrizations.weight.original0"] = np.ones((out_dim, 1, 1), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.1.parametrizations.weight.original1"] = np.ones((out_dim, out_dim, 7), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.2.alpha"] = np.ones((1, out_dim, 1), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.3.bias"] = np.zeros((out_dim,), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.3.parametrizations.weight.original0"] = np.ones((out_dim, 1, 1), dtype=np.float32)
                state[f"{prefix}.{source_block}.block.3.parametrizations.weight.original1"] = np.ones((out_dim, out_dim, 1), dtype=np.float32)
        return state

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
        self.assertIn("decoder.model.0.bias", tensors)
        self.assertNotIn("encoder.block.0.weight", tensors)
        self.assertEqual(tensors["quantizer.out_proj.parametrizations.weight.original1"].shape, (1024, 32, 1))

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
            self.assertTrue(artifact["has_mlx_decode"])
            self.assertTrue(artifact["has_executable_mlx_decode"])
            self.assertGreater(artifact["executable_dacvae_decode_tensor_count"], 0)

            metadata = artifact["metadata"]
            self.assertEqual(metadata["source_revision"], "hf-commit-for-test")
            self.assertEqual(metadata["dacvae_revision"], "dacvae-commit-for-test")
            self.assertEqual(metadata["license_review_status"], "pending")
            self.assertEqual(metadata["runtime_status"]["mlx_decoder_execution"], "available_unvalidated")
            self.assertEqual(
                metadata["tensor_manifest_sha256"],
                convert_dacvae_decoder.manifest_digest(metadata["tensors"]),
            )
            self.assertEqual(
                metadata["executable_tensor_manifest_sha256"],
                convert_dacvae_decoder.manifest_digest(metadata["executable_tensors"]),
            )

    def test_executable_tensor_mapping_transposes_weights_and_snake_alpha(self):
        tensors = convert_dacvae_decoder.extract_decoder_tensors(self._state_dict())
        executable = convert_dacvae_decoder.build_executable_decoder_tensors(tensors)

        self.assertEqual(executable["quantizer_out_proj.weight_v"].shape, (1024, 1, 32))
        self.assertEqual(executable["blocks.0.main_upsample.1.weight_v"].shape, (768, 24, 1536))
        self.assertEqual(executable["blocks.0.main_upsample.0.alpha"].shape, (1, 1, 1536))

    def test_capability_report_reports_executable_decoder_support(self):
        from irodori_mlx.dacvae import (
            EXECUTABLE_DECODER_PREFIX,
            SemanticDACVAEDecoderConfig,
            semantic_dacvae_decoder_expected_shapes,
        )

        decoder_config = SemanticDACVAEDecoderConfig(
            latent_dim=8,
            decoder_dim=16,
            decoder_rates=(2,),
            wm_rates=(2,),
            codebook_dim=4,
            output_channels=1,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "real-decoder.npz"
            arrays = {
                EXECUTABLE_DECODER_PREFIX + name: np.zeros(shape, dtype=np.float32)
                for name, shape in semantic_dacvae_decoder_expected_shapes(decoder_config).items()
            }
            metadata = {
                "artifact_format": "irodori-tts-mlx-dacvae-codec",
                "artifact_format_version": "0.2",
                "artifact_kind": "real_semantic_dacvae_decoder",
                "sample_rate": 48000,
                "hop_length": 1920,
                "latent_dim": 4,
                "semantic_dacvae_decoder_config": {
                    "latent_dim": 8,
                    "decoder_dim": 16,
                    "decoder_rates": [2],
                    "wm_rates": [2],
                    "codebook_dim": 4,
                    "output_channels": 1,
                },
                "tensors": [{"name": "decoder.final.bias"}],
                "executable_tensors": executable_decoder_manifest(arrays),
            }
            np.savez(
                path,
                sample_rate=np.array(48000),
                hop_length=np.array(1920),
                latent_dim=np.array(4),
                metadata_json=np.array(json.dumps(metadata)),
                **{"dacvae_decoder/decoder.final.bias": np.zeros((1,), dtype=np.float32)},
                **arrays,
            )

            report = describe_codec_capabilities(DACVAEBridgeConfig(runtime_mode="mlx-decode", codec_path=str(path)))

        self.assertTrue(report["mlx_decode_available"])
        self.assertIn("executable Semantic-DACVAE decoder tensors", "\n".join(report["messages"]))
        self.assertIn("acoustic parity", "\n".join(report["messages"]))


if __name__ == "__main__":
    unittest.main()
