from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import scripts.check_dacvae_decode_parity as check_dacvae_decode_parity


class FakeDecodeBridge:
    sample_rate = 8000
    hop_length = 4
    latent_dim = 2

    def __init__(self, *, offset: float = 0.0):
        self.offset = offset

    def decode_to_wav(self, latents, output_path, *, max_samples=None):
        samples = np.asarray(latents, dtype=np.float32).reshape(-1) + np.float32(self.offset)
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output.with_suffix(output.suffix + ".npy"), samples.astype(np.float32))
        output.write_bytes(b"fake wav")
        return output


class DACVAEDecodeParityScriptTests(unittest.TestCase):
    def test_compare_audio_passes_shape_range_and_metric_tolerances(self):
        tolerances = check_dacvae_decode_parity.DecodeParityTolerances(
            max_abs=0.01,
            mean_abs=0.01,
            rmse=0.01,
            min_cosine=0.99,
        )

        result = check_dacvae_decode_parity.compare_audio(
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
            np.array([0.101, -0.199, 0.299], dtype=np.float32),
            sample_rate=8000,
            tolerances=tolerances,
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["checks"]["shape"])
        self.assertTrue(result["checks"]["range"])
        self.assertLess(result["metrics"]["max_abs"], 0.01)

    def test_compare_audio_fails_when_shape_or_tolerances_drift(self):
        tolerances = check_dacvae_decode_parity.DecodeParityTolerances(
            max_abs=0.001,
            mean_abs=0.001,
            rmse=0.001,
            min_cosine=0.9999,
        )

        result = check_dacvae_decode_parity.compare_audio(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
            sample_rate=8000,
            tolerances=tolerances,
        )

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["shape"])
        self.assertFalse(result["checks"]["max_abs"])

    def test_decode_pair_uses_same_latents_for_upstream_and_mlx_and_writes_report_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latents_path = root / "latents.npy"
            codec_path = root / "codec.npz"
            np.save(latents_path, np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32))
            codec_path.write_bytes(b"fake codec")
            args = check_dacvae_decode_parity.parse_args(
                [
                    "--latents-npy",
                    str(latents_path),
                    "--codec-path",
                    str(codec_path),
                    "--output-dir",
                    str(root / "out"),
                    "--max-samples",
                    "4",
                ]
            )

            def fake_audio(path):
                return np.load(Path(path).with_suffix(Path(path).suffix + ".npy")), 8000

            with mock.patch.object(
                check_dacvae_decode_parity, "PyTorchDACVAEBridge", return_value=FakeDecodeBridge(offset=0.0)
            ) as upstream_factory, mock.patch.object(
                check_dacvae_decode_parity, "MLXDACVAEBridge", return_value=FakeDecodeBridge(offset=0.0)
            ) as mlx_factory, mock.patch.object(
                check_dacvae_decode_parity, "_load_audio_numpy", side_effect=fake_audio
            ):
                report = check_dacvae_decode_parity.decode_pair(args)

        upstream_factory.assert_called_once()
        mlx_factory.assert_called_once()
        self.assertEqual(report["comparison"]["status"], "passed")
        self.assertEqual(report["latents"]["shape"], [1, 2, 2])
        self.assertEqual(report["comparison"]["metrics"]["compared_samples"], 4)
        self.assertEqual(report["outputs"]["upstream_wav"].split("/")[-1], "upstream-decode.wav")
        self.assertEqual(report["outputs"]["mlx_wav"].split("/")[-1], "mlx-decode.wav")

    def test_main_returns_nonzero_for_metric_failure_and_persists_json(self):
        with tempfile.TemporaryDirectory() as td:
            report = {
                "comparison": {"status": "failed"},
            }
            with mock.patch.object(check_dacvae_decode_parity, "decode_pair", return_value=report):
                rc = check_dacvae_decode_parity.main(
                    [
                        "--latents-npy",
                        str(Path(td) / "latents.npy"),
                        "--codec-path",
                        str(Path(td) / "codec.npz"),
                        "--output-dir",
                        td,
                    ]
                )

            report_path = Path(td) / "dacvae-decode-parity.json"
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
