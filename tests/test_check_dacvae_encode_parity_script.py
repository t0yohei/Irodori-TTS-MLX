from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

import scripts.check_dacvae_encode_parity as check_dacvae_encode_parity


class FakeEncodeBridge:
    sample_rate = 8000
    hop_length = 2
    latent_dim = 2

    def __init__(self, *, offset: float = 0.0):
        self.offset = offset
        self.calls = []

    def encode_reference(self, path, *, max_seconds, normalize_db, ensure_max):
        self.calls.append((str(path), max_seconds, normalize_db, ensure_max))
        steps = 2 if max_seconds is None else 1
        base = np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32)[:, :steps, :]
        return base + np.float32(self.offset)


def write_wav(path: Path) -> None:
    pcm = (np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(8000)
        fh.writeframes(pcm.tobytes())


class DACVAEEncodeParityScriptTests(unittest.TestCase):
    def test_compare_latents_passes_shape_length_and_metric_tolerances(self):
        tolerances = check_dacvae_encode_parity.EncodeParityTolerances(
            max_abs=0.01,
            mean_abs=0.01,
            rmse=0.01,
            min_cosine=0.99,
        )

        result = check_dacvae_encode_parity.compare_latents(
            np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32),
            np.array([[[0.101, -0.199], [0.299, -0.401]]], dtype=np.float32),
            hop_length=2,
            tolerances=tolerances,
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["checks"]["latent_steps"])
        self.assertTrue(result["checks"]["latent_dim"])
        self.assertEqual(result["length_contract"]["speaker_mask_true_count_upstream"], 2)
        self.assertLess(result["metrics"]["max_abs"], 0.01)

    def test_compare_latents_fails_when_length_or_tolerances_drift(self):
        tolerances = check_dacvae_encode_parity.EncodeParityTolerances(
            max_abs=0.001,
            mean_abs=0.001,
            rmse=0.001,
            min_cosine=0.9999,
        )

        result = check_dacvae_encode_parity.compare_latents(
            np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
            np.array([[[0.5, 0.5]]], dtype=np.float32),
            hop_length=2,
            tolerances=tolerances,
        )

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["shape"])
        self.assertFalse(result["checks"]["latent_steps"])
        self.assertFalse(result["checks"]["max_abs"])

    def test_encode_pair_uses_same_audio_for_upstream_and_mlx_and_writes_latents(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_path = root / "ref.wav"
            codec_path = root / "codec.npz"
            write_wav(audio_path)
            codec_path.write_bytes(b"fake codec")
            args = check_dacvae_encode_parity.parse_args(
                [
                    "--audio-wav",
                    str(audio_path),
                    "--codec-path",
                    str(codec_path),
                    "--output-dir",
                    str(root / "out"),
                    "--max-seconds",
                    "0.25",
                    "--normalize-db",
                    "-16",
                    "--ensure-max",
                ]
            )

            upstream = FakeEncodeBridge(offset=0.0)
            mlx = FakeEncodeBridge(offset=0.0)
            with mock.patch.object(
                check_dacvae_encode_parity, "PyTorchDACVAEBridge", return_value=upstream
            ) as upstream_factory, mock.patch.object(
                check_dacvae_encode_parity, "MLXDACVAEBridge", return_value=mlx
            ) as mlx_factory:
                report = check_dacvae_encode_parity.encode_pair(args)

        upstream_factory.assert_called_once()
        mlx_factory.assert_called_once()
        self.assertEqual(upstream.calls, [(str(audio_path), 0.25, -16.0, True)])
        self.assertEqual(mlx.calls, [(str(audio_path), 0.25, -16.0, True)])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/155")
        self.assertEqual(report["parent_epic"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/160")
        self.assertEqual(report["comparison"]["status"], "passed")
        self.assertEqual(report["comparison"]["metrics"]["upstream"]["shape"], [1, 1, 2])
        self.assertEqual(report["outputs"]["upstream_latents_npy"].split("/")[-1], "upstream-encode-latents.npy")
        self.assertEqual(report["outputs"]["mlx_latents_npy"].split("/")[-1], "mlx-encode-latents.npy")

    def test_main_returns_nonzero_for_metric_failure_and_persists_json(self):
        with tempfile.TemporaryDirectory() as td:
            report = {
                "status": "failed",
                "comparison": {"status": "failed"},
            }
            with mock.patch.object(check_dacvae_encode_parity, "encode_pair", return_value=report):
                rc = check_dacvae_encode_parity.main(
                    [
                        "--audio-wav",
                        str(Path(td) / "ref.wav"),
                        "--codec-path",
                        str(Path(td) / "codec.npz"),
                        "--output-dir",
                        td,
                    ]
                )

            report_path = Path(td) / "dacvae-encode-parity.json"
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)

    def test_main_returns_partial_report_when_setup_or_encode_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_path = root / "ref.wav"
            codec_path = root / "missing-codec.npz"
            write_wav(audio_path)

            with mock.patch.object(
                check_dacvae_encode_parity,
                "encode_pair",
                side_effect=RuntimeError("real Semantic-DACVAE encoder conversion is blocked"),
            ):
                rc = check_dacvae_encode_parity.main(
                    [
                        "--audio-wav",
                        str(audio_path),
                        "--codec-path",
                        str(codec_path),
                        "--output-dir",
                        td,
                    ]
                )

            report = json.loads((root / "dacvae-encode-parity.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 2)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/155")
            self.assertEqual(report["parent_epic"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/160")
            self.assertEqual(report["blocker"]["stage"], "setup-or-encode")
            self.assertIn("blocked", report["blocker"]["message"])
            self.assertTrue(report["audio"]["stats_available"])
            self.assertFalse(report["codec"]["mlx_codec_path_exists"])
            self.assertIsNone(report["comparison"])


if __name__ == "__main__":
    unittest.main()
