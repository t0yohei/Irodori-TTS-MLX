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
    def test_encode_pair_uses_audio_for_mlx_and_writes_latents(self):
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
                    "--expected-latent-dim",
                    "2",
                ]
            )

            mlx = FakeEncodeBridge(offset=0.0)
            fake_audio = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
            with mock.patch.object(
                check_dacvae_encode_parity, "DACVAEBridgeConfig", side_effect=lambda **kwargs: kwargs
            ), mock.patch.object(
                check_dacvae_encode_parity, "_load_runtime_encode_dependencies"
            ), mock.patch.object(
                check_dacvae_encode_parity, "_load_audio_numpy", return_value=(fake_audio, 8000)
            ), mock.patch.object(
                check_dacvae_encode_parity, "MLXDACVAEBridge", return_value=mlx
            ) as mlx_factory:
                report = check_dacvae_encode_parity.encode_pair(args)

        mlx_factory.assert_called_once()
        self.assertEqual(mlx.calls, [(str(audio_path), 0.25, -16.0, True)])
        self.assertEqual(report["run"]["status"], "complete")
        self.assertEqual(report["schema_version"], check_dacvae_encode_parity.SCHEMA_VERSION)
        self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/185")
        self.assertEqual(report["parent_epic"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/169")
        self.assertEqual(report["comparison"]["status"], "passed")
        self.assertEqual(report["comparison"]["metrics"]["mlx"]["shape"], [1, 1, 2])
        self.assertTrue(report["codec"]["metadata_checks"]["sample_rate"])
        self.assertEqual(report["codec"]["expected_latent_dim"], 2)
        self.assertEqual(report["outputs"]["mlx_latents_npy"].split("/")[-1], "mlx-encode-latents.npy")

    def test_main_returns_nonzero_for_metric_failure_and_persists_json(self):
        with tempfile.TemporaryDirectory() as td:
            report = {
                "run": {"status": "complete"},
                "comparison": {"status": "failed"},
            }
            with mock.patch.object(check_dacvae_encode_parity, "_preflight_encode_pair"), mock.patch.object(
                check_dacvae_encode_parity, "encode_pair", return_value=report
            ):
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

    def test_main_writes_partial_report_for_missing_codec_when_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_path = root / "ref.wav"
            codec_path = root / "missing-codec.npz"
            write_wav(audio_path)

            rc = check_dacvae_encode_parity.main(
                [
                    "--audio-wav",
                    str(audio_path),
                    "--codec-path",
                    str(codec_path),
                    "--output-dir",
                    td,
                    "--allow-partial",
                ]
            )

            report = json.loads((root / "dacvae-encode-parity.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(report["run"]["status"], "partial")
            self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/185")
            self.assertEqual(report["parent_epic"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/169")
            self.assertIn("codec artifact", report["run"]["reason"])
            self.assertTrue(report["audio"]["stats_available"])
            self.assertFalse(report["codec"]["mlx_codec"]["exists"])
            self.assertEqual(report["comparison"]["status"], "partial")

    def test_main_writes_partial_report_for_missing_mlx_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_path = root / "ref.wav"
            codec_path = root / "codec.npz"
            write_wav(audio_path)
            codec_path.write_bytes(b"fake codec")

            original_find_spec = check_dacvae_encode_parity.importlib.util.find_spec

            def fake_find_spec(module_name):
                if module_name == "mlx":
                    return None
                return original_find_spec(module_name)

            with mock.patch.object(
                check_dacvae_encode_parity.importlib.util, "find_spec", side_effect=fake_find_spec
            ), mock.patch.object(
                check_dacvae_encode_parity,
                "_load_runtime_encode_dependencies",
                side_effect=AssertionError("runtime import should be deferred until after preflight"),
            ):
                rc = check_dacvae_encode_parity.main(
                    [
                        "--audio-wav",
                        str(audio_path),
                        "--codec-path",
                        str(codec_path),
                        "--output-dir",
                        td,
                        "--allow-partial",
                    ]
                )

            report = json.loads((root / "dacvae-encode-parity.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(report["run"]["status"], "partial")
            self.assertEqual(report["comparison"]["status"], "partial")
            self.assertIn("MLX runtime dependency", report["run"]["reason"])

    def test_main_does_not_treat_runtime_encode_errors_as_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_path = root / "ref.wav"
            codec_path = root / "codec.npz"
            write_wav(audio_path)
            codec_path.write_bytes(b"fake codec")

            with mock.patch.object(check_dacvae_encode_parity, "_preflight_encode_pair"), mock.patch.object(
                check_dacvae_encode_parity,
                "encode_pair",
                side_effect=RuntimeError("semantic encoder tensor shape drift"),
            ):
                rc = check_dacvae_encode_parity.main(
                    [
                        "--audio-wav",
                        str(audio_path),
                        "--codec-path",
                        str(codec_path),
                        "--output-dir",
                        td,
                        "--allow-partial",
                    ]
                )

            report = json.loads((root / "dacvae-encode-parity.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertEqual(report["run"]["status"], "failed")
            self.assertEqual(report["comparison"]["status"], "failed")
            self.assertIn("shape drift", report["comparison"]["reason"])


if __name__ == "__main__":
    unittest.main()
