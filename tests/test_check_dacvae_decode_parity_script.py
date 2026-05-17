from __future__ import annotations

import json
import os
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


def require_real_decode_parity_env(test_func):
    required = (
        "IRODORI_MLX_DACVAE_CODEC_NPZ",
        "IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    return unittest.skipIf(
        missing,
        "real DACVAE decode parity artifacts not set: " + ", ".join(missing),
    )(test_func)


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
                    "--expected-latent-dim",
                    "2",
                ]
            )

            def fake_audio(path):
                return np.load(Path(path).with_suffix(Path(path).suffix + ".npy")), 8000

            with mock.patch.object(
                check_dacvae_decode_parity, "DACVAEBridgeConfig", side_effect=lambda **kwargs: kwargs
            ), mock.patch.object(
                check_dacvae_decode_parity, "_load_runtime_decode_dependencies"
            ), mock.patch.object(
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
        self.assertEqual(report["schema_version"], check_dacvae_decode_parity.SCHEMA_VERSION)
        self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/184")
        self.assertEqual(report["parent_epic"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/169")
        self.assertEqual(report["latents"]["shape"], [1, 2, 2])
        self.assertEqual(report["codec"]["expected_latent_dim"], 2)
        self.assertTrue(report["codec"]["metadata_checks"]["sample_rate"])
        self.assertEqual(report["comparison"]["metrics"]["compared_samples"], 4)
        self.assertEqual(report["outputs"]["upstream_wav"].split("/")[-1], "upstream-decode.wav")
        self.assertEqual(report["outputs"]["mlx_wav"].split("/")[-1], "mlx-decode.wav")

    def test_decode_pair_rejects_latent_channel_drift_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latents_path = root / "latents.npy"
            codec_path = root / "codec.npz"
            np.save(latents_path, np.zeros((1, 2, 3), dtype=np.float32))
            codec_path.write_bytes(b"fake codec")
            args = check_dacvae_decode_parity.parse_args(
                [
                    "--latents-npy",
                    str(latents_path),
                    "--codec-path",
                    str(codec_path),
                    "--output-dir",
                    str(root / "out"),
                    "--expected-latent-dim",
                    "2",
                ]
            )

            with mock.patch.object(check_dacvae_decode_parity, "_load_runtime_decode_dependencies"):
                with self.assertRaisesRegex(ValueError, r"\(1,T,2\)"):
                    check_dacvae_decode_parity.decode_pair(args)

    def test_decode_pair_fails_fast_on_bridge_metadata_mismatch(self):
        class MismatchedBridge(FakeDecodeBridge):
            sample_rate = 16000

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
                    "--expected-latent-dim",
                    "2",
                ]
            )

            with mock.patch.object(
                check_dacvae_decode_parity, "DACVAEBridgeConfig", side_effect=lambda **kwargs: kwargs
            ), mock.patch.object(
                check_dacvae_decode_parity, "_load_runtime_decode_dependencies"
            ), mock.patch.object(
                check_dacvae_decode_parity, "PyTorchDACVAEBridge", return_value=FakeDecodeBridge(offset=0.0)
            ), mock.patch.object(
                check_dacvae_decode_parity, "MLXDACVAEBridge", return_value=MismatchedBridge(offset=0.0)
            ):
                with self.assertRaisesRegex(ValueError, "sample_rate mismatch"):
                    check_dacvae_decode_parity.decode_pair(args)

    def test_main_returns_nonzero_for_metric_failure_and_persists_json(self):
        with tempfile.TemporaryDirectory() as td:
            report = {
                "comparison": {"status": "failed"},
            }
            with mock.patch.object(check_dacvae_decode_parity, "_preflight_decode_pair"), mock.patch.object(
                check_dacvae_decode_parity, "decode_pair", return_value=report
            ):
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

    def test_main_does_not_treat_runtime_file_errors_as_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latents_path = root / "latents.npy"
            codec_path = root / "codec.npz"
            np.save(latents_path, np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32))
            codec_path.write_bytes(b"fake codec")

            with mock.patch.object(check_dacvae_decode_parity, "_preflight_decode_pair"), mock.patch.object(
                check_dacvae_decode_parity,
                "decode_pair",
                side_effect=FileNotFoundError("internal decode artifact missing"),
            ):
                rc = check_dacvae_decode_parity.main(
                    [
                        "--latents-npy",
                        str(latents_path),
                        "--codec-path",
                        str(codec_path),
                        "--output-dir",
                        td,
                        "--allow-partial",
                    ]
                )

            report_path = root / "dacvae-decode-parity.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertEqual(report["comparison"]["status"], "failed")
            self.assertEqual(report["run"]["status"], "failed")

    def test_main_writes_partial_report_for_missing_codec_when_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latents_path = root / "latents.npy"
            missing_codec_path = root / "missing-codec.npz"
            np.save(latents_path, np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32))

            rc = check_dacvae_decode_parity.main(
                [
                    "--latents-npy",
                    str(latents_path),
                    "--codec-path",
                    str(missing_codec_path),
                    "--output-dir",
                    td,
                    "--allow-partial",
                ]
            )

            report_path = root / "dacvae-decode-parity.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(report["comparison"]["status"], "partial")
            self.assertEqual(report["run"]["status"], "partial")
            self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/184")
            self.assertEqual(report["parent_epic"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/169")
            self.assertFalse(report["run"]["complete"])
            self.assertTrue(report["latents"]["exists"])
            self.assertFalse(report["codec"]["mlx_codec"]["exists"])

    def test_main_writes_partial_report_for_missing_mlx_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latents_path = root / "latents.npy"
            codec_path = root / "codec.npz"
            np.save(latents_path, np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32))
            codec_path.write_bytes(b"fake codec")
            original_find_spec = check_dacvae_decode_parity.importlib.util.find_spec

            def fake_find_spec(module_name):
                if module_name == "mlx":
                    return None
                return original_find_spec(module_name)

            with mock.patch.object(
                check_dacvae_decode_parity.importlib.util, "find_spec", side_effect=fake_find_spec
            ), mock.patch.object(
                check_dacvae_decode_parity,
                "_load_runtime_decode_dependencies",
                side_effect=AssertionError("runtime import should be deferred until after preflight"),
            ):
                rc = check_dacvae_decode_parity.main(
                    [
                        "--latents-npy",
                        str(latents_path),
                        "--codec-path",
                        str(codec_path),
                        "--output-dir",
                        td,
                        "--allow-partial",
                    ]
                )

            report = json.loads((root / "dacvae-decode-parity.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(report["comparison"]["status"], "partial")
            self.assertIn("MLX runtime dependency", report["run"]["reason"])

    def test_main_writes_partial_report_for_missing_torch_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latents_path = root / "latents.npy"
            codec_path = root / "codec.npz"
            np.save(latents_path, np.array([[[0.1, -0.2], [0.3, -0.4]]], dtype=np.float32))
            codec_path.write_bytes(b"fake codec")
            original_find_spec = check_dacvae_decode_parity.importlib.util.find_spec

            def fake_find_spec(module_name):
                if module_name == "irodori_tts.codec":
                    return object()
                if module_name == "torch":
                    return None
                return original_find_spec(module_name)

            with mock.patch.object(
                check_dacvae_decode_parity.importlib.util, "find_spec", side_effect=fake_find_spec
            ), mock.patch.object(
                check_dacvae_decode_parity,
                "_load_runtime_decode_dependencies",
                side_effect=AssertionError("runtime import should be deferred until after preflight"),
            ):
                rc = check_dacvae_decode_parity.main(
                    [
                        "--latents-npy",
                        str(latents_path),
                        "--codec-path",
                        str(codec_path),
                        "--output-dir",
                        td,
                        "--allow-partial",
                    ]
                )

            report = json.loads((root / "dacvae-decode-parity.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(report["comparison"]["status"], "partial")
            self.assertIn("PyTorch runtime dependency", report["run"]["reason"])

    @require_real_decode_parity_env
    def test_real_decode_parity_command_runs_when_artifact_env_is_set(self):
        with tempfile.TemporaryDirectory() as td:
            rc = check_dacvae_decode_parity.main(
                [
                    "--latents-npy",
                    os.environ["IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY"],
                    "--codec-path",
                    os.environ["IRODORI_MLX_DACVAE_CODEC_NPZ"],
                    "--output-dir",
                    td,
                ]
            )

            report_path = Path(td) / "dacvae-decode-parity.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn(rc, (0, 1))
            self.assertEqual(report["run"]["status"], "complete")
            self.assertEqual(report["source_issue"], "https://github.com/t0yohei/Irodori-TTS-MLX/issues/184")
            self.assertIn("sample_rate", report["comparison"])
            self.assertIn("max_abs", report["comparison"]["metrics"])


if __name__ == "__main__":
    unittest.main()
