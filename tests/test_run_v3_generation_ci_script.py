from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import scripts.run_v3_generation_ci as run_v3_generation_ci


class RunV3GenerationCIScriptTests(unittest.TestCase):
    def test_run_generation_converts_checkpoint_and_runs_generate_wav_without_manual_seconds(self):
        with tempfile.TemporaryDirectory() as td:
            download_dir = Path(td) / "download"
            output_dir = Path(td) / "artifacts"
            checkpoint_path = str(download_dir / "model.safetensors")
            fake_records = {"tensor": object()}
            fake_validation = {
                "ok": True,
                "checkpoint_family": "v3",
                "supported_checkpoint": "Aratako/Irodori-TTS-500M-v3",
                "missing_keys": [],
                "unexpected_keys": [],
                "unsupported_keys": [],
                "shape_mismatches": [],
                "dtype_mismatches": [],
                "config_errors": [],
            }
            generation_payload = {
                "result": {
                    "output_wav": str(output_dir / "v3-hosted.wav"),
                    "samples": 1234,
                    "duration_mode": "predicted",
                },
                "request": {"seconds": None},
            }

            def fake_subprocess_run(command, **kwargs):
                self.assertIn("--no-ref", command)
                self.assertNotIn("--seconds", command)
                metadata_path = Path(command[command.index("--metadata-json") + 1])
                output_wav = Path(command[command.index("--output-wav") + 1])
                metadata_path.write_text(json.dumps(generation_payload), encoding="utf-8")
                output_wav.write_bytes(b"fake wav")
                return CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout="[codec] loading checkpoint\n" + json.dumps(generation_payload),
                    stderr="",
                )

            download_calls = []

            def fake_download(**kwargs):
                download_calls.append(kwargs)
                return checkpoint_path

            with patch.object(
                run_v3_generation_ci,
                "_require_hf_hub_download",
                return_value=fake_download,
            ), patch.object(
                run_v3_generation_ci,
                "inspect_local_safetensors",
                return_value=type("Inspection", (), {"tensors": [1, 2], "config": {"use_duration_predictor": True}, "source": {"type": "local"}})(),
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "load_checkpoint",
                return_value=({"use_duration_predictor": True, "use_caption_condition": False}, fake_records),
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "validate_records",
                return_value=fake_validation,
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "build_report",
                return_value={"checkpoint_family": "v3", "validation": fake_validation},
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "records_to_arrays",
                return_value={"tensor": [1, 2, 3]},
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "write_npz_atomic",
                side_effect=lambda path, arrays: Path(path).write_bytes(b"fake npz"),
            ), patch.object(
                run_v3_generation_ci.subprocess,
                "run",
                side_effect=fake_subprocess_run,
            ):
                result = run_v3_generation_ci.run_generation(
                    repo_id="repo",
                    filename="model.safetensors",
                    revision="main",
                    download_dir=str(download_dir),
                    output_dir=str(output_dir),
                    text="hello",
                    num_steps=4,
                    codec_device="cpu",
                    upstream_root=None,
                    python_executable="python3",
                )

                self.assertEqual(result["report"]["checkpoint_family"], "v3")
                self.assertTrue(Path(result["weights_path"]).exists())
                self.assertTrue(Path(result["model_config_path"]).exists())
                self.assertTrue(Path(result["output_wav"]).exists())
                self.assertEqual(result["generation"]["result"]["duration_mode"], "predicted")
                self.assertEqual(len(download_calls), 1)
                self.assertNotIn("local_dir_use_symlinks", download_calls[0])

    def test_run_generation_rejects_non_v3_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = str(Path(td) / "model.safetensors")
            fake_validation = {
                "ok": True,
                "checkpoint_family": "v3",
                "supported_checkpoint": "Aratako/Irodori-TTS-500M-v3",
                "missing_keys": [],
                "unexpected_keys": [],
                "unsupported_keys": [],
                "shape_mismatches": [],
                "dtype_mismatches": [],
                "config_errors": [],
            }
            with patch.object(
                run_v3_generation_ci,
                "_require_hf_hub_download",
                return_value=lambda **kwargs: checkpoint_path,
            ), patch.object(
                run_v3_generation_ci,
                "inspect_local_safetensors",
                return_value=type("Inspection", (), {"tensors": [1], "config": {}, "source": {"type": "local"}})(),
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "load_checkpoint",
                return_value=({"use_duration_predictor": False}, {"tensor": object()}),
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "validate_records",
                return_value=fake_validation,
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "build_report",
                return_value={"checkpoint_family": "v3", "validation": fake_validation},
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "records_to_arrays",
                return_value={"tensor": [1, 2, 3]},
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "write_npz_atomic",
                side_effect=lambda path, arrays: Path(path).write_bytes(b"fake npz"),
            ):
                with self.assertRaisesRegex(RuntimeError, "duration-predictor checkpoint config"):
                    run_v3_generation_ci.run_generation(
                        repo_id="repo",
                        filename="model.safetensors",
                        revision="main",
                        download_dir=td,
                        output_dir=td,
                        text="hello",
                        num_steps=4,
                        codec_device="cpu",
                        upstream_root=None,
                        python_executable="python3",
                    )

    def test_run_generation_rejects_non_predicted_duration_result(self):
        with tempfile.TemporaryDirectory() as td:
            download_dir = Path(td) / "download"
            output_dir = Path(td) / "artifacts"
            checkpoint_path = str(download_dir / "model.safetensors")
            fake_validation = {
                "ok": True,
                "checkpoint_family": "v3",
                "supported_checkpoint": "Aratako/Irodori-TTS-500M-v3",
                "missing_keys": [],
                "unexpected_keys": [],
                "unsupported_keys": [],
                "shape_mismatches": [],
                "dtype_mismatches": [],
                "config_errors": [],
            }
            generation_payload = {
                "result": {"output_wav": str(output_dir / "v3-hosted.wav"), "duration_mode": "manual"},
                "request": {"seconds": 2.0},
            }

            def fake_subprocess_run(command, **kwargs):
                metadata_path = Path(command[command.index("--metadata-json") + 1])
                output_wav = Path(command[command.index("--output-wav") + 1])
                metadata_path.write_text(json.dumps(generation_payload), encoding="utf-8")
                output_wav.write_bytes(b"fake wav")
                return CompletedProcess(args=command, returncode=0, stdout=json.dumps(generation_payload), stderr="")

            with patch.object(
                run_v3_generation_ci,
                "_require_hf_hub_download",
                return_value=lambda **kwargs: checkpoint_path,
            ), patch.object(
                run_v3_generation_ci,
                "inspect_local_safetensors",
                return_value=type("Inspection", (), {"tensors": [1], "config": {"use_duration_predictor": True}, "source": {"type": "local"}})(),
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "load_checkpoint",
                return_value=({"use_duration_predictor": True, "use_caption_condition": False}, {"tensor": object()}),
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "validate_records",
                return_value=fake_validation,
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "build_report",
                return_value={"checkpoint_family": "v3", "validation": fake_validation},
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "records_to_arrays",
                return_value={"tensor": [1, 2, 3]},
            ), patch.object(
                run_v3_generation_ci.convert_weights,
                "write_npz_atomic",
                side_effect=lambda path, arrays: Path(path).write_bytes(b"fake npz"),
            ), patch.object(
                run_v3_generation_ci.subprocess,
                "run",
                side_effect=fake_subprocess_run,
            ):
                with self.assertRaisesRegex(RuntimeError, "duration_mode='predicted'"):
                    run_v3_generation_ci.run_generation(
                        repo_id="repo",
                        filename="model.safetensors",
                        revision="main",
                        download_dir=str(download_dir),
                        output_dir=str(output_dir),
                        text="hello",
                        num_steps=4,
                        codec_device="cpu",
                        upstream_root=None,
                        python_executable="python3",
                    )


if __name__ == "__main__":
    unittest.main()
