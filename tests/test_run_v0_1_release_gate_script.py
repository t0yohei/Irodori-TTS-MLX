from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_v0_1_release_gate as run_v0_1_release_gate


def _fake_generation_result(output_dir: Path, *, family: str, duration_mode: str | None = "predicted"):
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{family}.npz"
    output_wav = output_dir / f"{family}.wav"
    metadata_json = output_dir / f"{family}-metadata.json"
    stdout_path = output_dir / f"{family}.stdout.json"
    stderr_path = output_dir / f"{family}.stderr.txt"
    weights_path.write_bytes(b"fake weights")
    output_wav.write_bytes(b"fake wav")
    generation = {
        "request": {"seconds": None, "caption": "calm" if family == "voicedesign" else None},
        "result": {"samples": 1234},
    }
    if duration_mode is not None:
        generation["result"]["duration_mode"] = duration_mode
    metadata_json.write_text(json.dumps(generation), encoding="utf-8")
    stdout_path.write_text(json.dumps(generation), encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return {
        "repo_id": f"repo/{family}",
        "revision": "main",
        "checkpoint_path": str(output_dir / "model.safetensors"),
        "inspection": {"tensor_count": 2, "has_config": True, "source": {"type": "local"}},
        "report": {"checkpoint_family": family, "validation": {"ok": True}},
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "model_config_path": str(output_dir / f"{family}-config.json"),
        "output_wav": str(output_wav),
        "output_wav_bytes": output_wav.stat().st_size,
        "metadata_json": str(metadata_json),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": {"argv": ["python"], "shell": "python"},
        "generation": generation,
    }


class RunV01ReleaseGateScriptTests(unittest.TestCase):
    def test_release_gate_runs_required_v3_and_skips_optional_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "artifacts"
            calls = []

            def fake_v3(**kwargs):
                calls.append(kwargs)
                return _fake_generation_result(Path(kwargs["output_dir"]), family="v3")

            with patch.object(run_v0_1_release_gate.run_v3_generation_ci, "run_generation", side_effect=fake_v3), patch.object(
                run_v0_1_release_gate.run_voicedesign_generation_ci, "run_generation"
            ) as voicedesign:
                summary = run_v0_1_release_gate.run_release_gate(
                    output_dir=str(output_dir),
                    download_dir=None,
                    upstream_root="/tmp/Irodori-TTS",
                    codec_device="cpu",
                    num_steps=4,
                    include_optional_voicedesign=False,
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["required_check"], "required_v3")
            self.assertEqual(summary["checks"]["required_v3"]["status"], "pass")
            self.assertEqual(summary["checks"]["optional_voicedesign"]["status"], "skipped")
            self.assertTrue(Path(summary["summary_json"]).exists())
            self.assertEqual(calls[0]["upstream_root"], "/tmp/Irodori-TTS")
            self.assertEqual(calls[0]["num_steps"], 4)
            voicedesign.assert_not_called()

    def test_release_gate_keeps_default_output_artifacts_after_return(self):
        def fake_v3(**kwargs):
            return _fake_generation_result(Path(kwargs["output_dir"]), family="v3")

        with patch.object(run_v0_1_release_gate.run_v3_generation_ci, "run_generation", side_effect=fake_v3):
            summary = run_v0_1_release_gate.run_release_gate(
                output_dir=None,
                download_dir=None,
                upstream_root=None,
                codec_device="cpu",
                num_steps=4,
                include_optional_voicedesign=False,
            )

        self.assertTrue(Path(summary["summary_json"]).exists())
        self.assertTrue(Path(summary["checks"]["required_v3"]["artifacts"]["output_wav"]).exists())

    def test_release_gate_can_include_optional_voicedesign(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "artifacts"

            def fake_v3(**kwargs):
                return _fake_generation_result(Path(kwargs["output_dir"]), family="v3")

            def fake_voicedesign(**kwargs):
                return _fake_generation_result(Path(kwargs["output_dir"]), family="voicedesign", duration_mode=None)

            with patch.object(run_v0_1_release_gate.run_v3_generation_ci, "run_generation", side_effect=fake_v3), patch.object(
                run_v0_1_release_gate.run_voicedesign_generation_ci, "run_generation", side_effect=fake_voicedesign
            ):
                summary = run_v0_1_release_gate.run_release_gate(
                    output_dir=str(output_dir),
                    download_dir=None,
                    upstream_root=None,
                    codec_device="cpu",
                    num_steps=4,
                    include_optional_voicedesign=True,
                )

            self.assertEqual(summary["checks"]["optional_voicedesign"]["status"], "pass")
            self.assertEqual(summary["checks"]["optional_voicedesign"]["artifacts"]["checkpoint_family"], "voicedesign")

    def test_release_gate_rejects_v3_without_predicted_duration_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            def fake_v3(**kwargs):
                return _fake_generation_result(Path(kwargs["output_dir"]), family="v3", duration_mode="manual")

            with patch.object(run_v0_1_release_gate.run_v3_generation_ci, "run_generation", side_effect=fake_v3):
                with self.assertRaisesRegex(RuntimeError, "duration_mode='predicted'"):
                    run_v0_1_release_gate.run_release_gate(
                        output_dir=str(Path(td) / "artifacts"),
                        download_dir=None,
                        upstream_root=None,
                        codec_device="cpu",
                        num_steps=4,
                        include_optional_voicedesign=False,
                    )

    def test_release_gate_rejects_missing_required_metadata_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            def fake_v3(**kwargs):
                result = _fake_generation_result(Path(kwargs["output_dir"]), family="v3")
                Path(result["metadata_json"]).unlink()
                return result

            with patch.object(run_v0_1_release_gate.run_v3_generation_ci, "run_generation", side_effect=fake_v3):
                with self.assertRaisesRegex(RuntimeError, "metadata artifact"):
                    run_v0_1_release_gate.run_release_gate(
                        output_dir=str(Path(td) / "artifacts"),
                        download_dir=None,
                        upstream_root=None,
                        codec_device="cpu",
                        num_steps=4,
                        include_optional_voicedesign=False,
                    )


if __name__ == "__main__":
    unittest.main()
