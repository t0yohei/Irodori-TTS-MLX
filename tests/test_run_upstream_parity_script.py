from __future__ import annotations

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import scripts.run_upstream_parity as run_upstream_parity


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunUpstreamParityScriptTests(unittest.TestCase):
    def test_fixture_report_contains_contract_axes_and_expected_drift(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(["--fixture", "--scenario", "v3-no-reference", "--output-dir", td])
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["report_status"], "complete")
        self.assertEqual(report["scenario"]["checkpoint_family"], "v3")
        self.assertEqual(report["upstream"]["status"], "fixture")
        self.assertEqual(report["upstream"]["availability"]["state"], "fixture")
        self.assertEqual(report["mlx"]["status"], "fixture")
        self.assertEqual(report["comparison"]["status"], "expected_drift")
        self.assertEqual(report["metadata_axes"]["duration"]["expected_mode"], "predicted_or_upstream_default")
        self.assertIn("tokenizer", report["metadata_axes"])
        self.assertIn("sampling", report["metadata_axes"])
        self.assertIn("codec", report["metadata_axes"])

    def test_checked_in_schema_matches_fixture_report_contract(self):
        schema = json.loads((REPO_ROOT / "docs" / "upstream_parity_report_schema.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(["--fixture", "--scenario", "v3-no-reference", "--output-dir", td])
            report = run_upstream_parity.build_report(args)

        self.assertEqual(schema["properties"]["schema_version"]["const"], report["schema_version"])
        for key in schema["required"]:
            self.assertIn(key, report)
        for side in ("upstream", "mlx"):
            for key in schema["$defs"]["side"]["required"]:
                self.assertIn(key, report[side])

    def test_voicedesign_fixture_records_caption_and_manual_duration(self):
        args = run_upstream_parity.parse_args(
            [
                "--fixture",
                "--scenario",
                "voicedesign-no-reference",
                "--output-dir",
                "unused",
                "--caption-tokenizer-repo",
                "caption/repo",
                "--caption-max-length",
                "128",
            ]
        )

        report = run_upstream_parity.build_report(args)

        self.assertEqual(report["scenario"]["checkpoint_family"], "voicedesign")
        self.assertTrue(report["metadata_axes"]["tokenizer"]["caption_enabled"])
        self.assertEqual(report["metadata_axes"]["tokenizer"]["caption_tokenizer_repo"], "caption/repo")
        self.assertEqual(report["metadata_axes"]["duration"]["expected_mode"], "manual")
        self.assertIn("--caption", report["upstream"]["command"]["argv"])
        self.assertIn("--seconds", report["mlx"]["command"]["argv"])

    def test_real_mode_without_execution_writes_rerunnable_commands(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-no-reference",
                    "--output-dir",
                    td,
                    "--mlx-weights",
                    "/tmp/irodori-v3.npz",
                    "--mlx-model-config-json",
                    "/tmp/v3-model-config.json",
                    "--codec-device",
                    "cpu",
                ]
            )
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["upstream"]["status"], "not_run")
        self.assertEqual(report["mlx"]["status"], "not_run")
        self.assertEqual(report["report_status"], "partial")
        self.assertEqual(report["upstream"]["availability"]["reason"], "not_requested")
        self.assertEqual(report["mlx"]["availability"]["reason"], "not_requested")
        self.assertEqual(report["comparison"]["status"], "not_comparable")
        self.assertIn("infer.py", report["upstream"]["command"]["argv"])
        self.assertIn("generate_wav.py", " ".join(report["mlx"]["command"]["argv"]))
        self.assertIn("--model-config-json", report["mlx"]["command"]["argv"])

    def test_run_upstream_resolves_relative_output_dir_before_changing_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            self.addCleanup(os.chdir, old_cwd)
            os.chdir(td)
            upstream_root = Path(td) / "upstream"
            upstream_root.mkdir()

            def fake_run(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
                self.assertEqual(cwd, upstream_root.resolve())
                output_wav = Path(command[command.index("--output-wav") + 1])
                self.assertTrue(output_wav.is_absolute())
                self.assertTrue(output_wav.parent.is_relative_to(Path(td).resolve()))
                self.assertFalse(output_wav.parent.is_relative_to(upstream_root.resolve()))
                output_wav.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output_wav), "wb") as fh:
                    fh.setnchannels(1)
                    fh.setsampwidth(2)
                    fh.setframerate(24000)
                    fh.writeframes(b"\x00\x00" * 240)
                return {"status": "passed", "returncode": 0, "elapsed_seconds": 0.0, "stdout_excerpt": "", "stderr_excerpt": ""}

            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-no-reference",
                    "--output-dir",
                    "relative-parity",
                    "--run-upstream",
                    "--upstream-root",
                    str(upstream_root),
                ]
            )
            with mock.patch.object(run_upstream_parity, "_run", side_effect=fake_run):
                report = run_upstream_parity.build_report(args)

        upstream_wav = Path(report["upstream"]["command"]["argv"][report["upstream"]["command"]["argv"].index("--output-wav") + 1])
        self.assertTrue(upstream_wav.is_absolute())
        self.assertEqual(report["upstream"]["audio"]["path"], str(upstream_wav))
        self.assertEqual(report["upstream"]["audio"]["sample_rate"], 24000)
        self.assertEqual(report["upstream"]["availability"]["state"], "passed")
        self.assertEqual(report["report_status"], "partial")

    def test_run_mlx_resolves_relative_output_dir_before_running_from_repo_root(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            self.addCleanup(os.chdir, old_cwd)
            os.chdir(td)

            def fake_run(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, object]:
                self.assertEqual(cwd, run_upstream_parity.ROOT)
                output_wav = Path(command[command.index("--output") + 1])
                metadata_json = Path(command[command.index("--metadata-json") + 1])
                self.assertTrue(output_wav.is_absolute())
                self.assertTrue(metadata_json.is_absolute())
                self.assertTrue(output_wav.parent.is_relative_to(Path(td).resolve()))
                self.assertFalse(output_wav.parent.is_relative_to(run_upstream_parity.ROOT))
                output_wav.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output_wav), "wb") as fh:
                    fh.setnchannels(1)
                    fh.setsampwidth(2)
                    fh.setframerate(24000)
                    fh.writeframes(b"\x00\x00" * 240)
                metadata_json.write_text(json.dumps({"result": {"duration_mode": "predicted"}}), encoding="utf-8")
                return {"status": "passed", "returncode": 0, "elapsed_seconds": 0.0, "stdout_excerpt": "", "stderr_excerpt": ""}

            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-no-reference",
                    "--output-dir",
                    "relative-parity",
                    "--run-mlx",
                    "--mlx-weights",
                    "/tmp/irodori-v3.npz",
                ]
            )
            with mock.patch.object(run_upstream_parity, "_run", side_effect=fake_run):
                report = run_upstream_parity.build_report(args)

        mlx_wav = Path(report["mlx"]["command"]["argv"][report["mlx"]["command"]["argv"].index("--output") + 1])
        self.assertTrue(mlx_wav.is_absolute())
        self.assertEqual(report["mlx"]["audio"]["path"], str(mlx_wav))
        self.assertEqual(report["mlx"]["audio"]["sample_rate"], 24000)
        self.assertEqual(report["mlx"]["availability"]["state"], "passed")
        self.assertEqual(report["report_status"], "partial")

    def test_missing_requested_upstream_is_partial_unavailable_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-no-reference",
                    "--output-dir",
                    td,
                    "--run-upstream",
                    "--mlx-weights",
                    "/tmp/irodori-v3.npz",
                ]
            )
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["report_status"], "partial")
        self.assertEqual(report["upstream"]["status"], "unavailable")
        self.assertEqual(report["upstream"]["availability"]["reason"], "missing_upstream_root")
        self.assertEqual(report["comparison"]["status"], "not_comparable")

    def test_missing_requested_mlx_weights_is_partial_unavailable_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-no-reference",
                    "--output-dir",
                    td,
                    "--run-mlx",
                ]
            )
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["report_status"], "partial")
        self.assertEqual(report["mlx"]["status"], "unavailable")
        self.assertEqual(report["mlx"]["availability"]["reason"], "missing_mlx_weights")
        self.assertEqual(report["comparison"]["status"], "not_comparable")

    def test_unavailable_side_ignores_stale_audio_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            stale_wav = Path(td) / "v3-no-reference.mlx.wav"
            with wave.open(str(stale_wav), "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(24000)
                fh.writeframes(b"\x00\x00" * 240)

            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-no-reference",
                    "--output-dir",
                    td,
                    "--run-mlx",
                ]
            )
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["mlx"]["status"], "unavailable")
        self.assertIsNone(report["mlx"]["audio"])

    def test_default_report_path_expands_output_dir(self):
        with tempfile.TemporaryDirectory() as td:
            old_home = os.environ.get("HOME")
            self.addCleanup(lambda: os.environ.__setitem__("HOME", old_home) if old_home is not None else os.environ.pop("HOME", None))
            os.environ["HOME"] = td
            result = run_upstream_parity.main(["--fixture", "--scenario", "v3-no-reference", "--output-dir", "~/parity-runs"])

            report_path = Path(td) / "parity-runs" / "v3-no-reference.parity.json"
            literal_path = Path.cwd() / "~" / "parity-runs" / "v3-no-reference.parity.json"
            self.assertTrue(report_path.exists())
            self.assertFalse(literal_path.exists())

        self.assertEqual(result, 0)

    def test_scenario_json_overrides_core_fields(self):
        with tempfile.TemporaryDirectory() as td:
            scenario_path = Path(td) / "scenario.json"
            scenario_path.write_text(
                json.dumps({"text": "カスタムです。", "num_steps": 4, "seed": 11, "seconds": 1.25}),
                encoding="utf-8",
            )
            args = run_upstream_parity.parse_args(
                ["--fixture", "--scenario", "v3-no-reference", "--scenario-json", str(scenario_path), "--output-dir", td]
            )
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["scenario"]["text"], "カスタムです。")
        self.assertEqual(report["metadata_axes"]["sampling"]["num_steps"], 4)
        self.assertEqual(report["metadata_axes"]["sampling"]["seed"], 11)
        self.assertEqual(report["metadata_axes"]["duration"]["expected_mode"], "manual")


if __name__ == "__main__":
    unittest.main()
