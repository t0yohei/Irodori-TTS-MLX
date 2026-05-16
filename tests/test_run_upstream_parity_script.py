from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import scripts.run_upstream_parity as run_upstream_parity


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_schema_subset(testcase: unittest.TestCase, schema: dict[str, object], value: object, path: str = "$") -> None:
    if "const" in schema:
        testcase.assertEqual(value, schema["const"], path)
    if "enum" in schema:
        testcase.assertIn(value, schema["enum"], path)
    if "$ref" in schema:
        ref = str(schema["$ref"])
        testcase.assertTrue(ref.startswith("#/$defs/"), ref)
        name = ref.removeprefix("#/$defs/")
        _assert_schema_subset(testcase, _REPORT_SCHEMA["$defs"][name], value, path)
        return
    if "type" in schema:
        expected = schema["type"]
        expected_types = expected if isinstance(expected, list) else [expected]
        if value is None:
            testcase.assertIn("null", expected_types, path)
        elif "object" in expected_types and isinstance(value, dict):
            pass
        elif "array" in expected_types and isinstance(value, list):
            pass
        elif "string" in expected_types and isinstance(value, str):
            pass
        else:
            testcase.fail(f"{path} has unsupported schema type {expected!r} for value {value!r}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            testcase.assertIn(key, value, f"{path}.{key}")
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                _assert_schema_subset(testcase, child_schema, value[key], f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _assert_schema_subset(testcase, schema["items"], item, f"{path}[{index}]")


_REPORT_SCHEMA = json.loads((REPO_ROOT / "docs" / "upstream_parity_report_schema.json").read_text(encoding="utf-8"))


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
        self.assertIn("metrics", report["upstream"]["audio"])
        self.assertIn("rms", report["upstream"]["audio"]["metrics"])
        self.assertIn("tail_rms", report["mlx"]["audio"]["metrics"])
        self.assertEqual(report["upstream"]["intermediates"]["duration"]["latent_steps"], 75)
        self.assertTrue(report["comparison"]["intermediate_comparisons"]["sampling.latent_shape"]["match"])
        self.assertIn("rms_ratio", report["comparison"]["audio_metric_deltas"])

    def test_checked_in_schema_matches_fixture_report_contract(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(["--fixture", "--scenario", "v3-no-reference", "--output-dir", td])
            report = run_upstream_parity.build_report(args)

        _assert_schema_subset(self, _REPORT_SCHEMA, report)

    def test_voicedesign_fixture_records_caption_and_manual_duration(self):
        args = run_upstream_parity.parse_args(
            [
                "--fixture",
                "--scenario",
                "voicedesign-contrastive-caption",
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
        self.assertEqual(report["scenario"]["name"], "voicedesign-contrastive-caption")
        self.assertEqual(report["scenario"]["seed"], 20260518)
        self.assertEqual(report["scenario"]["num_steps"], 12)
        self.assertEqual(report["scenario"]["seconds"], 2.0)
        self.assertIn("低めの落ち着いた男性の声", report["scenario"]["caption"])
        self.assertTrue(report["metadata_axes"]["tokenizer"]["caption_enabled"])
        self.assertEqual(report["metadata_axes"]["tokenizer"]["caption_tokenizer_repo"], "caption/repo")
        self.assertEqual(report["metadata_axes"]["duration"]["expected_mode"], "manual")
        self.assertEqual(report["metadata_axes"]["sampling"]["cfg_scale_caption"], 3.0)
        self.assertEqual(report["upstream"]["audio"]["sample_rate"], 24000)
        self.assertEqual(report["mlx"]["audio"]["duration_seconds"], 2.0)
        self.assertIn("--caption", report["upstream"]["command"]["argv"])
        self.assertIn("--seconds", report["mlx"]["command"]["argv"])
        _assert_schema_subset(self, _REPORT_SCHEMA, report)

    def test_v3_reference_predicted_fixture_is_discoverable_without_artifacts(self):
        args = run_upstream_parity.parse_args(
            ["--fixture", "--scenario", "v3-reference-predicted", "--output-dir", "unused"]
        )

        report = run_upstream_parity.build_report(args)

        self.assertEqual(report["scenario"]["checkpoint_family"], "v3")
        self.assertEqual(report["scenario"]["name"], "v3-reference-predicted")
        self.assertFalse(report["scenario"]["no_reference"])
        self.assertEqual(report["scenario"]["reference_wav"], "tests/fixtures/v3-reference.wav")
        self.assertIsNone(report["scenario"]["seconds"])
        self.assertEqual(report["metadata_axes"]["duration"]["expected_mode"], "predicted_or_upstream_default")
        self.assertEqual(report["metadata_axes"]["codec"]["reference_wav"], "tests/fixtures/v3-reference.wav")
        self.assertEqual(report["metadata_axes"]["sampling"]["seed"], 20260519)
        self.assertIn("--ref-wav", report["upstream"]["command"]["argv"])
        self.assertIn("--reference-wav", report["mlx"]["command"]["argv"])
        self.assertNotIn("--seconds", report["mlx"]["command"]["argv"])
        self.assertEqual(report["mlx"]["metadata"]["result"]["duration_mode"], "predicted")
        self.assertIsNone(report["mlx"]["metadata"]["result"]["requested_seconds"])
        self.assertEqual(report["mlx"]["metadata"]["result"]["predicted_duration"]["source"], "fixture")
        self.assertIn("predicted duration active", "\n".join(report["mlx"]["metadata"]["result"]["messages"]))
        self.assertEqual(report["mlx"]["audio"]["sample_rate"], 24000)
        self.assertEqual(report["comparison"]["status"], "expected_drift")
        _assert_schema_subset(self, _REPORT_SCHEMA, report)

    def test_v3_reference_real_run_without_fixture_wav_is_clear_partial_report(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "v3-reference-predicted",
                    "--run-upstream",
                    "--run-mlx",
                    "--output-dir",
                    td,
                    "--mlx-weights",
                    "/tmp/irodori-v3.npz",
                ]
            )

            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["report_status"], "partial")
        self.assertEqual(report["upstream"]["status"], "unavailable")
        self.assertEqual(report["upstream"]["availability"]["reason"], "missing_reference_wav")
        self.assertEqual(report["mlx"]["status"], "unavailable")
        self.assertEqual(report["mlx"]["availability"]["reason"], "missing_reference_wav")
        self.assertEqual(report["comparison"]["status"], "not_comparable")
        _assert_schema_subset(self, _REPORT_SCHEMA, report)

    def test_voicedesign_requested_real_sides_emit_clear_partial_report_without_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            args = run_upstream_parity.parse_args(
                [
                    "--scenario",
                    "voicedesign-contrastive-caption",
                    "--run-upstream",
                    "--run-mlx",
                    "--output-dir",
                    td,
                ]
            )
            report = run_upstream_parity.build_report(args)

        self.assertEqual(report["report_status"], "partial")
        self.assertEqual(report["upstream"]["status"], "unavailable")
        self.assertEqual(report["upstream"]["availability"]["reason"], "missing_upstream_root")
        self.assertEqual(report["mlx"]["status"], "unavailable")
        self.assertEqual(report["mlx"]["availability"]["reason"], "missing_mlx_weights")
        self.assertEqual(report["comparison"]["status"], "not_comparable")
        self.assertEqual(report["metadata_axes"]["duration"]["expected_mode"], "manual")
        self.assertEqual(report["scenario"]["cfg_scale_caption"], 3.0)
        _assert_schema_subset(self, _REPORT_SCHEMA, report)

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
                    fh.writeframes(b"\x00\x00\x00@\x00\xc0\x00\x00" * 60)
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
        self.assertAlmostEqual(report["upstream"]["audio"]["metrics"]["peak_abs"], 0.5)
        self.assertGreater(report["upstream"]["audio"]["metrics"]["rms"], 0.0)
        self.assertGreater(report["upstream"]["audio"]["metrics"]["zero_crossing_rate"], 0.0)
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
                metadata_json.write_text(
                    json.dumps(
                        {
                            "result": {
                                "duration_mode": "predicted",
                                "resolved_seconds": 1.0,
                                "latent_steps": 24,
                                "patched_steps": 24,
                                "seed": 20260516,
                            },
                            "request": {"text_max_length": 256, "caption_max_length": None, "caption": None},
                            "boundaries": {"config": {"model_config": {"latent_dim": 32}}},
                        }
                    ),
                    encoding="utf-8",
                )
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
        self.assertEqual(report["mlx"]["intermediates"]["duration"]["latent_steps"], 24)
        self.assertEqual(report["mlx"]["intermediates"]["sampling"]["latent_shape"], [1, 24, 32])
        self.assertEqual(report["mlx"]["availability"]["state"], "passed")
        self.assertEqual(report["report_status"], "partial")

    def test_wav_properties_computes_metrics_for_ieee_float_wav(self):
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "float.wav"
            samples = struct.pack("<ffff", 0.0, 0.5, -0.5, 0.0)
            fmt = struct.pack("<HHIIHH", 3, 1, 24000, 24000 * 4, 4, 32)
            with wav_path.open("wb") as fh:
                fh.write(b"RIFF")
                fh.write(struct.pack("<I", 4 + (8 + len(fmt)) + (8 + len(samples))))
                fh.write(b"WAVE")
                fh.write(b"fmt ")
                fh.write(struct.pack("<I", len(fmt)))
                fh.write(fmt)
                fh.write(b"data")
                fh.write(struct.pack("<I", len(samples)))
                fh.write(samples)

            props = run_upstream_parity.wav_properties(wav_path)

        self.assertIsNotNone(props)
        assert props is not None
        self.assertTrue(props["readable"])
        self.assertEqual(props["format"], "ieee_float")
        self.assertEqual(props["metrics_status"], "computed")
        self.assertEqual(props["sample_width_bytes"], 4)
        self.assertAlmostEqual(props["metrics"]["peak_abs"], 0.5)
        self.assertAlmostEqual(props["metrics"]["rms"], 0.3535533905932738)

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
