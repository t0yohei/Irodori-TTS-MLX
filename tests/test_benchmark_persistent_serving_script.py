from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import scripts.benchmark_persistent_serving as bench


class PersistentServingBenchmarkScriptTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            case_label="serving test",
            text="hello",
            caption=None,
            seed=10,
            requests=2,
            warmup_requests=1,
            seconds=2.0,
            omit_seconds=False,
            num_steps=12,
            reference_wav=None,
            upstream_root="/tmp/upstream",
            mlx_python="python3",
            weights=None,
            weights_dir=None,
            weights_repo="owner/repo",
            weights_revision="abc",
            codec_device="cpu",
            codec_repo="codec-repo",
            codec_runtime_mode="mlx-decode",
            codec_path=None,
            codec_artifact_dir=None,
            codec_artifact_repo="owner/codec",
            codec_artifact_revision="def",
            cleanup_between_requests=False,
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            dry_run=False,
        )

    def test_build_request_overrides_uses_no_reference_short_prompt(self):
        args = self._args()
        with tempfile.TemporaryDirectory() as td:
            requests = bench.build_request_overrides(args, Path(td))
        self.assertEqual(len(requests), 3)
        self.assertEqual([item["seed"] for item in requests], [10, 11, 12])
        self.assertTrue(all(item["no_reference"] for item in requests))
        self.assertTrue(requests[0]["output"].endswith("serving-test.request-01.wav"))
        self.assertEqual(requests[1]["num_steps"], 12)

    def test_build_worker_command_forwards_runtime_inputs(self):
        args = self._args()
        argv, env = bench.build_worker_command(args, Path.cwd())
        self.assertEqual(argv[:2], ["python3", "scripts/benchmark_persistent_serving.py"])
        self.assertIn("--worker", argv)
        self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], "mlx-decode")
        self.assertEqual(argv[argv.index("--codec-artifact-repo") + 1], "owner/codec")
        self.assertIn(str(Path("/tmp/upstream").resolve()), env["PYTHONPATH"])

    def test_parse_args_defaults_to_runnable_pytorch_codec_mode(self):
        args = bench.parse_args(["--weights-repo", "owner/repo"])
        self.assertEqual(args.codec_runtime_mode, "persistent")

    def test_script_adds_repo_root_to_import_path_for_worker_mode(self):
        self.assertEqual(bench.ROOT, Path(bench.__file__).resolve().parents[1])
        self.assertIn(str(bench.ROOT), sys.path)

    def test_validate_worker_request_reuses_layout_constraints(self):
        class FakeGen:
            @staticmethod
            def validate_checkpoint_family_request(**kwargs):
                raise AssertionError("family validation should not run after layout rejection")

        args = argparse.Namespace(reference_wav=None, no_reference=False, caption=None)
        with self.assertRaises(SystemExit):
            bench.validate_worker_request(
                FakeGen,
                model_config=object(),
                gen_args=args,
                layout_runtime={"supports_no_reference": False},
                overrides={"no_reference": True},
                index=1,
            )

    def test_validate_worker_request_calls_family_validation(self):
        calls = []

        class FakeGen:
            @staticmethod
            def validate_checkpoint_family_request(**kwargs):
                calls.append(kwargs)

        args = argparse.Namespace(reference_wav=None, no_reference=True, caption=None)
        model_config = object()
        bench.validate_worker_request(
            FakeGen,
            model_config=model_config,
            gen_args=args,
            layout_runtime={"supports_no_reference": True},
            overrides={"no_reference": True},
            index=2,
        )
        self.assertEqual(calls[0]["model_config"], model_config)
        self.assertEqual(calls[0]["index"], 2)

    def test_parse_response_records_persistent_latency_and_timing_split(self):
        payload = {
            "result": {
                "output_wav": "/tmp/out.wav",
                "codec_decode_backend": "mlx",
                "timings_ms": {
                    "sample_rf": 80,
                    "decode_dacvae": 30,
                    "decode_dacvae_model": 25,
                    "audio_write": 5,
                    "total_to_decode": 120,
                },
            },
            "request": {"text": "hello", "seed": 1, "num_steps": 12},
            "worker": {"generate_latency_ms": 125, "json_serialization_ms": 0.2},
        }
        parsed = bench.parse_response(payload, index=2, phase="measured", latency_ms=130.0)
        self.assertEqual(parsed.persistent_request_latency_ms, 130.0)
        self.assertEqual(parsed.timings_ms["audio_write"], 5.0)
        self.assertEqual(parsed.codec_decode_backend, "mlx")

    def test_build_json_summary_and_report_include_contract_fields(self):
        args = self._args()
        requests = (
            bench.RequestResult(1, "warmup", "/tmp/1.wav", "one", 1, 12, 150, 145, 0.1, {"sample_rf": 90, "decode_dacvae": 40, "decode_dacvae_model": 35, "audio_write": 5, "total_to_decode": 140}, None, "mlx"),
            bench.RequestResult(2, "measured", "/tmp/2.wav", "two", 2, 12, 130, 126, 0.1, {"sample_rf": 80, "decode_dacvae": 30, "decode_dacvae_model": 25, "audio_write": 5, "total_to_decode": 120}, None, "mlx"),
            bench.RequestResult(3, "measured", "/tmp/3.wav", "three", 3, 12, 170, 166, 0.1, {"sample_rf": 100, "decode_dacvae": 45, "decode_dacvae_model": 38, "audio_write": 7, "total_to_decode": 160}, None, "mlx"),
        )
        result = bench.ServingRunResult("python ...", "/tmp/repo", 3, 1, 2, "/tmp/requests.json", "/tmp/stderr.log", "passed", 1000, 1234, requests)
        summary = bench.build_json_summary(result, args=args)
        self.assertEqual(summary["benchmark_kind"], "persistent-local-serving")
        self.assertEqual(summary["aggregates"]["measured_persistent_request_latency_ms"]["median"], 150.0)
        self.assertEqual(summary["aggregates"]["measured_audio_write_ms"]["median"], 6.0)
        report = bench.build_report(result, args=args)
        self.assertIn("Persistent Local Serving Benchmark Report", report)
        self.assertIn("persistent request latency", report)
        self.assertIn("audio_write", report)

    def test_self_test_path(self):
        self.assertEqual(bench.run_self_test(), 0)


if __name__ == "__main__":
    unittest.main()
