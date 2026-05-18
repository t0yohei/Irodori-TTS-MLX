from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import scripts.benchmark_persistent_batch as bench


class PersistentBatchBenchmarkScriptTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            case_label="batch test",
            text="hello",
            caption=None,
            seed=10,
            requests=2,
            warmup_requests=1,
            seconds=2.0,
            omit_seconds=False,
            num_steps=12,
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=None,
            cfg_scale_speaker=None,
            reference_wav=None,
            upstream_root="/tmp/upstream",
            mlx_python="python3",
            weights=None,
            weights_dir=None,
            weights_repo="owner/repo",
            weights_revision="abc",
            codec_device="cpu",
            codec_repo="codec-repo",
            codec_runtime_mode="mlx",
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

    def test_build_request_overrides_splits_warmup_and_measured_outputs(self):
        args = self._args()
        with tempfile.TemporaryDirectory() as td:
            requests = bench.build_request_overrides(args, Path(td))
        self.assertEqual(len(requests), 3)
        self.assertEqual([item["seed"] for item in requests], [10, 11, 12])
        self.assertTrue(all(item["no_reference"] for item in requests))
        self.assertTrue(all(item["cfg_guidance_mode"] == "independent" for item in requests))
        self.assertTrue(all(item["cfg_scale_text"] == 3.0 for item in requests))
        self.assertTrue(all(item["cfg_scale_caption"] == 0.0 for item in requests))
        self.assertTrue(all(item["cfg_scale_speaker"] == 0.0 for item in requests))
        self.assertTrue(requests[0]["output"].endswith("batch-test.request-01.wav"))
        self.assertEqual(requests[1]["seconds"], 2.0)

    def test_build_request_overrides_forwards_explicit_companion_cfg_scales(self):
        args = self._args()
        args.reference_wav = "/tmp/ref.wav"
        args.caption = "calm"
        args.cfg_scale_caption = 1.0
        args.cfg_scale_speaker = 1.0
        with tempfile.TemporaryDirectory() as td:
            requests = bench.build_request_overrides(args, Path(td))

        self.assertEqual(requests[0]["reference_wav"], "/tmp/ref.wav")
        self.assertEqual(requests[0]["caption"], "calm")
        self.assertEqual(requests[0]["cfg_scale_caption"], 1.0)
        self.assertEqual(requests[0]["cfg_scale_speaker"], 1.0)

    def test_build_report_shows_effective_default_companion_cfg_scales(self):
        args = self._args()
        args.caption = "calm"
        args.reference_wav = "/tmp/ref.wav"
        result = bench.BatchRunResult(
            command="python ...",
            cwd="/tmp/repo",
            request_count=0,
            warmup_requests=0,
            measured_requests=0,
            metadata_json="/tmp/metadata.json",
            requests_json="/tmp/requests.json",
            stdout_log="/tmp/stdout.log",
            stderr_log="/tmp/stderr.log",
            status="dry-run",
            wall_seconds=None,
            max_rss_bytes=None,
            process_setup_overhead_ms=None,
            requests=(),
        )

        report = bench.build_report(result, args=args)

        self.assertIn("CFG caption scale: 3.0", report)
        self.assertIn("CFG speaker scale: 5.0", report)

    def test_build_json_summary_persists_effective_default_companion_cfg_scales(self):
        args = self._args()
        args.caption = "calm"
        args.reference_wav = "/tmp/ref.wav"
        result = bench.BatchRunResult(
            command="python ...",
            cwd="/tmp/repo",
            request_count=0,
            warmup_requests=0,
            measured_requests=0,
            metadata_json="/tmp/metadata.json",
            requests_json="/tmp/requests.json",
            stdout_log="/tmp/stdout.log",
            stderr_log="/tmp/stderr.log",
            status="dry-run",
            wall_seconds=None,
            max_rss_bytes=None,
            process_setup_overhead_ms=None,
            requests=(),
        )

        summary = bench.build_json_summary(result, args=args)

        self.assertEqual(summary["invocation"]["cfg_scale_caption"], 3.0)
        self.assertEqual(summary["invocation"]["cfg_scale_speaker"], 5.0)

    def test_build_command_forwards_hosted_codec_artifact(self):
        args = self._args()
        argv, env = bench.build_command(args, Path("/tmp/requests.json"), Path("/tmp/metadata.json"))
        self.assertIn("--requests-json", argv)
        self.assertIn("--metadata-json", argv)
        self.assertIn("--json", argv)
        self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], "mlx")
        self.assertEqual(argv[argv.index("--codec-artifact-repo") + 1], "owner/codec")
        self.assertIn(str(Path("/tmp/upstream").resolve()), env["PYTHONPATH"])

    def test_build_command_forwards_cleanup_between_requests(self):
        args = self._args()
        args.cleanup_between_requests = True
        argv, _ = bench.build_command(args, Path("/tmp/requests.json"), Path("/tmp/metadata.json"))
        self.assertIn("--cleanup-between-requests", argv)

    def test_parse_batch_metadata_labels_warmup(self):
        metadata = {
            "results": [
                {
                    "result": {"output_wav": "/tmp/1.wav", "resolved_seconds": 1.2, "timings_ms": {"total_to_decode": 10, "sample_rf": 7}},
                    "request": {"text": "one", "seed": 1, "num_steps": 12, "cfg_guidance_mode": "independent", "cfg_scale_text": 3.0},
                    "batch": {"index": 1},
                },
                {
                    "result": {"output_wav": "/tmp/2.wav", "resolved_seconds": 1.1, "timings_ms": {"total_to_decode": 8, "sample_rf": 5, "audio_write_wav": 1.0}},
                    "request": {"text": "two", "seed": 2, "num_steps": 12, "cfg_guidance_mode": "joint", "cfg_scale_text": 1.0},
                    "batch": {"index": 2},
                },
            ]
        }
        parsed = bench.parse_batch_metadata(metadata, warmup_requests=1)
        self.assertEqual([item.phase for item in parsed], ["warmup", "measured"])
        self.assertEqual(parsed[1].timings_ms["sample_rf"], 5.0)
        self.assertEqual(parsed[1].cfg_guidance_mode, "joint")
        self.assertEqual(parsed[1].cfg_scale_text, 1.0)
        self.assertEqual(parsed[1].output_duration_seconds, 1.1)

    def test_build_json_summary_computes_throughput(self):
        requests = (
            bench.RequestResult(
                1,
                "warmup",
                "/tmp/1.wav",
                "one",
                1,
                12,
                "independent",
                3.0,
                1.0,
                {"total_to_decode": 10.0},
                None,
                "mlx",
            ),
            bench.RequestResult(
                2,
                "measured",
                "/tmp/2.wav",
                "two",
                2,
                12,
                "independent",
                3.0,
                1.0,
                {"total_to_decode": 8.0, "audio_write_wav": 1.0, "decode_dacvae_materialization": 3.0},
                None,
                "mlx",
            ),
            bench.RequestResult(
                3,
                "measured",
                "/tmp/3.wav",
                "three",
                3,
                12,
                "independent",
                3.0,
                1.0,
                {"total_to_decode": 12.0, "audio_write_wav": 2.0, "decode_dacvae_materialization": 5.0},
                None,
                "mlx",
            ),
        )
        result = bench.BatchRunResult(
            command="python ...",
            cwd="/tmp/repo",
            request_count=3,
            warmup_requests=1,
            measured_requests=2,
            metadata_json="/tmp/metadata.json",
            requests_json="/tmp/requests.json",
            stdout_log="/tmp/stdout.log",
            stderr_log="/tmp/stderr.log",
            status="passed",
            wall_seconds=2.0,
            max_rss_bytes=100,
            process_setup_overhead_ms=1970.0,
            requests=requests,
        )
        summary = bench.build_json_summary(result, args=self._args())
        self.assertEqual(summary["aggregates"]["measured_total_to_decode_ms"]["median"], 10.0)
        self.assertEqual(summary["aggregates"]["measured_audio_write_ms"]["median"], 1.5)
        self.assertEqual(summary["aggregates"]["measured_output_duration_seconds"]["median"], 1.0)
        self.assertEqual(summary["process"]["measured_generation_throughput_rps"], 100.0)
        report = bench.build_report(result, args=self._args())
        self.assertIn("Persistent Batch Benchmark Report", report)
        self.assertIn("100.000 req/s", report)
        self.assertIn("audio_write_wav", report)
        self.assertIn("MLX decode subphase aggregates", report)
        self.assertIn("| materialization/sync |", report)
        self.assertIn("4.0 ms", report)

    def test_self_test_path(self):
        self.assertEqual(bench.run_self_test(), 0)


if __name__ == "__main__":
    unittest.main()
