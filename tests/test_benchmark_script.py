from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import scripts.benchmark as benchmark


class BenchmarkScriptTests(unittest.TestCase):
    def test_parse_timing_lines_and_time_l_output(self):
        sample = "\n".join(
            [
                "[timing] sample_rf: 23713.9 ms",
                "[timing] decode_dacvae: 5648.5 ms",
                "[timing] total_to_decode: 1.75 s",
                "122.86 real 0.00 user 0.00 sys",
                "1718976512  maximum resident set size",
            ]
        )
        self.assertEqual(
            benchmark.parse_timing_lines(sample),
            {"sample_rf": 23713.9, "decode_dacvae": 5648.5, "total_to_decode": 1750.0},
        )
        self.assertEqual(benchmark.parse_wall_seconds(sample), 122.86)
        self.assertEqual(benchmark.parse_max_rss_bytes(sample), 1718976512)

    def test_build_mlx_command_adds_no_reference_and_pythonpath(self):
        args = Namespace(
            weights="weights.npz",
            ref_wav=None,
            text="hello",
            caption="soft voice",
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root="/tmp/upstream",
            mlx_python="python3",
            weights_dir=None,
            weights_repo=None,
            weights_revision=None,
        )
        argv, env = benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out.wav"), seconds=5.0, num_steps=40)
        self.assertIn("--no-ref", argv)
        self.assertIn("--caption", argv)
        self.assertEqual(argv[argv.index("--caption") + 1], "soft voice")
        self.assertIn("--codec-runtime-mode", argv)
        self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], "mlx")
        self.assertIn(str(Path("/tmp/upstream").resolve()), env["PYTHONPATH"])
        self.assertEqual(argv[argv.index("--output-wav") + 1], "/tmp/out.wav")

    def test_build_mlx_command_can_omit_seconds_for_predicted_duration(self):
        args = Namespace(
            weights="weights.npz",
            ref_wav=None,
            text="hello",
            caption=None,
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            codec_path=None,
            codec_artifact_dir=None,
            codec_artifact_repo=None,
            codec_artifact_revision=None,
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root=None,
            mlx_python="python3",
            weights_dir=None,
            weights_repo=None,
            weights_revision=None,
        )
        argv, _env = benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out.wav"), seconds=None, num_steps=12)
        self.assertNotIn("--seconds", argv)
        self.assertEqual(argv[argv.index("--num-steps") + 1], "12")

    def test_resolve_case_cwd_raises_benchmark_error_when_upstream_root_missing(self):
        case = benchmark.BenchmarkCase(
            name="upstream-base-no-reference-steps-40",
            slug="upstream-base-no-reference-steps-40",
            kind="upstream",
            case_label="base",
            reference_mode="no-reference",
            seconds=None,
            num_steps=40,
        )
        args = Namespace(upstream_root=None)
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.resolve_case_cwd(case, args, Path("/tmp/repo"))

    def test_build_cases_expands_repeatable_mlx_sweeps(self):
        args = Namespace(
            mode="mlx",
            seconds=5.0,
            seconds_sweep="3,5",
            num_steps=40,
            num_steps_sweep="20,40",
            repeat=2,
            warmup_runs=1,
            ref_wav="ref.wav",
            omit_seconds=False,
            case_label="v3-text",
        )
        cases = benchmark.build_cases(args)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0].kind, "mlx")
        self.assertEqual(cases[0].case_label, "v3-text")
        self.assertEqual(cases[0].reference_mode, "reference")
        self.assertEqual({case.seconds for case in cases}, {3.0, 5.0})
        self.assertEqual({case.num_steps for case in cases}, {20, 40})

    def test_build_cases_rejects_seconds_sweep_for_upstream(self):
        args = Namespace(
            mode="both",
            seconds=5.0,
            seconds_sweep="3,5",
            num_steps=40,
            num_steps_sweep=None,
            repeat=1,
            warmup_runs=0,
            ref_wav=None,
            omit_seconds=False,
            case_label=None,
        )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.build_cases(args)

    def test_build_cases_rejects_duplicate_sweep_values_that_collide_on_slug(self):
        args = Namespace(
            mode="mlx",
            seconds=5.0,
            seconds_sweep="5,5",
            num_steps=40,
            num_steps_sweep="20,20",
            repeat=1,
            warmup_runs=0,
            ref_wav=None,
            omit_seconds=False,
            case_label=None,
        )
        with self.assertRaises(benchmark.BenchmarkError) as ctx:
            benchmark.build_cases(args)
        self.assertIn("duplicate benchmark cases/log paths", str(ctx.exception))

    def test_build_cases_supports_predicted_duration_labeling(self):
        args = Namespace(
            mode="mlx",
            seconds=5.0,
            seconds_sweep=None,
            omit_seconds=True,
            num_steps=12,
            num_steps_sweep="8,12",
            repeat=1,
            warmup_runs=0,
            ref_wav=None,
            case_label="v3-text",
        )
        cases = benchmark.build_cases(args)
        self.assertEqual([case.seconds for case in cases], [None, None])
        self.assertTrue(all("predicted" in case.name for case in cases))
        self.assertTrue(all(case.case_label == "v3-text" for case in cases))

    def test_build_cases_rejects_omit_seconds_with_seconds_sweep(self):
        args = Namespace(
            mode="mlx",
            seconds=5.0,
            seconds_sweep="3,5",
            omit_seconds=True,
            num_steps=12,
            num_steps_sweep=None,
            repeat=1,
            warmup_runs=0,
            ref_wav=None,
            case_label=None,
        )
        with self.assertRaises(benchmark.BenchmarkError) as ctx:
            benchmark.build_cases(args)
        self.assertIn("--omit-seconds cannot be combined with --seconds-sweep", str(ctx.exception))

    def test_run_case_dry_run_preserves_warmup_and_measured_metadata(self):
        case = benchmark.BenchmarkCase(
            name="mlx-bridge-reference-seconds-5-steps-40",
            slug="mlx-bridge-reference-seconds-5-steps-40",
            kind="mlx",
            case_label="base",
            reference_mode="reference",
            seconds=5.0,
            num_steps=40,
        )
        args = Namespace(
            warmup_runs=1,
            repeat=2,
            dry_run=True,
            weights="weights.npz",
            ref_wav="ref.wav",
            text="hello",
            caption=None,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root=None,
            mlx_python="python3",
            weights_dir=None,
            weights_repo=None,
            weights_revision=None,
            cache_state="auto",
        )
        results = benchmark.run_case(case, args, Path("/tmp/repo"), Path("/tmp/out"))
        self.assertEqual([result.phase for result in results], ["warmup", "measured", "measured"])
        self.assertEqual([result.run_index for result in results], [1, 1, 2])
        self.assertEqual([result.overall_run_index for result in results], [1, 2, 3])
        self.assertEqual([result.cache_state for result in results], ["cold", "warm", "warm"])
        self.assertTrue(results[0].output_wav.endswith(".warmup.run-01.wav"))
        self.assertTrue(results[1].output_wav.endswith(".measured.run-01.wav"))
        self.assertEqual(results[0].status, "dry-run")

    def test_build_mlx_command_supports_hosted_weights_repo_revision(self):
        args = Namespace(
            weights=None,
            weights_dir=None,
            weights_repo="owner/irodori-hosted",
            weights_revision="abc123",
            ref_wav=None,
            text="hello",
            caption=None,
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root=None,
            mlx_python="python3",
        )
        argv, _env = benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out.wav"), seconds=2.0, num_steps=12)
        self.assertIn("--weights-repo", argv)
        self.assertEqual(argv[argv.index("--weights-repo") + 1], "owner/irodori-hosted")
        self.assertIn("--weights-revision", argv)
        self.assertEqual(argv[argv.index("--weights-revision") + 1], "abc123")
        self.assertNotIn("--model-config-json", argv)

    def test_build_mlx_command_rejects_multiple_weights_sources(self):
        args = Namespace(
            weights="weights.npz",
            weights_dir="/tmp/layout",
            weights_repo=None,
            weights_revision=None,
            ref_wav=None,
            text="hello",
            caption=None,
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            codec_path=None,
            codec_artifact_dir=None,
            codec_artifact_repo=None,
            codec_artifact_revision=None,
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root=None,
            mlx_python="python3",
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "choose only one MLX weights source"):
            benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out.wav"), seconds=2.0, num_steps=12)

    def test_build_mlx_command_supports_hosted_codec_artifact_revision(self):
        args = Namespace(
            weights=None,
            weights_dir=None,
            weights_repo="owner/irodori-hosted",
            weights_revision="weights123",
            ref_wav=None,
            text="hello",
            caption=None,
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            codec_path=None,
            codec_artifact_dir=None,
            codec_artifact_repo="owner/codec-hosted",
            codec_artifact_revision="codec123",
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root=None,
            mlx_python="python3",
        )
        argv, _env = benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out.wav"), seconds=None, num_steps=12)
        self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], "mlx")
        self.assertIn("--codec-artifact-repo", argv)
        self.assertEqual(argv[argv.index("--codec-artifact-repo") + 1], "owner/codec-hosted")
        self.assertIn("--codec-artifact-revision", argv)
        self.assertEqual(argv[argv.index("--codec-artifact-revision") + 1], "codec123")

    def test_build_mlx_command_rejects_codec_revision_without_repo(self):
        args = Namespace(
            weights="weights.npz",
            weights_dir=None,
            weights_repo=None,
            weights_revision=None,
            ref_wav=None,
            text="hello",
            caption=None,
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="mlx",
            codec_path=None,
            codec_artifact_dir=None,
            codec_artifact_repo=None,
            codec_artifact_revision="codec123",
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root=None,
            mlx_python="python3",
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "--codec-artifact-revision requires --codec-artifact-repo"):
            benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out.wav"), seconds=2.0, num_steps=12)

    def test_resolve_cache_state_auto_separates_cold_and_warm(self):
        args = Namespace(cache_state="auto", warmup_runs=0, repeat=3)
        self.assertEqual(
            benchmark.resolve_cache_state(args, phase="measured", overall_run_index=1, measured_run_index=1),
            "cold",
        )
        self.assertEqual(
            benchmark.resolve_cache_state(args, phase="measured", overall_run_index=2, measured_run_index=2),
            "warm",
        )

        warm_args = Namespace(cache_state="auto", warmup_runs=1, repeat=2)
        self.assertEqual(
            benchmark.resolve_cache_state(warm_args, phase="warmup", overall_run_index=1, measured_run_index=None),
            "cold",
        )
        self.assertEqual(
            benchmark.resolve_cache_state(warm_args, phase="measured", overall_run_index=2, measured_run_index=1),
            "warm",
        )

    def test_build_aggregates_summarizes_by_case_phase_and_cache_state(self):
        results = [
            benchmark.BenchmarkResult(
                name="mlx-case-measured-run-01",
                case_name="mlx-case",
                kind="mlx",
                phase="measured",
                run_index=1,
                overall_run_index=1,
                cache_state="warm",
                reference_mode="reference",
                seconds=5.0,
                num_steps=40,
                command="python ...",
                cwd="/tmp/repo",
                output_wav="/tmp/out1.wav",
                stdout_log="/tmp/out1.stdout.log",
                stderr_log="/tmp/out1.stderr.log",
                status="passed",
                timings_ms={"sample_rf": 1000.0, "total_to_decode": 1400.0},
                wall_seconds=2.0,
                max_rss_bytes=100,
            ),
            benchmark.BenchmarkResult(
                name="mlx-case-measured-run-02",
                case_name="mlx-case",
                kind="mlx",
                phase="measured",
                run_index=2,
                overall_run_index=2,
                cache_state="warm",
                reference_mode="reference",
                seconds=5.0,
                num_steps=40,
                command="python ...",
                cwd="/tmp/repo",
                output_wav="/tmp/out2.wav",
                stdout_log="/tmp/out2.stdout.log",
                stderr_log="/tmp/out2.stderr.log",
                status="passed",
                timings_ms={"sample_rf": 1500.0, "total_to_decode": 1600.0},
                wall_seconds=3.0,
                max_rss_bytes=200,
            ),
        ]
        aggregates = benchmark.build_aggregates(results)
        self.assertEqual(len(aggregates), 1)
        aggregate = aggregates[0]
        self.assertEqual(aggregate["runs"], 2)
        self.assertEqual(aggregate["timings_ms"]["sample_rf"]["median"], 1250.0)
        self.assertEqual(aggregate["wall_seconds"]["max"], 3.0)
        self.assertEqual(aggregate["max_rss_bytes"]["min"], 100)

    def test_build_report_renders_aggregate_and_raw_run_tables(self):
        results = [
            benchmark.BenchmarkResult(
                name="mlx-case-measured-run-01",
                case_name="mlx-case",
                kind="mlx",
                phase="measured",
                run_index=1,
                overall_run_index=1,
                cache_state="warm",
                reference_mode="reference",
                seconds=5.0,
                num_steps=40,
                command="python scripts/generate_wav.py ...",
                cwd="/tmp/repo",
                output_wav="/tmp/out.wav",
                stdout_log="/tmp/out.stdout.log",
                stderr_log="/tmp/out.stderr.log",
                status="passed",
                timings_ms={"sample_rf": 1200.0, "encode_dacvae": 240.0, "decode_dacvae": 340.0, "total_to_decode": 1700.0},
                wall_seconds=3.25,
                max_rss_bytes=1024,
            )
        ]
        report = benchmark.build_report(results, text="hello", seed=1, repeat=1, warmup_runs=0, cache_state_mode="auto")
        self.assertIn("Aggregate results", report)
        self.assertIn("setup/load median", report)
        self.assertIn("codec encode median", report)
        self.assertIn("| mlx-case | measured | warm | 1 | 1550.0 ms | 1200.0 ms | 1200.0 ms / 1200.0 ms | 240.0 ms | 340.0 ms | 1700.0 ms |", report)
        self.assertIn("Raw runs:", report)
        self.assertIn("python scripts/generate_wav.py ...", report)

    def test_build_report_labels_mlx_none_seconds_as_predicted_duration(self):
        results = [
            benchmark.BenchmarkResult(
                name="mlx-predicted-measured-run-01",
                case_name="mlx-predicted",
                kind="mlx",
                phase="measured",
                run_index=1,
                overall_run_index=1,
                cache_state="warm",
                reference_mode="no-reference",
                seconds=None,
                num_steps=12,
                command="python scripts/generate_wav.py ...",
                cwd="/tmp/repo",
                output_wav="/tmp/out.wav",
                stdout_log="/tmp/out.stdout.log",
                stderr_log="/tmp/out.stderr.log",
                status="passed",
                timings_ms={"sample_rf": 1200.0},
                wall_seconds=3.25,
                max_rss_bytes=1024,
            )
        ]
        report = benchmark.build_report(results, text="hello", seed=1, repeat=1, warmup_runs=0, cache_state_mode="auto")
        self.assertIn("- Kind: `mlx`", report)
        self.assertIn("- Seconds: predicted duration (`--seconds` omitted)", report)
        self.assertNotIn("- Seconds: n/a (upstream)", report)

    def test_write_json_summary_emits_schema_v2_payload(self):
        results = [
            benchmark.BenchmarkResult(
                name="mlx-case-measured-run-01",
                case_name="mlx-case",
                kind="mlx",
                phase="measured",
                run_index=1,
                overall_run_index=1,
                cache_state="unknown",
                reference_mode="no-reference",
                seconds=5.0,
                num_steps=40,
                command="python ...",
                cwd="/tmp/repo",
                output_wav="/tmp/out.wav",
                stdout_log="/tmp/out.stdout.log",
                stderr_log="/tmp/out.stderr.log",
                status="passed",
                timings_ms={"sample_rf": 1000.0},
                wall_seconds=2.0,
                max_rss_bytes=100,
            )
        ]
        args = Namespace(
            mode="mlx",
            case_label="v3-text",
            text="hello",
            caption="soft voice",
            seed=1,
            repeat=1,
            warmup_runs=0,
            cache_state="auto",
            seconds=5.0,
            seconds_sweep=None,
            omit_seconds=True,
            num_steps=40,
            num_steps_sweep=None,
            ref_wav=None,
            weights="weights.npz",
            weights_dir=None,
            weights_repo=None,
            weights_revision=None,
            codec_runtime_mode="mlx",
            codec_path=None,
            codec_artifact_dir=None,
            codec_artifact_repo=None,
            codec_artifact_revision=None,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "summary.json"
            benchmark.write_json_summary(results, path, args=args)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["invocation"]["mode"], "mlx")
        self.assertEqual(payload["invocation"]["case_label"], "v3-text")
        self.assertEqual(payload["invocation"]["caption"], "soft voice")
        self.assertTrue(payload["invocation"]["omit_seconds"])
        self.assertEqual(payload["invocation"]["codec_runtime_mode"], "mlx")
        self.assertIsNone(payload["invocation"]["codec_artifact_repo"])
        self.assertEqual(payload["results"][0]["case_name"], "mlx-case")
        self.assertEqual(payload["aggregates"][0]["timings_ms"]["sample_rf"]["median"], 1000.0)

    def test_self_test_path(self):
        self.assertEqual(benchmark.run_self_test(), 0)


if __name__ == "__main__":
    unittest.main()
