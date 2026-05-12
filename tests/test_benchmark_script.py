from __future__ import annotations

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
            reference_wav=None,
            text="hello",
            seconds=5.0,
            num_steps=40,
            seed=123,
            codec_repo="codec-repo",
            codec_device="cpu",
            codec_runtime_mode="subprocess",
            model_config_json=None,
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            upstream_root="/tmp/upstream",
            mlx_python="python3",
        )
        argv, output_wav, env = benchmark.build_mlx_command(args, Path("/tmp/repo"), Path("/tmp/out"))
        self.assertIn("--no-reference", argv)
        self.assertIn("--codec-runtime-mode", argv)
        self.assertIn("subprocess", argv)
        self.assertEqual(output_wav, Path("/tmp/out/mlx-no-ref.wav"))
        self.assertIn(str(Path("/tmp/upstream").resolve()), env["PYTHONPATH"])

    def test_build_report_renders_summary_table(self):
        result = benchmark.BenchmarkResult(
            name="mlx-bridge",
            kind="mlx",
            command="python scripts/generate_wav.py ...",
            cwd="/tmp/repo",
            output_wav="/tmp/out.wav",
            stdout_log="/tmp/out.stdout.log",
            stderr_log="/tmp/out.stderr.log",
            status="passed",
            timings_ms={"sample_rf": 1200.0, "decode_dacvae": 340.0, "total_to_decode": 1700.0},
            wall_seconds=3.25,
            max_rss_bytes=1024,
        )
        report = benchmark.build_report([result], text="hello", seed=1, num_steps=40)
        self.assertIn("| mlx-bridge | passed | 1200.0 ms | 340.0 ms | 1700.0 ms | 3.25 s |", report)
        self.assertIn("python scripts/generate_wav.py ...", report)

    def test_self_test_path(self):
        self.assertEqual(benchmark.run_self_test(), 0)


if __name__ == "__main__":
    unittest.main()
