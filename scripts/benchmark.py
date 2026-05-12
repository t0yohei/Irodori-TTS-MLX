#!/usr/bin/env python3
"""Run reproducible upstream/MLX bridge benchmarks for Irodori-TTS.

The script intentionally keeps the orchestration lightweight:
- it shells out to upstream `infer.py` or this repo's `scripts/generate_wav.py`
- it parses timing lines and `/usr/bin/time -l` output
- it can emit a Markdown report that is safe to commit

Heavy model/runtime dependencies are optional. Use `--self-test` to validate the
parser/report logic without local checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TIME_L_BIN = "/usr/bin/time"
DEFAULT_TEXT = "今日はいい天気ですね。"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
TIMING_RE = re.compile(
    r"^\[timing\]\s+([a-zA-Z0-9_\-]+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\s*$"
)
WALL_RE = re.compile(r"(?:^|\s)([0-9]+(?:\.[0-9]+)?)\s+real(?:\s|$)")
RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$")


class BenchmarkError(RuntimeError):
    """Raised when benchmark execution or parsing fails."""


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: str
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    kind: str
    command: str
    cwd: str
    output_wav: str
    stdout_log: str
    stderr_log: str
    status: str
    timings_ms: dict[str, float]
    wall_seconds: float | None
    max_rss_bytes: int | None
    notes: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark upstream PyTorch and/or MLX bridge inference.")
    parser.add_argument("--self-test", action="store_true", help="Run dependency-light parser/report self-tests.")
    parser.add_argument("--mode", choices=("upstream", "mlx", "both"), default="both")
    parser.add_argument("--output-dir", default="benchmark-runs", help="Directory for logs, wavs, and JSON summaries.")
    parser.add_argument("--report", help="Optional Markdown report path to write.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--seconds", type=float, default=5.0, help="Target output seconds for MLX bridge runs.")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--reference-wav", help="Reference audio path. Optional for both modes.")
    parser.add_argument("--upstream-root", help="Path to upstream Irodori-TTS checkout.")
    parser.add_argument("--upstream-python", default="python3", help="Python executable for upstream benchmark.")
    parser.add_argument("--mlx-python", default="python3", help="Python executable for MLX benchmark.")
    parser.add_argument("--weights", help="Converted MLX .npz weights for MLX bridge benchmark.")
    parser.add_argument("--codec-device", default="cpu", help="Codec device for MLX bridge benchmark.")
    parser.add_argument(
        "--codec-runtime-mode",
        default="persistent",
        choices=("persistent", "subprocess"),
        help="How to host the PyTorch DACVAE bridge during MLX runs.",
    )
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--model-config-json", help="Optional inline/path JSON for MLX ModelConfig.")
    parser.add_argument("--text-tokenizer-repo")
    parser.add_argument("--caption-tokenizer-repo")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and exit without running them.")
    return parser.parse_args()


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> CommandResult:
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return CommandResult(
        argv=list(argv),
        cwd=str(cwd),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=int(completed.returncode),
    )


def parse_timing_lines(text: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for line in text.splitlines():
        match = TIMING_RE.match(line.strip())
        if not match:
            continue
        value = float(match.group(2))
        unit = match.group(3)
        if unit == "s":
            value *= 1000.0
        timings[match.group(1)] = value
    return timings


def parse_wall_seconds(stderr: str) -> float | None:
    for line in stderr.splitlines():
        match = WALL_RE.search(line)
        if match:
            return float(match.group(1))
    return None


def parse_max_rss_bytes(stderr: str) -> int | None:
    for line in stderr.splitlines():
        match = RSS_RE.match(line)
        if match:
            return int(match.group(1))
    return None


def write_logs(base_path: Path, result: CommandResult) -> tuple[str, str]:
    stdout_path = base_path.with_suffix(".stdout.log")
    stderr_path = base_path.with_suffix(".stderr.log")
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def build_upstream_command(args: argparse.Namespace, output_dir: Path) -> tuple[list[str], Path]:
    if not args.upstream_root:
        raise BenchmarkError("--upstream-root is required for upstream mode")
    output_wav = output_dir / ("upstream-ref.wav" if args.reference_wav else "upstream-no-ref.wav")
    argv = [
        TIME_L_BIN,
        "-l",
        args.upstream_python,
        "infer.py",
        "--hf-checkpoint",
        "Aratako/Irodori-TTS-500M-v2",
        "--text",
        args.text,
        "--output-wav",
        str(output_wav),
        "--model-device",
        "mps",
        "--codec-device",
        "mps",
        "--model-precision",
        "fp32",
        "--codec-precision",
        "fp32",
        "--num-steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
        "--show-timings",
    ]
    if args.reference_wav:
        argv.extend([
            "--ref-wav",
            args.reference_wav,
            "--max-ref-seconds",
            "30",
            "--ref-normalize-db",
            "-16",
        ])
    else:
        argv.append("--no-ref")
    return argv, output_wav


def build_mlx_command(args: argparse.Namespace, repo_root: Path, output_dir: Path) -> tuple[list[str], Path, dict[str, str]]:
    if not args.weights:
        raise BenchmarkError("--weights is required for MLX mode")
    output_wav = output_dir / ("mlx-ref.wav" if args.reference_wav else "mlx-no-ref.wav")
    argv = [
        TIME_L_BIN,
        "-l",
        args.mlx_python,
        "scripts/generate_wav.py",
        "--weights",
        args.weights,
        "--output",
        str(output_wav),
        "--text",
        args.text,
        "--seconds",
        str(args.seconds),
        "--num-steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
        "--codec-repo",
        args.codec_repo,
        "--codec-device",
        args.codec_device,
        "--codec-runtime-mode",
        args.codec_runtime_mode,
    ]
    if args.reference_wav:
        argv.extend(["--reference-wav", args.reference_wav])
    else:
        argv.append("--no-reference")
    if args.model_config_json:
        argv.extend(["--model-config-json", args.model_config_json])
    if args.text_tokenizer_repo:
        argv.extend(["--text-tokenizer-repo", args.text_tokenizer_repo])
    if args.caption_tokenizer_repo:
        argv.extend(["--caption-tokenizer-repo", args.caption_tokenizer_repo])
    env = os.environ.copy()
    upstream_root = args.upstream_root
    if upstream_root:
        pythonpath_parts = [str(Path(upstream_root).resolve())]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return argv, output_wav, env


def summarize_result(name: str, kind: str, command_result: CommandResult, output_wav: Path, stdout_log: str, stderr_log: str) -> BenchmarkResult:
    timings_ms = parse_timing_lines(command_result.stdout + "\n" + command_result.stderr)
    wall_seconds = parse_wall_seconds(command_result.stderr)
    max_rss_bytes = parse_max_rss_bytes(command_result.stderr)
    status = "passed" if command_result.returncode == 0 else f"failed ({command_result.returncode})"
    notes: list[str] = []
    if not timings_ms:
        notes.append("No [timing] lines were detected.")
    return BenchmarkResult(
        name=name,
        kind=kind,
        command=shell_join(command_result.argv),
        cwd=command_result.cwd,
        output_wav=str(output_wav),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        status=status,
        timings_ms=timings_ms,
        wall_seconds=wall_seconds,
        max_rss_bytes=max_rss_bytes,
        notes=tuple(notes),
    )


def run_upstream(args: argparse.Namespace, repo_root: Path, output_dir: Path) -> BenchmarkResult:
    argv, output_wav = build_upstream_command(args, output_dir)
    cwd = Path(args.upstream_root).resolve()
    if args.dry_run:
        return BenchmarkResult(
            name="upstream-base",
            kind="upstream",
            command=shell_join(argv),
            cwd=str(cwd),
            output_wav=str(output_wav),
            stdout_log="",
            stderr_log="",
            status="dry-run",
            timings_ms={},
            wall_seconds=None,
            max_rss_bytes=None,
        )
    command_result = run_command(argv, cwd=cwd)
    stdout_log, stderr_log = write_logs(output_dir / "upstream-base", command_result)
    if command_result.returncode != 0:
        raise BenchmarkError(command_result.stderr.strip() or "upstream benchmark failed")
    return summarize_result("upstream-base", "upstream", command_result, output_wav, stdout_log, stderr_log)


def run_mlx(args: argparse.Namespace, repo_root: Path, output_dir: Path) -> BenchmarkResult:
    argv, output_wav, env = build_mlx_command(args, repo_root, output_dir)
    if args.dry_run:
        return BenchmarkResult(
            name="mlx-bridge",
            kind="mlx",
            command=shell_join(argv),
            cwd=str(repo_root),
            output_wav=str(output_wav),
            stdout_log="",
            stderr_log="",
            status="dry-run",
            timings_ms={},
            wall_seconds=None,
            max_rss_bytes=None,
        )
    command_result = run_command(argv, cwd=repo_root, env=env)
    stdout_log, stderr_log = write_logs(output_dir / "mlx-bridge", command_result)
    if command_result.returncode != 0:
        raise BenchmarkError(command_result.stderr.strip() or command_result.stdout.strip() or "MLX benchmark failed")
    return summarize_result("mlx-bridge", "mlx", command_result, output_wav, stdout_log, stderr_log)


def format_bytes(value: int | None) -> str:
    if value is None:
        return ""
    gib = value / float(1024**3)
    return f"{value} bytes ({gib:.2f} GiB)"


def format_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f} ms"


def build_report(results: list[BenchmarkResult], *, text: str, seed: int, num_steps: int) -> str:
    lines = [
        "# Apple Silicon Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Prompt text: `{text}`",
        f"- Seed: `{seed}`",
        f"- Num steps: `{num_steps}`",
        "",
        "## Results",
        "",
        "| Run | Status | sample_rf | decode | total_to_decode | wall clock | max RSS |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            "| {name} | {status} | {sample_rf} | {decode} | {total} | {wall} | {rss} |".format(
                name=result.name,
                status=result.status,
                sample_rf=format_ms(result.timings_ms.get("sample_rf")),
                decode=format_ms(result.timings_ms.get("decode_dacvae") or result.timings_ms.get("decode_latent")),
                total=format_ms(result.timings_ms.get("total_to_decode")),
                wall=(f"{result.wall_seconds:.2f} s" if result.wall_seconds is not None else ""),
                rss=format_bytes(result.max_rss_bytes),
            )
        )
    for result in results:
        lines.extend([
            "",
            f"## {result.name}",
            "",
            f"- Kind: `{result.kind}`",
            f"- Status: `{result.status}`",
            f"- CWD: `{result.cwd}`",
            f"- Output WAV: `{result.output_wav}`",
            f"- stdout log: `{result.stdout_log}`" if result.stdout_log else "- stdout log: n/a",
            f"- stderr log: `{result.stderr_log}`" if result.stderr_log else "- stderr log: n/a",
            "",
            "Command:",
            "",
            "```bash",
            result.command,
            "```",
            "",
            "Timings:",
            "",
            "| Stage | Time |",
            "| --- | ---: |",
        ])
        for key in sorted(result.timings_ms):
            lines.append(f"| `{key}` | {format_ms(result.timings_ms[key])} |")
        if not result.timings_ms:
            lines.append("| _(none parsed)_ | |")
        if result.notes:
            lines.extend(["", "Notes:", ""])
            for note in result.notes:
                lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def write_json_summary(results: list[BenchmarkResult], path: Path) -> None:
    path.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2), encoding="utf-8")


def run_self_test() -> int:
    sample = """
[timing] sample_rf: 23713.9 ms
[timing] decode_dacvae: 5648.5 ms
122.86 real
1718976512  maximum resident set size
""".strip()
    timings = parse_timing_lines(sample)
    assert timings["sample_rf"] == 23713.9
    assert timings["decode_dacvae"] == 5648.5
    assert parse_wall_seconds(sample) == 122.86
    assert parse_max_rss_bytes(sample) == 1718976512
    report = build_report(
        [
            BenchmarkResult(
                name="self-test",
                kind="mlx",
                command="python scripts/generate_wav.py ...",
                cwd="/tmp/repo",
                output_wav="/tmp/out.wav",
                stdout_log="/tmp/out.stdout.log",
                stderr_log="/tmp/out.stderr.log",
                status="passed",
                timings_ms=timings,
                wall_seconds=122.86,
                max_rss_bytes=1718976512,
            )
        ],
        text=DEFAULT_TEXT,
        seed=20260512,
        num_steps=40,
    )
    assert "sample_rf" in report
    assert "122.86 s" in report
    print("self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_directory((repo_root / args.output_dir).resolve())
    results: list[BenchmarkResult] = []

    if args.mode in {"upstream", "both"}:
        results.append(run_upstream(args, repo_root, output_dir))
    if args.mode in {"mlx", "both"}:
        results.append(run_mlx(args, repo_root, output_dir))

    write_json_summary(results, output_dir / "benchmark-summary.json")
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(results, text=args.text, seed=args.seed, num_steps=args.num_steps), encoding="utf-8")

    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
