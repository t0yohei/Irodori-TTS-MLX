#!/usr/bin/env python3
"""Run reproducible upstream/MLX bridge benchmarks for Irodori-TTS.

The script intentionally keeps the orchestration lightweight:
- it shells out to upstream `infer.py` or this repo's `scripts/generate_wav.py`
- it parses timing lines and `/usr/bin/time -l` output
- it can repeat runs, label warm/cold phases, and sweep selected parameters
- it can emit Markdown and JSON summaries that are safe to commit

Heavy model/runtime dependencies are optional. Use `--self-test` to validate the
parser/report logic without local checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
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
class BenchmarkCase:
    name: str
    slug: str
    kind: str
    case_label: str
    reference_mode: str
    seconds: float | None
    num_steps: int


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    case_name: str
    kind: str
    phase: str
    run_index: int
    overall_run_index: int
    cache_state: str
    reference_mode: str
    seconds: float | None
    num_steps: int
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
    parser.add_argument("--caption", help="Optional caption/style prompt for caption-conditioned MLX runs.")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--seconds", type=float, default=5.0, help="Target output seconds for MLX bridge runs.")
    parser.add_argument("--seconds-sweep", help="Optional comma-separated MLX output-length sweep, e.g. 3,5,8.")
    parser.add_argument(
        "--omit-seconds",
        action="store_true",
        help="Omit --seconds on MLX runs so checkpoints with predicted duration can exercise their automatic duration path.",
    )
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--num-steps-sweep", help="Optional comma-separated diffusion-step sweep, e.g. 20,40,60.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of measured runs per benchmark case.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Optional warmup runs executed before measured runs. Warmups are recorded separately from measured runs.",
    )
    parser.add_argument(
        "--cache-state",
        choices=("auto", "cold", "warm", "unknown"),
        default="auto",
        help="Label cache state for each run. 'auto' uses invocation order heuristics to separate first-run vs steady-state behavior.",
    )
    parser.add_argument("--reference-wav", help="Reference audio path. Optional for both modes.")
    parser.add_argument("--upstream-root", help="Path to upstream Irodori-TTS checkout.")
    parser.add_argument("--upstream-python", default="python3", help="Python executable for upstream benchmark.")
    parser.add_argument("--mlx-python", default="python3", help="Python executable for MLX benchmark.")
    parser.add_argument("--weights", help="Converted MLX .npz weights for MLX bridge benchmark.")
    parser.add_argument(
        "--weights-dir",
        help="Local hosted/pre-converted weights layout directory for MLX bridge benchmark.",
    )
    parser.add_argument(
        "--weights-repo",
        dest="weights_repo",
        help="Hugging Face repo id with a hosted/pre-converted weights layout for MLX bridge benchmark.",
    )
    parser.add_argument("--weights-revision", help="Optional Hugging Face revision for --weights-repo.")
    parser.add_argument("--codec-device", default="cpu", help="Codec device for MLX bridge benchmark.")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument(
        "--codec-runtime-mode",
        choices=("persistent", "subprocess"),
        default="persistent",
        help="DACVAE bridge runtime mode for MLX benchmark runs.",
    )
    parser.add_argument("--model-config-json", help="Optional inline/path JSON for MLX ModelConfig.")
    parser.add_argument("--text-tokenizer-repo")
    parser.add_argument("--caption-tokenizer-repo")
    parser.add_argument(
        "--case-label",
        help="Optional label to prefix benchmark case names/log slugs, e.g. v3-text or voicedesign-caption.",
    )
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


def parse_number_list(raw: str | None, *, cast, option_name: str) -> list[int] | list[float]:
    if raw is None:
        return []
    values: list[int] | list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            values.append(cast(text))
        except ValueError as exc:
            raise BenchmarkError(f"{option_name} contains an invalid value: {text!r}") from exc
    if not values:
        raise BenchmarkError(f"{option_name} did not contain any usable values")
    return values


def slug_token(value: str) -> str:
    token = value.strip().lower()
    token = token.replace(" ", "-")
    token = token.replace(".", "p")
    return re.sub(r"[^a-z0-9\-]+", "-", token).strip("-")


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.repeat < 1:
        raise BenchmarkError("--repeat must be >= 1")
    if args.warmup_runs < 0:
        raise BenchmarkError("--warmup-runs must be >= 0")

    if args.omit_seconds and args.seconds_sweep:
        raise BenchmarkError("--omit-seconds cannot be combined with --seconds-sweep")

    seconds_values: list[float | None] = [None] if args.omit_seconds else [float(args.seconds)]
    if args.seconds_sweep:
        seconds_values = [float(v) for v in parse_number_list(args.seconds_sweep, cast=float, option_name="--seconds-sweep")]
    step_values = [int(args.num_steps)]
    if args.num_steps_sweep:
        step_values = [int(v) for v in parse_number_list(args.num_steps_sweep, cast=int, option_name="--num-steps-sweep")]

    if args.mode in {"upstream", "both"} and args.seconds_sweep:
        raise BenchmarkError("--seconds-sweep is only supported for --mode mlx because upstream infer.py has no output-length flag")

    reference_mode = "reference" if args.reference_wav else "no-reference"
    case_label = slug_token(args.case_label) if getattr(args, "case_label", None) else "base"
    cases: list[BenchmarkCase] = []

    if args.mode in {"upstream", "both"}:
        for steps in step_values:
            suffix = f"{case_label}-{reference_mode}-steps-{steps}"
            cases.append(
                BenchmarkCase(
                    name=f"upstream-{suffix}",
                    slug=f"upstream-{slug_token(suffix)}",
                    kind="upstream",
                    case_label=case_label,
                    reference_mode=reference_mode,
                    seconds=None,
                    num_steps=steps,
                )
            )

    if args.mode in {"mlx", "both"}:
        for seconds in seconds_values:
            for steps in step_values:
                seconds_label = "predicted" if seconds is None else f"seconds-{seconds:g}"
                suffix = f"{case_label}-{reference_mode}-{seconds_label}-steps-{steps}"
                cases.append(
                    BenchmarkCase(
                        name=f"mlx-bridge-{suffix}",
                        slug=f"mlx-bridge-{slug_token(suffix)}",
                        kind="mlx",
                        case_label=case_label,
                        reference_mode=reference_mode,
                        seconds=None if seconds is None else float(seconds),
                        num_steps=steps,
                    )
                )

    seen_slugs: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.slug in seen_slugs:
            duplicates.append(case.slug)
            continue
        seen_slugs.add(case.slug)
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise BenchmarkError(
            f"Sweep arguments produced duplicate benchmark cases/log paths: {joined}. "
            "Deduplicate --seconds-sweep/--num-steps-sweep values before running."
        )

    return cases


def build_upstream_command(args: argparse.Namespace, output_wav: Path, *, num_steps: int) -> list[str]:
    if not args.upstream_root:
        raise BenchmarkError("--upstream-root is required for upstream mode")
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
        str(num_steps),
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
    return argv


def build_mlx_command(args: argparse.Namespace, repo_root: Path, output_wav: Path, *, seconds: float | None, num_steps: int) -> tuple[list[str], dict[str, str]]:
    weight_sources = [
        ("--weights", args.weights),
        ("--weights-dir", getattr(args, "weights_dir", None)),
        ("--weights-repo", getattr(args, "weights_repo", None)),
    ]
    selected_weight_sources = [(flag, value) for flag, value in weight_sources if value]
    if not selected_weight_sources:
        raise BenchmarkError("--weights, --weights-dir, or --weights-repo is required for MLX mode")
    if len(selected_weight_sources) > 1:
        selected = ", ".join(flag for flag, _value in selected_weight_sources)
        raise BenchmarkError(f"choose only one MLX weights source, got: {selected}")
    weight_flag, weight_value = selected_weight_sources[0]
    argv = [
        TIME_L_BIN,
        "-l",
        args.mlx_python,
        "scripts/generate_wav.py",
        weight_flag,
        str(weight_value),
        "--output",
        str(output_wav),
        "--text",
        args.text,
        "--num-steps",
        str(num_steps),
        "--seed",
        str(args.seed),
        "--codec-repo",
        args.codec_repo,
        "--codec-device",
        args.codec_device,
        "--codec-runtime-mode",
        args.codec_runtime_mode,
    ]
    if weight_flag == "--weights-repo" and getattr(args, "weights_revision", None):
        argv.extend(["--weights-revision", args.weights_revision])
    if seconds is not None:
        argv.extend(["--seconds", str(seconds)])
    if args.reference_wav:
        argv.extend(["--reference-wav", args.reference_wav])
    else:
        argv.append("--no-reference")
    if args.caption:
        argv.extend(["--caption", args.caption])
    if args.model_config_json:
        if weight_flag != "--weights":
            raise BenchmarkError("--model-config-json is only valid with --weights; hosted layouts provide model_config.json")
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
    return argv, env


def resolve_case_cwd(case: BenchmarkCase, args: argparse.Namespace, repo_root: Path) -> Path:
    if case.kind == "upstream":
        if not args.upstream_root:
            raise BenchmarkError("--upstream-root is required for upstream mode")
        return Path(args.upstream_root).resolve()
    return repo_root


def write_logs(base_path: Path, result: CommandResult) -> tuple[str, str]:
    stdout_path = base_path.with_suffix(".stdout.log")
    stderr_path = base_path.with_suffix(".stderr.log")
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def resolve_cache_state(args: argparse.Namespace, *, phase: str, overall_run_index: int, measured_run_index: int | None) -> str:
    if args.cache_state != "auto":
        return str(args.cache_state)
    if phase == "warmup":
        return "cold" if overall_run_index == 1 else "warm"
    if args.warmup_runs > 0:
        return "warm"
    if args.repeat == 1:
        return "unknown"
    assert measured_run_index is not None
    return "cold" if measured_run_index == 1 else "warm"


def summarize_result(
    *,
    case: BenchmarkCase,
    run_name: str,
    phase: str,
    run_index: int,
    overall_run_index: int,
    cache_state: str,
    command_result: CommandResult,
    output_wav: Path,
    stdout_log: str,
    stderr_log: str,
) -> BenchmarkResult:
    timings_ms = parse_timing_lines(command_result.stdout + "\n" + command_result.stderr)
    wall_seconds = parse_wall_seconds(command_result.stderr)
    max_rss_bytes = parse_max_rss_bytes(command_result.stderr)
    status = "passed" if command_result.returncode == 0 else f"failed ({command_result.returncode})"
    notes: list[str] = []
    if not timings_ms:
        notes.append("No [timing] lines were detected.")
    return BenchmarkResult(
        name=run_name,
        case_name=case.name,
        kind=case.kind,
        phase=phase,
        run_index=run_index,
        overall_run_index=overall_run_index,
        cache_state=cache_state,
        reference_mode=case.reference_mode,
        seconds=case.seconds,
        num_steps=case.num_steps,
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


def run_case(case: BenchmarkCase, args: argparse.Namespace, repo_root: Path, output_dir: Path) -> list[BenchmarkResult]:
    cwd = resolve_case_cwd(case, args, repo_root)

    if args.dry_run:
        dry_results: list[BenchmarkResult] = []
        total_runs = args.warmup_runs + args.repeat
        for overall_run_index in range(1, total_runs + 1):
            is_warmup = overall_run_index <= args.warmup_runs
            phase = "warmup" if is_warmup else "measured"
            phase_run_index = overall_run_index if is_warmup else overall_run_index - args.warmup_runs
            measured_run_index = None if is_warmup else phase_run_index
            cache_state = resolve_cache_state(args, phase=phase, overall_run_index=overall_run_index, measured_run_index=measured_run_index)
            run_slug = f"{case.slug}.{phase}.run-{phase_run_index:02d}"
            run_name = case.name if total_runs == 1 else f"{case.name}-{phase}-run-{phase_run_index:02d}"
            output_wav = output_dir / f"{run_slug}.wav"
            if case.kind == "upstream":
                argv = build_upstream_command(args, output_wav, num_steps=case.num_steps)
            else:
                argv, _env = build_mlx_command(args, repo_root, output_wav, seconds=case.seconds, num_steps=case.num_steps)
            dry_results.append(BenchmarkResult(name=run_name, case_name=case.name, kind=case.kind, phase=phase, run_index=phase_run_index, overall_run_index=overall_run_index, cache_state=cache_state, reference_mode=case.reference_mode, seconds=case.seconds, num_steps=case.num_steps, command=shell_join(argv), cwd=str(cwd), output_wav=str(output_wav), stdout_log="", stderr_log="", status="dry-run", timings_ms={}, wall_seconds=None, max_rss_bytes=None))
        return dry_results

    results: list[BenchmarkResult] = []
    total_runs = args.warmup_runs + args.repeat
    for overall_run_index in range(1, total_runs + 1):
        is_warmup = overall_run_index <= args.warmup_runs
        phase = "warmup" if is_warmup else "measured"
        phase_run_index = overall_run_index if is_warmup else overall_run_index - args.warmup_runs
        measured_run_index = None if is_warmup else phase_run_index
        cache_state = resolve_cache_state(
            args,
            phase=phase,
            overall_run_index=overall_run_index,
            measured_run_index=measured_run_index,
        )
        run_slug = f"{case.slug}.{phase}.run-{phase_run_index:02d}"
        run_name = case.name if total_runs == 1 else f"{case.name}-{phase}-run-{phase_run_index:02d}"
        output_wav = output_dir / f"{run_slug}.wav"
        if case.kind == "upstream":
            argv = build_upstream_command(args, output_wav, num_steps=case.num_steps)
            env = None
        else:
            argv, env = build_mlx_command(args, repo_root, output_wav, seconds=case.seconds, num_steps=case.num_steps)
        command_result = run_command(argv, cwd=cwd, env=env)
        stdout_log, stderr_log = write_logs(output_dir / run_slug, command_result)
        results.append(
            summarize_result(
                case=case,
                run_name=run_name,
                phase=phase,
                run_index=phase_run_index,
                overall_run_index=overall_run_index,
                cache_state=cache_state,
                command_result=command_result,
                output_wav=output_wav,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
        )
        if command_result.returncode != 0:
            raise BenchmarkError(command_result.stderr.strip() or command_result.stdout.strip() or f"{case.kind} benchmark failed")
    return results


def format_bytes(value: int | None) -> str:
    if value is None:
        return ""
    gib = value / float(1024**3)
    return f"{value} bytes ({gib:.2f} GiB)"


def format_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f} ms"


def format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f} s"


def format_case_seconds(kind: str, seconds: float | None) -> str:
    if seconds is not None:
        return f"`{seconds}`"
    if kind == "mlx":
        return "predicted duration (`--seconds` omitted)"
    return "n/a (upstream)"


def summarize_numeric(values: list[float | int]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def build_aggregates(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.case_name, result.kind, result.phase, result.cache_state)
        grouped.setdefault(key, []).append(result)

    aggregates: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        timing_names = sorted({name for result in bucket for name in result.timings_ms})
        timings_summary: dict[str, dict[str, float | int]] = {}
        for timing_name in timing_names:
            values = [result.timings_ms[timing_name] for result in bucket if timing_name in result.timings_ms]
            summary = summarize_numeric(values)
            if summary is not None:
                timings_summary[timing_name] = summary
        wall_summary = summarize_numeric([result.wall_seconds for result in bucket if result.wall_seconds is not None])
        rss_summary = summarize_numeric([result.max_rss_bytes for result in bucket if result.max_rss_bytes is not None])
        status_counts: dict[str, int] = {}
        for result in bucket:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        sample = bucket[0]
        aggregates.append(
            {
                "case_name": sample.case_name,
                "kind": sample.kind,
                "phase": sample.phase,
                "cache_state": sample.cache_state,
                "reference_mode": sample.reference_mode,
                "seconds": sample.seconds,
                "num_steps": sample.num_steps,
                "runs": len(bucket),
                "status_counts": status_counts,
                "timings_ms": timings_summary,
                "wall_seconds": wall_summary,
                "max_rss_bytes": rss_summary,
            }
        )
    return aggregates


def build_report(
    results: list[BenchmarkResult],
    *,
    text: str,
    seed: int,
    repeat: int,
    warmup_runs: int,
    cache_state_mode: str,
) -> str:
    aggregates = build_aggregates(results)
    lines = [
        "# Apple Silicon Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Prompt text: `{text}`",
        f"- Seed: `{seed}`",
        f"- Measured repeats per case: `{repeat}`",
        f"- Warmup runs per case: `{warmup_runs}`",
        f"- Cache-state labeling mode: `{cache_state_mode}`",
        "",
        "## Aggregate results",
        "",
        "| Case | Phase | Cache | Runs | sample_rf median | sample_rf min/max | decode median | total median | wall median | max RSS median |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for aggregate in aggregates:
        sample_rf = aggregate["timings_ms"].get("sample_rf")
        decode = aggregate["timings_ms"].get("decode_dacvae") or aggregate["timings_ms"].get("decode_latent")
        total = aggregate["timings_ms"].get("total_to_decode")
        wall = aggregate["wall_seconds"]
        rss = aggregate["max_rss_bytes"]
        sample_rf_range = ""
        if sample_rf:
            sample_rf_range = f"{format_ms(float(sample_rf['min']))} / {format_ms(float(sample_rf['max']))}"
        lines.append(
            "| {case} | {phase} | {cache} | {runs} | {sample_rf_med} | {sample_rf_range} | {decode_med} | {total_med} | {wall_med} | {rss_med} |".format(
                case=aggregate["case_name"],
                phase=aggregate["phase"],
                cache=aggregate["cache_state"],
                runs=aggregate["runs"],
                sample_rf_med=format_ms(float(sample_rf["median"])) if sample_rf else "",
                sample_rf_range=sample_rf_range,
                decode_med=format_ms(float(decode["median"])) if decode else "",
                total_med=format_ms(float(total["median"])) if total else "",
                wall_med=format_seconds(float(wall["median"])) if wall else "",
                rss_med=format_bytes(int(rss["median"])) if rss else "",
            )
        )

    for aggregate in aggregates:
        lines.extend([
            "",
            f"## {aggregate['case_name']} · {aggregate['phase']} · {aggregate['cache_state']}",
            "",
            f"- Kind: `{aggregate['kind']}`",
            f"- Reference mode: `{aggregate['reference_mode']}`",
            f"- Num steps: `{aggregate['num_steps']}`",
            f"- Seconds: {format_case_seconds(str(aggregate['kind']), aggregate['seconds'])}",
            f"- Runs: `{aggregate['runs']}`",
            f"- Status counts: `{json.dumps(aggregate['status_counts'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "Aggregate timings:",
            "",
            "| Metric | Min | Median | Max |",
            "| --- | ---: | ---: | ---: |",
        ])
        for key, summary in sorted(aggregate["timings_ms"].items()):
            lines.append(
                f"| `{key}` | {format_ms(float(summary['min']))} | {format_ms(float(summary['median']))} | {format_ms(float(summary['max']))} |"
            )
        if not aggregate["timings_ms"]:
            lines.append("| _(none parsed)_ | | | |")
        if aggregate["wall_seconds"]:
            wall = aggregate["wall_seconds"]
            lines.append(
                f"| `wall_seconds` | {format_seconds(float(wall['min']))} | {format_seconds(float(wall['median']))} | {format_seconds(float(wall['max']))} |"
            )
        if aggregate["max_rss_bytes"]:
            rss = aggregate["max_rss_bytes"]
            lines.append(
                f"| `max_rss_bytes` | {format_bytes(int(rss['min']))} | {format_bytes(int(rss['median']))} | {format_bytes(int(rss['max']))} |"
            )

        lines.extend([
            "",
            "Raw runs:",
            "",
            "| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ])
        matching_results = [
            result
            for result in results
            if result.case_name == aggregate["case_name"]
            and result.phase == aggregate["phase"]
            and result.cache_state == aggregate["cache_state"]
        ]
        for result in matching_results:
            lines.append(
                "| {name} | {status} | {sample_rf} | {decode} | {total} | {wall} | {rss} | `{stdout}` | `{stderr}` |".format(
                    name=result.name,
                    status=result.status,
                    sample_rf=format_ms(result.timings_ms.get("sample_rf")),
                    decode=format_ms(result.timings_ms.get("decode_dacvae") or result.timings_ms.get("decode_latent")),
                    total=format_ms(result.timings_ms.get("total_to_decode")),
                    wall=format_seconds(result.wall_seconds),
                    rss=format_bytes(result.max_rss_bytes),
                    stdout=result.stdout_log or "n/a",
                    stderr=result.stderr_log or "n/a",
                )
            )
        for result in matching_results:
            lines.extend([
                "",
                f"### {result.name}",
                "",
                f"- Output WAV: `{result.output_wav}`",
                f"- CWD: `{result.cwd}`",
                "",
                "Command:",
                "",
                "```bash",
                result.command,
                "```",
            ])
            if result.notes:
                lines.extend(["", "Notes:", ""])
                for note in result.notes:
                    lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def write_json_summary(results: list[BenchmarkResult], path: Path, *, args: argparse.Namespace) -> None:
    payload = {
        "schema_version": 2,
        "invocation": {
            "mode": args.mode,
            "case_label": args.case_label,
            "text": args.text,
            "caption": args.caption,
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup_runs": args.warmup_runs,
            "cache_state": args.cache_state,
            "seconds": args.seconds,
            "seconds_sweep": args.seconds_sweep,
            "omit_seconds": bool(args.omit_seconds),
            "num_steps": args.num_steps,
            "num_steps_sweep": args.num_steps_sweep,
            "reference_wav": args.reference_wav,
            "weights": args.weights,
            "weights_dir": args.weights_dir,
            "weights_repo": args.weights_repo,
            "weights_revision": args.weights_revision,
            "codec_runtime_mode": args.codec_runtime_mode,
        },
        "results": [asdict(result) for result in results],
        "aggregates": build_aggregates(results),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_self_test() -> int:
    sample = """
[timing] sample_rf: 23713.9 ms
[timing] decode_dacvae: 5648.5 ms
[timing] total_to_decode: 1.75 s
122.86 real
1718976512  maximum resident set size
""".strip()
    timings = parse_timing_lines(sample)
    assert timings["sample_rf"] == 23713.9
    assert timings["decode_dacvae"] == 5648.5
    assert timings["total_to_decode"] == 1750.0
    assert parse_wall_seconds(sample) == 122.86
    assert parse_max_rss_bytes(sample) == 1718976512

    results = [
        BenchmarkResult(
            name="mlx-bridge-reference-seconds-5-steps-40-measured-run-01",
            case_name="mlx-bridge-reference-seconds-5-steps-40",
            kind="mlx",
            phase="measured",
            run_index=1,
            overall_run_index=1,
            cache_state="cold",
            reference_mode="reference",
            seconds=5.0,
            num_steps=40,
            command="python scripts/generate_wav.py ...",
            cwd="/tmp/repo",
            output_wav="/tmp/out.wav",
            stdout_log="/tmp/out.stdout.log",
            stderr_log="/tmp/out.stderr.log",
            status="passed",
            timings_ms=timings,
            wall_seconds=122.86,
            max_rss_bytes=1718976512,
        ),
        BenchmarkResult(
            name="mlx-bridge-reference-seconds-5-steps-40-measured-run-02",
            case_name="mlx-bridge-reference-seconds-5-steps-40",
            kind="mlx",
            phase="measured",
            run_index=2,
            overall_run_index=2,
            cache_state="warm",
            reference_mode="reference",
            seconds=5.0,
            num_steps=40,
            command="python scripts/generate_wav.py ...",
            cwd="/tmp/repo",
            output_wav="/tmp/out-2.wav",
            stdout_log="/tmp/out-2.stdout.log",
            stderr_log="/tmp/out-2.stderr.log",
            status="passed",
            timings_ms={"sample_rf": 20000.0, "decode_dacvae": 5000.0, "total_to_decode": 26000.0},
            wall_seconds=100.0,
            max_rss_bytes=1600000000,
        ),
    ]
    aggregates = build_aggregates(results)
    assert any(item["cache_state"] == "cold" for item in aggregates)
    assert any(item["cache_state"] == "warm" for item in aggregates)
    report = build_report(
        results,
        text=DEFAULT_TEXT,
        seed=20260512,
        repeat=2,
        warmup_runs=0,
        cache_state_mode="auto",
    )
    assert "Aggregate results" in report
    assert "sample_rf min/max" in report
    assert "mlx-bridge-reference-seconds-5-steps-40" in report
    print("self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_directory((repo_root / args.output_dir).resolve())
    results: list[BenchmarkResult] = []

    for case in build_cases(args):
        results.extend(run_case(case, args, repo_root, output_dir))

    write_json_summary(results, output_dir / "benchmark-summary.json", args=args)
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_report(
                results,
                text=args.text,
                seed=args.seed,
                repeat=args.repeat,
                warmup_runs=args.warmup_runs,
                cache_state_mode=args.cache_state,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"results": [asdict(result) for result in results], "aggregates": build_aggregates(results)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
