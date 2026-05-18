#!/usr/bin/env python3
"""Benchmark persistent MLX batch generation via generate_wav --requests-json."""

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
DEFAULT_CFG_SCALE_CAPTION = 3.0
DEFAULT_CFG_SCALE_SPEAKER = 5.0
WALL_RE = re.compile(r"(?:^|\s)([0-9]+(?:\.[0-9]+)?)\s+real(?:\s|$)")
RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$")


class PersistentBatchBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: str
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class RequestResult:
    index: int
    phase: str
    output_wav: str
    text: str
    seed: int | None
    num_steps: int | None
    cfg_guidance_mode: str | None
    cfg_scale_text: float | None
    output_duration_seconds: float | None
    timings_ms: dict[str, float]
    codec_encode_backend: str | None
    codec_decode_backend: str | None


@dataclass(frozen=True)
class BatchRunResult:
    command: str
    cwd: str
    request_count: int
    warmup_requests: int
    measured_requests: int
    metadata_json: str
    requests_json: str
    stdout_log: str
    stderr_log: str
    status: str
    wall_seconds: float | None
    max_rss_bytes: int | None
    process_setup_overhead_ms: float | None
    requests: tuple[RequestResult, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark persistent MLX batch generation.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", default="benchmark-runs/persistent-batch")
    parser.add_argument("--report")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--caption")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--omit-seconds", action="store_true")
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--cfg-guidance-mode", choices=("independent", "joint", "reduced"), default="independent")
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float)
    parser.add_argument("--cfg-scale-speaker", type=float)
    parser.add_argument("--reference-wav")
    parser.add_argument("--upstream-root")
    parser.add_argument("--mlx-python", default="python3")
    parser.add_argument("--weights")
    parser.add_argument("--weights-dir")
    parser.add_argument("--weights-repo")
    parser.add_argument("--weights-revision")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument(
        "--codec-runtime-mode",
        choices=("mlx",),
        default="mlx",
    )
    parser.add_argument("--codec-path")
    parser.add_argument("--codec-artifact-dir")
    parser.add_argument("--codec-artifact-repo")
    parser.add_argument("--codec-artifact-revision")
    parser.add_argument("--model-config-json")
    parser.add_argument("--text-tokenizer-repo")
    parser.add_argument("--caption-tokenizer-repo")
    parser.add_argument("--case-label", default="persistent-batch")
    parser.add_argument(
        "--cleanup-between-requests",
        action="store_true",
        help="Forward generate_wav --cleanup-between-requests to test explicit MLX cache cleanup in one persistent process.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def slug_token(value: str) -> str:
    token = value.strip().lower().replace(" ", "-").replace(".", "p")
    return re.sub(r"[^a-z0-9\-]+", "-", token).strip("-")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def validate_args(args: argparse.Namespace) -> None:
    if args.requests < 1:
        raise PersistentBatchBenchmarkError("--requests must be >= 1")
    if args.warmup_requests < 0:
        raise PersistentBatchBenchmarkError("--warmup-requests must be >= 0")
    if args.num_steps <= 0:
        raise PersistentBatchBenchmarkError("--num-steps must be > 0")
    if not args.omit_seconds and float(args.seconds) <= 0:
        raise PersistentBatchBenchmarkError("--seconds must be > 0 unless --omit-seconds is used")
    if sum(1 for value in (args.weights, args.weights_dir, args.weights_repo) if value) != 1:
        raise PersistentBatchBenchmarkError("choose exactly one of --weights, --weights-dir, or --weights-repo")
    if sum(1 for value in (args.codec_path, args.codec_artifact_dir, args.codec_artifact_repo) if value) > 1:
        raise PersistentBatchBenchmarkError("choose at most one of --codec-path, --codec-artifact-dir, or --codec-artifact-repo")
    if args.weights_revision and not args.weights_repo:
        raise PersistentBatchBenchmarkError("--weights-revision requires --weights-repo")
    if args.codec_artifact_revision and not args.codec_artifact_repo:
        raise PersistentBatchBenchmarkError("--codec-artifact-revision requires --codec-artifact-repo")


def build_request_overrides(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    total = int(args.warmup_requests) + int(args.requests)
    slug = slug_token(args.case_label or "persistent-batch")
    overrides: list[dict[str, Any]] = []
    for index in range(1, total + 1):
        item: dict[str, Any] = {
            "text": args.text,
            "output": str(output_dir / f"{slug}.request-{index:02d}.wav"),
            "num_steps": int(args.num_steps),
            "cfg_guidance_mode": args.cfg_guidance_mode,
            "cfg_scale_text": float(args.cfg_scale_text),
            "seed": int(args.seed) + index - 1,
        }
        if args.caption:
            item["caption"] = args.caption
            if args.cfg_scale_caption is not None:
                item["cfg_scale_caption"] = float(args.cfg_scale_caption)
        else:
            item["cfg_scale_caption"] = 0.0 if args.cfg_scale_caption is None else float(args.cfg_scale_caption)
        if args.reference_wav:
            item["reference_wav"] = args.reference_wav
            if args.cfg_scale_speaker is not None:
                item["cfg_scale_speaker"] = float(args.cfg_scale_speaker)
        else:
            item["no_reference"] = True
            item["cfg_scale_speaker"] = 0.0 if args.cfg_scale_speaker is None else float(args.cfg_scale_speaker)
        if not args.omit_seconds:
            item["seconds"] = float(args.seconds)
        overrides.append(item)
    return overrides


def effective_cfg_scale_caption(args: argparse.Namespace) -> float:
    if args.cfg_scale_caption is not None:
        return float(args.cfg_scale_caption)
    return DEFAULT_CFG_SCALE_CAPTION if args.caption else 0.0


def effective_cfg_scale_speaker(args: argparse.Namespace) -> float:
    if args.cfg_scale_speaker is not None:
        return float(args.cfg_scale_speaker)
    return DEFAULT_CFG_SCALE_SPEAKER if args.reference_wav else 0.0


def build_command(args: argparse.Namespace, requests_json: Path, metadata_json: Path) -> tuple[list[str], dict[str, str]]:
    weight_flag, weight_value = next(
        (flag, value)
        for flag, value in (("--weights", args.weights), ("--weights-dir", args.weights_dir), ("--weights-repo", args.weights_repo))
        if value
    )
    argv = [
        TIME_L_BIN,
        "-l",
        args.mlx_python,
        "scripts/generate_wav.py",
        weight_flag,
        str(weight_value),
        "--requests-json",
        str(requests_json),
        "--metadata-json",
        str(metadata_json),
        "--json",
        "--codec-repo",
        args.codec_repo,
        "--codec-device",
        args.codec_device,
        "--codec-runtime-mode",
        args.codec_runtime_mode,
    ]
    if weight_flag == "--weights-repo" and args.weights_revision:
        argv.extend(["--weights-revision", args.weights_revision])
    for flag, value in (
        ("--codec-path", args.codec_path),
        ("--codec-artifact-dir", args.codec_artifact_dir),
        ("--codec-artifact-repo", args.codec_artifact_repo),
        ("--codec-artifact-revision", args.codec_artifact_revision),
        ("--text-tokenizer-repo", args.text_tokenizer_repo),
        ("--caption-tokenizer-repo", args.caption_tokenizer_repo),
    ):
        if value:
            argv.extend([flag, value])
    if args.model_config_json:
        if weight_flag != "--weights":
            raise PersistentBatchBenchmarkError("--model-config-json is only valid with --weights")
        argv.extend(["--model-config-json", args.model_config_json])
    if args.cleanup_between_requests:
        argv.append("--cleanup-between-requests")
    env = os.environ.copy()
    if args.upstream_root:
        parts = [str(Path(args.upstream_root).resolve())]
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return argv, env


def run_command(argv: list[str], *, cwd: Path, env: dict[str, str] | None) -> CommandResult:
    completed = subprocess.run(argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return CommandResult(argv=list(argv), cwd=str(cwd), stdout=completed.stdout, stderr=completed.stderr, returncode=int(completed.returncode))


def write_logs(base_path: Path, result: CommandResult) -> tuple[str, str]:
    stdout_path = base_path.with_suffix(".stdout.log")
    stderr_path = base_path.with_suffix(".stderr.log")
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def result_payloads(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if "results" in metadata:
        if not isinstance(metadata["results"], list):
            raise PersistentBatchBenchmarkError("metadata results field is not a list")
        return metadata["results"]
    if "result" in metadata:
        return [metadata]
    raise PersistentBatchBenchmarkError("metadata does not contain result payloads")


def parse_batch_metadata(metadata: dict[str, Any], *, warmup_requests: int) -> tuple[RequestResult, ...]:
    parsed: list[RequestResult] = []
    for default_index, payload in enumerate(result_payloads(metadata), start=1):
        result = payload.get("result", {})
        request = payload.get("request", {})
        batch = payload.get("batch", {})
        index = int(batch.get("index") or default_index)
        timings = result.get("timings_ms") or {}
        if not isinstance(timings, dict):
            raise PersistentBatchBenchmarkError(f"request #{index} timings_ms is not an object")
        parsed.append(
            RequestResult(
                index=index,
                phase="warmup" if index <= warmup_requests else "measured",
                output_wav=str(result.get("output_wav") or ""),
                text=str(request.get("text") or ""),
                seed=request.get("seed"),
                num_steps=request.get("num_steps"),
                cfg_guidance_mode=request.get("cfg_guidance_mode"),
                cfg_scale_text=None if request.get("cfg_scale_text") is None else float(request["cfg_scale_text"]),
                output_duration_seconds=None if result.get("resolved_seconds") is None else float(result["resolved_seconds"]),
                timings_ms={str(key): float(value) for key, value in timings.items()},
                codec_encode_backend=result.get("codec_encode_backend"),
                codec_decode_backend=result.get("codec_decode_backend"),
            )
        )
    return tuple(sorted(parsed, key=lambda item: item.index))


def process_setup_overhead_ms(wall_seconds: float | None, requests: tuple[RequestResult, ...]) -> float | None:
    if wall_seconds is None:
        return None
    total_to_decode = sum(item.timings_ms.get("total_to_decode", 0.0) for item in requests)
    return max(0.0, wall_seconds * 1000.0 - total_to_decode)


def run_batch(args: argparse.Namespace, repo_root: Path, output_dir: Path) -> BatchRunResult:
    validate_args(args)
    output_dir = ensure_directory(output_dir)
    requests_json = output_dir / "requests.json"
    metadata_json = output_dir / "metadata.json"
    request_overrides = build_request_overrides(args, output_dir)
    requests_json.write_text(json.dumps(request_overrides, ensure_ascii=False, indent=2), encoding="utf-8")
    argv, env = build_command(args, requests_json, metadata_json)
    if args.dry_run:
        return BatchRunResult(
            command=shell_join(argv),
            cwd=str(repo_root),
            request_count=len(request_overrides),
            warmup_requests=int(args.warmup_requests),
            measured_requests=int(args.requests),
            metadata_json=str(metadata_json),
            requests_json=str(requests_json),
            stdout_log="",
            stderr_log="",
            status="dry-run",
            wall_seconds=None,
            max_rss_bytes=None,
            process_setup_overhead_ms=None,
            requests=(),
        )

    command_result = run_command(argv, cwd=repo_root, env=env)
    stdout_log, stderr_log = write_logs(output_dir / "persistent-batch", command_result)
    if command_result.returncode != 0:
        raise PersistentBatchBenchmarkError(command_result.stderr.strip() or command_result.stdout.strip() or "persistent batch benchmark failed")
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    requests = parse_batch_metadata(metadata, warmup_requests=int(args.warmup_requests))
    wall_seconds = parse_wall_seconds(command_result.stderr)
    return BatchRunResult(
        command=shell_join(command_result.argv),
        cwd=command_result.cwd,
        request_count=len(requests),
        warmup_requests=int(args.warmup_requests),
        measured_requests=sum(1 for item in requests if item.phase == "measured"),
        metadata_json=str(metadata_json),
        requests_json=str(requests_json),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        status="passed",
        wall_seconds=wall_seconds,
        max_rss_bytes=parse_max_rss_bytes(command_result.stderr),
        process_setup_overhead_ms=process_setup_overhead_ms(wall_seconds, requests),
        requests=requests,
    )


def summarize_numeric(values: list[float | int]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(ordered), "min": ordered[0], "median": statistics.median(ordered), "max": ordered[-1]}


def format_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.1f} ms"


def format_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.2f} s"


def format_bytes(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value} bytes ({value / float(1024**3):.2f} GiB)"


def request_metric_summary(result: BatchRunResult, metric: str, *, phase: str = "measured") -> dict[str, float | int] | None:
    return summarize_numeric([item.timings_ms[metric] for item in result.requests if item.phase == phase and metric in item.timings_ms])


DECODE_SUBPHASE_LABELS = (
    ("decode_dacvae_model_compute", "model compute/schedule"),
    ("decode_dacvae_materialization", "materialization/sync"),
    ("decode_dacvae_host_transfer", "host transfer"),
    ("decode_dacvae_postprocess", "postprocess"),
    ("decode_dacvae_wav_serialization", "WAV serialization"),
    ("decode_dacvae_cleanup", "cleanup"),
)


def build_json_summary(result: BatchRunResult, *, args: argparse.Namespace) -> dict[str, Any]:
    measured_sum_ms = sum(item.timings_ms.get("total_to_decode", 0.0) for item in result.requests if item.phase == "measured")
    return {
        "schema_version": 1,
        "invocation": {
            "case_label": args.case_label,
            "text": args.text,
            "caption": args.caption,
            "seed": args.seed,
            "requests": args.requests,
            "warmup_requests": args.warmup_requests,
            "seconds": args.seconds,
            "omit_seconds": bool(args.omit_seconds),
            "num_steps": args.num_steps,
            "cfg_guidance_mode": args.cfg_guidance_mode,
            "cfg_scale_text": args.cfg_scale_text,
            "cfg_scale_caption": effective_cfg_scale_caption(args),
            "cfg_scale_speaker": effective_cfg_scale_speaker(args),
            "reference_wav": args.reference_wav,
            "weights": args.weights,
            "weights_dir": args.weights_dir,
            "weights_repo": args.weights_repo,
            "weights_revision": args.weights_revision,
            "codec_runtime_mode": args.codec_runtime_mode,
            "codec_path": args.codec_path,
            "codec_artifact_dir": args.codec_artifact_dir,
            "codec_artifact_repo": args.codec_artifact_repo,
            "codec_artifact_revision": args.codec_artifact_revision,
            "cleanup_between_requests": bool(args.cleanup_between_requests),
        },
        "process": {
            "status": result.status,
            "command": result.command,
            "cwd": result.cwd,
            "wall_seconds": result.wall_seconds,
            "max_rss_bytes": result.max_rss_bytes,
            "process_setup_overhead_ms": result.process_setup_overhead_ms,
            "request_count": result.request_count,
            "warmup_requests": result.warmup_requests,
            "measured_requests": result.measured_requests,
            "measured_total_to_decode_sum_ms": measured_sum_ms,
            "measured_generation_throughput_rps": None if measured_sum_ms <= 0 else float(result.measured_requests) / (measured_sum_ms / 1000.0),
            "process_throughput_rps": None if not result.wall_seconds else float(result.request_count) / float(result.wall_seconds),
        },
        "aggregates": {
            "measured_total_to_decode_ms": request_metric_summary(result, "total_to_decode"),
            "measured_sample_rf_ms": request_metric_summary(result, "sample_rf"),
            "measured_decode_dacvae_ms": request_metric_summary(result, "decode_dacvae"),
            "measured_decode_dacvae_model_ms": request_metric_summary(result, "decode_dacvae_model"),
            "measured_audio_write_ms": request_metric_summary(result, "audio_write_wav"),
            "measured_encode_dacvae_ms": request_metric_summary(result, "encode_dacvae"),
            "measured_output_duration_seconds": summarize_numeric([
                item.output_duration_seconds
                for item in result.requests
                if item.phase == "measured" and item.output_duration_seconds is not None
            ]),
        },
        "requests": [asdict(item) for item in result.requests],
    }


def build_report(result: BatchRunResult, *, args: argparse.Namespace) -> str:
    summary = build_json_summary(result, args=args)
    process = summary["process"]
    aggregates = summary["aggregates"]
    total = aggregates["measured_total_to_decode_ms"] or {}
    sample = aggregates["measured_sample_rf_ms"] or {}
    decode = aggregates["measured_decode_dacvae_ms"] or {}
    decode_model = aggregates["measured_decode_dacvae_model_ms"] or {}
    audio_write = aggregates["measured_audio_write_ms"] or {}
    encode = aggregates["measured_encode_dacvae_ms"] or {}
    output_duration = aggregates["measured_output_duration_seconds"] or {}
    first = result.requests[0] if result.requests else None

    def metric_range(item: dict[str, float | int]) -> str:
        return "" if not item else f"{format_ms(float(item['min']))} / {format_ms(float(item['max']))}"

    lines = [
        "# Persistent Batch Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Case label: {args.case_label}",
        f"- Requests: {result.request_count} ({result.warmup_requests} warmup, {result.measured_requests} measured)",
        f"- Prompt text: {args.text}",
        f"- Seed start: {args.seed}",
        f"- Num steps: {args.num_steps}",
        f"- CFG guidance mode: {args.cfg_guidance_mode}",
        f"- CFG text scale: {args.cfg_scale_text}",
        f"- CFG caption scale: {effective_cfg_scale_caption(args)}",
        f"- CFG speaker scale: {effective_cfg_scale_speaker(args)}",
        f"- Seconds: {'predicted duration (--seconds omitted)' if args.omit_seconds else args.seconds}",
        f"- Codec runtime mode: {args.codec_runtime_mode}",
        f"- Cleanup between requests: {bool(args.cleanup_between_requests)}",
        "",
        "## Process results",
        "",
        "| Status | Wall | Setup/load overhead | Max RSS | Process throughput | Measured generation throughput |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| {status} | {wall} | {setup} | {rss} | {process_rps} req/s | {measured_rps} req/s |".format(
            status=result.status,
            wall=format_seconds(result.wall_seconds),
            setup=format_ms(result.process_setup_overhead_ms),
            rss=format_bytes(result.max_rss_bytes),
            process_rps="" if process["process_throughput_rps"] is None else f"{process['process_throughput_rps']:.3f}",
            measured_rps="" if process["measured_generation_throughput_rps"] is None else f"{process['measured_generation_throughput_rps']:.3f}",
        ),
        "",
        "## Request timing aggregates",
        "",
        "| Scope | first request | measured median | measured min/max |",
        "| --- | ---: | ---: | --- |",
        f"| total_to_decode | {format_ms(first.timings_ms.get('total_to_decode') if first else None)} | {format_ms(float(total['median'])) if total else ''} | {metric_range(total)} |",
        f"| sample_rf | {format_ms(first.timings_ms.get('sample_rf') if first else None)} | {format_ms(float(sample['median'])) if sample else ''} | {metric_range(sample)} |",
        f"| encode_dacvae | {format_ms(first.timings_ms.get('encode_dacvae') if first else None)} | {format_ms(float(encode['median'])) if encode else ''} | {metric_range(encode)} |",
        f"| decode_dacvae | {format_ms(first.timings_ms.get('decode_dacvae') if first else None)} | {format_ms(float(decode['median'])) if decode else ''} | {metric_range(decode)} |",
        f"| decode_dacvae_model | {format_ms(first.timings_ms.get('decode_dacvae_model') if first else None)} | {format_ms(float(decode_model['median'])) if decode_model else ''} | {metric_range(decode_model)} |",
        f"| audio_write_wav | {format_ms(first.timings_ms.get('audio_write_wav') if first else None)} | {format_ms(float(audio_write['median'])) if audio_write else ''} | {metric_range(audio_write)} |",
        f"| output_duration_seconds | {format_seconds(first.output_duration_seconds if first else None)} | {format_seconds(float(output_duration['median'])) if output_duration else ''} | {'' if not output_duration else format_seconds(float(output_duration['min'])) + ' / ' + format_seconds(float(output_duration['max']))} |",
        "",
    ]
    decode_subphase_rows = []
    for key, label in DECODE_SUBPHASE_LABELS:
        subphase = request_metric_summary(result, key)
        if subphase:
            decode_subphase_rows.append(
                f"| {label} | {format_ms(first.timings_ms.get(key) if first else None)} | "
                f"{format_ms(float(subphase['median']))} | {metric_range(subphase)} |"
            )
    if decode_subphase_rows:
        lines.extend(
            [
                "## MLX decode subphase aggregates",
                "",
                "These rows are emitted by the MLX codec bridge when codec runtime mode uses MLX decode.",
                "",
                "| Scope | first request | measured median | measured min/max |",
                "| --- | ---: | ---: | --- |",
                *decode_subphase_rows,
                "",
            ]
        )
    lines.extend(
        [
            "## Raw requests",
            "",
            "| # | Phase | Seed | Steps | CFG mode | CFG text | Duration | total_to_decode | sample_rf | encode_dacvae | decode_dacvae | decode_model | audio_write | Output |",
            "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in result.requests:
        lines.append(
            "| {index} | {phase} | {seed} | {steps} | {cfg_mode} | {cfg_text} | {duration} | {total} | {sample} | {encode} | {decode} | {decode_model} | {audio_write} | {output} |".format(
                index=item.index,
                phase=item.phase,
                seed="" if item.seed is None else item.seed,
                steps="" if item.num_steps is None else item.num_steps,
                cfg_mode=item.cfg_guidance_mode or "",
                cfg_text="" if item.cfg_scale_text is None else item.cfg_scale_text,
                duration=format_seconds(item.output_duration_seconds),
                total=format_ms(item.timings_ms.get("total_to_decode")),
                sample=format_ms(item.timings_ms.get("sample_rf")),
                encode=format_ms(item.timings_ms.get("encode_dacvae")),
                decode=format_ms(item.timings_ms.get("decode_dacvae")),
                decode_model=format_ms(item.timings_ms.get("decode_dacvae_model")),
                audio_write=format_ms(item.timings_ms.get("audio_write_wav")),
                output=item.output_wav,
            )
        )
    lines.extend([
        "",
        "Command:",
        "",
        "    " + result.command,
        "",
        f"- Requests JSON: {result.requests_json}",
        f"- Metadata JSON: {result.metadata_json}",
        f"- stdout log: {result.stdout_log or 'n/a'}",
        f"- stderr log: {result.stderr_log or 'n/a'}",
    ])
    return "\n".join(lines) + "\n"


def run_self_test() -> int:
    metadata = {
        "results": [
            {"result": {"output_wav": "/tmp/one.wav", "codec_encode_backend": "not-required", "codec_decode_backend": "mlx", "timings_ms": {"sample_rf": 100.0, "decode_dacvae": 20.0, "total_to_decode": 150.0}}, "request": {"text": "one", "seed": 1, "num_steps": 12}, "batch": {"index": 1, "count": 3}},
            {"result": {"output_wav": "/tmp/two.wav", "codec_encode_backend": "not-required", "codec_decode_backend": "mlx", "timings_ms": {"sample_rf": 90.0, "decode_dacvae": 18.0, "total_to_decode": 130.0}}, "request": {"text": "two", "seed": 2, "num_steps": 12}, "batch": {"index": 2, "count": 3}},
            {"result": {"output_wav": "/tmp/three.wav", "codec_encode_backend": "not-required", "codec_decode_backend": "mlx", "timings_ms": {"sample_rf": 95.0, "decode_dacvae": 19.0, "total_to_decode": 140.0}}, "request": {"text": "three", "seed": 3, "num_steps": 12}, "batch": {"index": 3, "count": 3}},
        ]
    }
    parsed = parse_batch_metadata(metadata, warmup_requests=1)
    assert [item.phase for item in parsed] == ["warmup", "measured", "measured"]
    assert summarize_numeric([item.timings_ms["total_to_decode"] for item in parsed if item.phase == "measured"])["median"] == 135.0
    result = BatchRunResult(
        command="python scripts/generate_wav.py ...",
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
        max_rss_bytes=1000,
        process_setup_overhead_ms=process_setup_overhead_ms(2.0, parsed),
        requests=parsed,
    )
    args = argparse.Namespace(case_label="self-test", text=DEFAULT_TEXT, caption=None, seed=1, requests=2, warmup_requests=1, seconds=5.0, omit_seconds=False, num_steps=12, cfg_guidance_mode="independent", cfg_scale_text=3.0, cfg_scale_caption=None, cfg_scale_speaker=None, reference_wav=None, weights=None, weights_dir=None, weights_repo="repo", weights_revision=None, codec_runtime_mode="mlx", codec_path=None, codec_artifact_dir=None, codec_artifact_repo="codec", codec_artifact_revision=None, cleanup_between_requests=False)
    report = build_report(result, args=args)
    assert "Persistent Batch Benchmark Report" in report
    assert "Measured generation throughput" in report
    assert "135.0 ms" in report
    print("self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_directory((repo_root / args.output_dir).resolve())
    result = run_batch(args, repo_root, output_dir)
    summary = build_json_summary(result, args=args)
    (output_dir / "persistent-batch-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(result, args=args), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PersistentBatchBenchmarkError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc
