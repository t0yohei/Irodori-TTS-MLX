#!/usr/bin/env python3
"""Benchmark request latency against one long-lived local generation worker."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEXT = "今日はいい天気ですね。"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


class PersistentServingBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class RequestResult:
    index: int
    phase: str
    output_wav: str
    text: str
    seed: int | None
    num_steps: int | None
    persistent_request_latency_ms: float
    worker_generate_latency_ms: float | None
    worker_json_serialization_ms: float | None
    timings_ms: dict[str, float]
    codec_encode_backend: str | None
    codec_decode_backend: str | None


@dataclass(frozen=True)
class ServingRunResult:
    command: str
    cwd: str
    request_count: int
    warmup_requests: int
    measured_requests: int
    requests_json: str
    stderr_log: str
    status: str
    worker_startup_ms: float | None
    max_rss_bytes: int | None
    requests: tuple[RequestResult, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark persistent local serving request latency.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="benchmark-runs/persistent-serving")
    parser.add_argument("--report")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--caption")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--omit-seconds", action="store_true")
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--reference-wav")
    parser.add_argument("--upstream-root")
    parser.add_argument("--mlx-python", default=sys.executable)
    parser.add_argument("--weights")
    parser.add_argument("--weights-dir")
    parser.add_argument("--weights-repo")
    parser.add_argument("--weights-revision")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument(
        "--codec-runtime-mode",
        choices=("persistent", "subprocess", "mlx", "mlx-decode", "mlx-decode-subprocess"),
        default="persistent",
    )
    parser.add_argument("--codec-path")
    parser.add_argument("--codec-artifact-dir")
    parser.add_argument("--codec-artifact-repo")
    parser.add_argument("--codec-artifact-revision")
    parser.add_argument("--model-config-json")
    parser.add_argument("--text-tokenizer-repo")
    parser.add_argument("--caption-tokenizer-repo")
    parser.add_argument("--case-label", default="persistent-serving")
    parser.add_argument("--cleanup-between-requests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug_token(value: str) -> str:
    import re

    token = value.strip().lower().replace(" ", "-").replace(".", "p")
    return re.sub(r"[^a-z0-9\-]+", "-", token).strip("-")


def validate_args(args: argparse.Namespace) -> None:
    if args.requests < 1:
        raise PersistentServingBenchmarkError("--requests must be >= 1")
    if args.warmup_requests < 0:
        raise PersistentServingBenchmarkError("--warmup-requests must be >= 0")
    if args.num_steps <= 0:
        raise PersistentServingBenchmarkError("--num-steps must be > 0")
    if not args.omit_seconds and float(args.seconds) <= 0:
        raise PersistentServingBenchmarkError("--seconds must be > 0 unless --omit-seconds is used")
    if sum(1 for value in (args.weights, args.weights_dir, args.weights_repo) if value) != 1:
        raise PersistentServingBenchmarkError("choose exactly one of --weights, --weights-dir, or --weights-repo")
    if sum(1 for value in (args.codec_path, args.codec_artifact_dir, args.codec_artifact_repo) if value) > 1:
        raise PersistentServingBenchmarkError("choose at most one of --codec-path, --codec-artifact-dir, or --codec-artifact-repo")
    if args.weights_revision and not args.weights_repo:
        raise PersistentServingBenchmarkError("--weights-revision requires --weights-repo")
    if args.codec_artifact_revision and not args.codec_artifact_repo:
        raise PersistentServingBenchmarkError("--codec-artifact-revision requires --codec-artifact-repo")


def build_request_overrides(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    total = int(args.warmup_requests) + int(args.requests)
    slug = slug_token(args.case_label or "persistent-serving")
    requests: list[dict[str, Any]] = []
    for index in range(1, total + 1):
        item: dict[str, Any] = {
            "text": args.text,
            "output": str(output_dir / f"{slug}.request-{index:02d}.wav"),
            "num_steps": int(args.num_steps),
            "seed": int(args.seed) + index - 1,
        }
        if args.caption:
            item["caption"] = args.caption
        if args.reference_wav:
            item["reference_wav"] = args.reference_wav
        else:
            item["no_reference"] = True
        if not args.omit_seconds:
            item["seconds"] = float(args.seconds)
        requests.append(item)
    return requests


def build_worker_command(args: argparse.Namespace, repo_root: Path) -> tuple[list[str], dict[str, str]]:
    argv = [args.mlx_python, str(Path(__file__).relative_to(repo_root)), "--worker"]
    for flag, value in (
        ("--weights", args.weights),
        ("--weights-dir", args.weights_dir),
        ("--weights-repo", args.weights_repo),
        ("--weights-revision", args.weights_revision),
        ("--codec-device", args.codec_device),
        ("--codec-repo", args.codec_repo),
        ("--codec-runtime-mode", args.codec_runtime_mode),
        ("--codec-path", args.codec_path),
        ("--codec-artifact-dir", args.codec_artifact_dir),
        ("--codec-artifact-repo", args.codec_artifact_repo),
        ("--codec-artifact-revision", args.codec_artifact_revision),
        ("--model-config-json", args.model_config_json),
        ("--text-tokenizer-repo", args.text_tokenizer_repo),
        ("--caption-tokenizer-repo", args.caption_tokenizer_repo),
    ):
        if value:
            argv.extend([flag, str(value)])
    if args.cleanup_between_requests:
        argv.append("--cleanup-between-requests")
    env = os.environ.copy()
    if args.upstream_root:
        parts = [str(Path(args.upstream_root).resolve())]
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return argv, env


def _generate_wav_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--requests-json", "[]", "--no-json"]
    for flag, value in (
        ("--weights", args.weights),
        ("--weights-dir", args.weights_dir),
        ("--weights-repo", args.weights_repo),
        ("--weights-revision", args.weights_revision),
        ("--codec-device", args.codec_device),
        ("--codec-repo", args.codec_repo),
        ("--codec-runtime-mode", args.codec_runtime_mode),
        ("--codec-path", args.codec_path),
        ("--codec-artifact-dir", args.codec_artifact_dir),
        ("--codec-artifact-repo", args.codec_artifact_repo),
        ("--codec-artifact-revision", args.codec_artifact_revision),
        ("--model-config-json", args.model_config_json),
        ("--text-tokenizer-repo", args.text_tokenizer_repo),
        ("--caption-tokenizer-repo", args.caption_tokenizer_repo),
    ):
        if value:
            argv.extend([flag, str(value)])
    if args.cleanup_between_requests:
        argv.append("--cleanup-between-requests")
    return argv


def validate_worker_request(
    gen: Any,
    *,
    model_config: Any,
    gen_args: argparse.Namespace,
    layout_runtime: dict[str, Any] | None,
    overrides: dict[str, Any],
    index: int,
) -> None:
    request_reference = overrides.get("reference_wav", gen_args.reference_wav)
    request_no_reference = bool(overrides.get("no_reference", gen_args.no_reference))
    request_caption = overrides.get("caption", gen_args.caption)
    if layout_runtime is not None:
        if layout_runtime.get("requires_reference_audio") and not request_reference:
            raise SystemExit(f"error: generation request #{index}: selected weights layout requires reference_wav")
        if not layout_runtime.get("supports_no_reference", False) and request_no_reference:
            raise SystemExit(f"error: generation request #{index}: selected weights layout does not support no_reference")
        if (
            "supports_caption" in layout_runtime
            and not layout_runtime.get("supports_caption", False)
            and request_caption is not None
            and str(request_caption).strip()
        ):
            raise SystemExit(f"error: generation request #{index}: selected weights layout does not support caption conditioning")
    gen.validate_checkpoint_family_request(
        model_config=model_config,
        args=gen_args,
        overrides=overrides,
        index=index,
    )


def run_worker(args: argparse.Namespace) -> int:
    import scripts.generate_wav as gen

    gen_args = gen.parse_args(_generate_wav_argv(args))
    gen._ensure_runtime_imports()
    layout = gen.resolve_weights_layout_source(
        weights_dir=gen_args.weights_dir,
        weights_repo=gen_args.weights_repo,
        revision=gen_args.weights_revision,
    )
    if layout is not None:
        gen_args.weights = str(layout.weights_path)
        model_config = layout.model_config
        layout_runtime = dict(layout.manifest.get("runtime", {}))
    else:
        model_config = gen.load_model_config_json(gen_args.model_config_json)
        layout_runtime = None
    gen.resolve_codec_artifact_args(gen_args)
    runtime = gen.MLXDACVAERuntime(config=gen.build_runtime_config(gen_args, model_config))
    print(json.dumps({"type": "ready", "boundaries": runtime.describe_boundaries()}, ensure_ascii=False), flush=True)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if payload.get("type") == "shutdown":
            print(json.dumps({"type": "shutdown"}), flush=True)
            return 0
        if payload.get("type") != "request":
            print(json.dumps({"type": "error", "error": "unknown message type"}), flush=True)
            continue
        started = time.perf_counter()
        overrides = payload.get("request") or {}
        validate_worker_request(
            gen,
            model_config=model_config,
            gen_args=gen_args,
            layout_runtime=layout_runtime,
            overrides=overrides,
            index=int(payload.get("index") or 1),
        )
        request = gen.build_generation_request(gen_args, overrides)
        result = runtime.generate(request)
        generate_latency_ms = (time.perf_counter() - started) * 1000.0
        response = gen.build_result_payload(result=result, request=request, runtime=runtime, args=gen_args)
        response["type"] = "result"
        response["worker"] = {"generate_latency_ms": generate_latency_ms, "json_serialization_ms": None}
        started = time.perf_counter()
        json.dumps(response, ensure_ascii=False, sort_keys=True, default=str)
        response["worker"]["json_serialization_ms"] = (time.perf_counter() - started) * 1000.0
        body = json.dumps(response, ensure_ascii=False, sort_keys=True, default=str)
        print(body, flush=True)
        if args.cleanup_between_requests:
            gen.release_mlx_runtime_memory()
    return 0


def _ru_maxrss_value_to_bytes(value: int, *, platform: str = sys.platform) -> int | None:
    if not value:
        return None
    if platform == "darwin":
        return int(value)
    return int(value) * 1024


def _child_max_rss_bytes() -> int | None:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    value = int(getattr(usage, "ru_maxrss", 0) or 0)
    return _ru_maxrss_value_to_bytes(value)


def parse_response(payload: dict[str, Any], *, index: int, phase: str, latency_ms: float) -> RequestResult:
    result = payload.get("result") or {}
    request = payload.get("request") or {}
    timings = result.get("timings_ms") or {}
    worker = payload.get("worker") or {}
    return RequestResult(
        index=index,
        phase=phase,
        output_wav=str(result.get("output_wav") or ""),
        text=str(request.get("text") or ""),
        seed=request.get("seed"),
        num_steps=request.get("num_steps"),
        persistent_request_latency_ms=float(latency_ms),
        worker_generate_latency_ms=None if worker.get("generate_latency_ms") is None else float(worker["generate_latency_ms"]),
        worker_json_serialization_ms=None
        if worker.get("json_serialization_ms") is None
        else float(worker["json_serialization_ms"]),
        timings_ms={str(key): float(value) for key, value in timings.items()},
        codec_encode_backend=result.get("codec_encode_backend"),
        codec_decode_backend=result.get("codec_decode_backend"),
    )


def run_serving(args: argparse.Namespace, repo_root: Path, output_dir: Path) -> ServingRunResult:
    validate_args(args)
    output_dir = ensure_directory(output_dir)
    requests = build_request_overrides(args, output_dir)
    requests_json = output_dir / "requests.json"
    requests_json.write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    argv, env = build_worker_command(args, repo_root)
    if args.dry_run:
        return ServingRunResult(
            command=shell_join(argv),
            cwd=str(repo_root),
            request_count=len(requests),
            warmup_requests=int(args.warmup_requests),
            measured_requests=int(args.requests),
            requests_json=str(requests_json),
            stderr_log="",
            status="dry-run",
            worker_startup_ms=None,
            max_rss_bytes=None,
            requests=(),
        )
    stderr_log = output_dir / "worker.stderr.log"
    started = time.perf_counter()
    stderr_fh = stderr_log.open("w", encoding="utf-8")
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(repo_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None and proc.stdout is not None
        ready_line = proc.stdout.readline()
        worker_startup_ms = (time.perf_counter() - started) * 1000.0
        if not ready_line:
            proc.wait(timeout=5)
            stderr_fh.flush()
            stderr = stderr_log.read_text(encoding="utf-8")
            raise PersistentServingBenchmarkError(stderr.strip() or "worker exited before ready")
        ready = json.loads(ready_line)
        if ready.get("type") != "ready":
            raise PersistentServingBenchmarkError(f"worker did not send ready: {ready_line.strip()}")
        parsed: list[RequestResult] = []
        for index, overrides in enumerate(requests, start=1):
            phase = "warmup" if index <= int(args.warmup_requests) else "measured"
            line = json.dumps({"type": "request", "index": index, "request": overrides}, ensure_ascii=False)
            started = time.perf_counter()
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
            response_line = proc.stdout.readline()
            latency_ms = (time.perf_counter() - started) * 1000.0
            if not response_line:
                raise PersistentServingBenchmarkError("worker exited before request response")
            response = json.loads(response_line)
            if response.get("type") != "result":
                raise PersistentServingBenchmarkError(str(response.get("error") or response))
            parsed.append(parse_response(response, index=index, phase=phase, latency_ms=latency_ms))
        proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
        proc.stdin.flush()
    finally:
        if proc is not None:
            try:
                proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        stderr_fh.close()
    if proc is None:
        raise PersistentServingBenchmarkError("worker process did not start")
    if proc.returncode != 0:
        stderr = stderr_log.read_text(encoding="utf-8")
        raise PersistentServingBenchmarkError(stderr.strip() or f"worker exited with {proc.returncode}")
    return ServingRunResult(
        command=shell_join(argv),
        cwd=str(repo_root),
        request_count=len(requests),
        warmup_requests=int(args.warmup_requests),
        measured_requests=int(args.requests),
        requests_json=str(requests_json),
        stderr_log=str(stderr_log),
        status="passed",
        worker_startup_ms=worker_startup_ms,
        max_rss_bytes=_child_max_rss_bytes(),
        requests=tuple(parsed),
    )


def summarize_numeric(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[-1] if len(ordered) < 2 else statistics.quantiles(ordered, n=20, method="inclusive")[18],
        "max": ordered[-1],
    }


def request_metric_summary(result: ServingRunResult, metric: str, *, phase: str = "measured") -> dict[str, float | int] | None:
    if metric == "persistent_request_latency":
        return summarize_numeric([item.persistent_request_latency_ms for item in result.requests if item.phase == phase])
    return summarize_numeric([item.timings_ms[metric] for item in result.requests if item.phase == phase and metric in item.timings_ms])


def build_json_summary(result: ServingRunResult, *, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_kind": "persistent-local-serving",
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
        "worker": {
            "status": result.status,
            "command": result.command,
            "cwd": result.cwd,
            "startup_ms": result.worker_startup_ms,
            "max_rss_bytes": result.max_rss_bytes,
            "request_count": result.request_count,
            "warmup_requests": result.warmup_requests,
            "measured_requests": result.measured_requests,
        },
        "aggregates": {
            "measured_persistent_request_latency_ms": request_metric_summary(result, "persistent_request_latency"),
            "measured_sample_rf_ms": request_metric_summary(result, "sample_rf"),
            "measured_decode_dacvae_ms": request_metric_summary(result, "decode_dacvae"),
            "measured_decode_dacvae_model_ms": request_metric_summary(result, "decode_dacvae_model"),
            "measured_audio_write_ms": request_metric_summary(result, "audio_write"),
            "measured_total_to_decode_ms": request_metric_summary(result, "total_to_decode"),
        },
        "requests": [asdict(item) for item in result.requests],
    }


def format_ms(value: float | int | None) -> str:
    return "" if value is None else f"{float(value):.1f} ms"


def format_bytes(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value} bytes ({value / float(1024**3):.2f} GiB)"


def build_report(result: ServingRunResult, *, args: argparse.Namespace) -> str:
    summary = build_json_summary(result, args=args)
    aggregates = summary["aggregates"]
    first = result.requests[0] if result.requests else None

    def median(name: str) -> str:
        item = aggregates[name] or {}
        return format_ms(item.get("median") if item else None)

    def p95(name: str) -> str:
        item = aggregates[name] or {}
        return format_ms(item.get("p95") if item else None)

    lines = [
        "# Persistent Local Serving Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Case label: {args.case_label}",
        f"- Requests: {result.request_count} ({result.warmup_requests} warmup, {result.measured_requests} measured)",
        f"- Prompt text: {args.text}",
        f"- Seed start: {args.seed}",
        f"- Num steps: {args.num_steps}",
        f"- Seconds: {'predicted duration (--seconds omitted)' if args.omit_seconds else args.seconds}",
        f"- Codec runtime mode: {args.codec_runtime_mode}",
        "",
        "## Worker results",
        "",
        "| Status | Startup | Max RSS |",
        "| --- | ---: | ---: |",
        f"| {result.status} | {format_ms(result.worker_startup_ms)} | {format_bytes(result.max_rss_bytes)} |",
        "",
        "## Request timing aggregates",
        "",
        "| Scope | first request | measured median | measured p95 |",
        "| --- | ---: | ---: | ---: |",
        f"| persistent request latency | {format_ms(first.persistent_request_latency_ms if first else None)} | {median('measured_persistent_request_latency_ms')} | {p95('measured_persistent_request_latency_ms')} |",
        f"| total_to_decode | {format_ms(first.timings_ms.get('total_to_decode') if first else None)} | {median('measured_total_to_decode_ms')} | {p95('measured_total_to_decode_ms')} |",
        f"| sample_rf | {format_ms(first.timings_ms.get('sample_rf') if first else None)} | {median('measured_sample_rf_ms')} | {p95('measured_sample_rf_ms')} |",
        f"| decode_dacvae | {format_ms(first.timings_ms.get('decode_dacvae') if first else None)} | {median('measured_decode_dacvae_ms')} | {p95('measured_decode_dacvae_ms')} |",
        f"| decode_dacvae_model | {format_ms(first.timings_ms.get('decode_dacvae_model') if first else None)} | {median('measured_decode_dacvae_model_ms')} | {p95('measured_decode_dacvae_model_ms')} |",
        f"| audio_write | {format_ms(first.timings_ms.get('audio_write') if first else None)} | {median('measured_audio_write_ms')} | {p95('measured_audio_write_ms')} |",
        "",
        "## Raw requests",
        "",
        "| # | Phase | Seed | request latency | total_to_decode | sample_rf | decode_dacvae | decode_model | audio_write | Output |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.requests:
        lines.append(
            "| {index} | {phase} | {seed} | {latency} | {total} | {sample} | {decode} | {decode_model} | {write} | {output} |".format(
                index=item.index,
                phase=item.phase,
                seed="" if item.seed is None else item.seed,
                latency=format_ms(item.persistent_request_latency_ms),
                total=format_ms(item.timings_ms.get("total_to_decode")),
                sample=format_ms(item.timings_ms.get("sample_rf")),
                decode=format_ms(item.timings_ms.get("decode_dacvae")),
                decode_model=format_ms(item.timings_ms.get("decode_dacvae_model")),
                write=format_ms(item.timings_ms.get("audio_write")),
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
        f"- stderr log: {result.stderr_log or 'n/a'}",
    ])
    return "\n".join(lines) + "\n"


def run_self_test() -> int:
    requests = (
        RequestResult(1, "warmup", "/tmp/1.wav", "one", 1, 12, 150.0, 145.0, 0.2, {"sample_rf": 90.0, "decode_dacvae": 40.0, "decode_dacvae_model": 35.0, "audio_write": 5.0, "total_to_decode": 140.0}, None, "mlx"),
        RequestResult(2, "measured", "/tmp/2.wav", "two", 2, 12, 130.0, 126.0, 0.2, {"sample_rf": 80.0, "decode_dacvae": 30.0, "decode_dacvae_model": 25.0, "audio_write": 5.0, "total_to_decode": 120.0}, None, "mlx"),
        RequestResult(3, "measured", "/tmp/3.wav", "three", 3, 12, 170.0, 166.0, 0.2, {"sample_rf": 100.0, "decode_dacvae": 45.0, "decode_dacvae_model": 38.0, "audio_write": 7.0, "total_to_decode": 160.0}, None, "mlx"),
    )
    result = ServingRunResult("python scripts/benchmark_persistent_serving.py ...", "/tmp/repo", 3, 1, 2, "/tmp/requests.json", "/tmp/stderr.log", "passed", 1000.0, 1234, requests)
    args = argparse.Namespace(case_label="self-test", text=DEFAULT_TEXT, caption=None, seed=1, requests=2, warmup_requests=1, seconds=5.0, omit_seconds=False, num_steps=12, reference_wav=None, weights=None, weights_dir=None, weights_repo="repo", weights_revision=None, codec_runtime_mode="mlx-decode", codec_path=None, codec_artifact_dir=None, codec_artifact_repo="codec", codec_artifact_revision=None, cleanup_between_requests=False)
    summary = build_json_summary(result, args=args)
    assert summary["aggregates"]["measured_persistent_request_latency_ms"]["median"] == 150.0
    assert summary["aggregates"]["measured_audio_write_ms"]["median"] == 6.0
    assert _ru_maxrss_value_to_bytes(1234, platform="linux") == 1263616
    assert _ru_maxrss_value_to_bytes(1234, platform="darwin") == 1234
    report = build_report(result, args=args)
    assert "Persistent Local Serving Benchmark Report" in report
    assert "audio_write" in report
    print("self-test passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.worker:
        return run_worker(args)
    repo_root = ROOT
    output_dir = ensure_directory((repo_root / args.output_dir).resolve())
    result = run_serving(args, repo_root, output_dir)
    summary = build_json_summary(result, args=args)
    (output_dir / "persistent-serving-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    except PersistentServingBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
