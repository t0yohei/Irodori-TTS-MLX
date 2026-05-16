#!/usr/bin/env python3
"""Run or plan upstream PyTorch vs MLX generation parity checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "こんにちは。今日は良い天気です。"
DEFAULT_CAPTION = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


@dataclass(frozen=True)
class Scenario:
    name: str
    checkpoint_family: str
    checkpoint: str
    text: str
    no_reference: bool
    reference_wav: str | None
    caption: str | None
    seconds: float | None
    duration_scale: float
    num_steps: int
    seed: int
    model_device: str
    codec_device: str
    codec_repo: str
    text_tokenizer_repo: str | None
    caption_tokenizer_repo: str | None
    text_max_length: int
    caption_max_length: int | None
    cfg_scale_text: float
    cfg_scale_caption: float
    cfg_scale_speaker: float
    cfg_guidance_mode: str
    cfg_min_t: float
    cfg_max_t: float


def _scenario_presets() -> dict[str, dict[str, Any]]:
    return {
        "v3-no-reference": {
            "checkpoint_family": "v3",
            "checkpoint": "Aratako/Irodori-TTS-500M-v3",
            "text": DEFAULT_TEXT,
            "no_reference": True,
            "reference_wav": None,
            "caption": None,
            "seconds": None,
            "duration_scale": 1.0,
            "num_steps": 8,
            "seed": 20260516,
        },
        "voicedesign-no-reference": {
            "checkpoint_family": "voicedesign",
            "checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
            "text": DEFAULT_TEXT,
            "no_reference": True,
            "reference_wav": None,
            "caption": DEFAULT_CAPTION,
            "seconds": 2.0,
            "duration_scale": 1.0,
            "num_steps": 8,
            "seed": 20260516,
        },
    }


def _load_json_object(path_or_inline: str | None) -> dict[str, Any]:
    if not path_or_inline:
        return {}
    raw = path_or_inline.strip()
    payload = json.loads(raw) if raw.startswith("{") else json.loads(Path(raw).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario JSON must be an object")
    return payload


def build_scenario(args: argparse.Namespace) -> Scenario:
    payload = dict(_scenario_presets()[args.scenario])
    payload.update(_load_json_object(args.scenario_json))
    payload.update(
        {
            "name": args.scenario_name or args.scenario,
            "model_device": args.model_device,
            "codec_device": args.codec_device,
            "codec_repo": args.codec_repo,
            "text_tokenizer_repo": args.text_tokenizer_repo,
            "caption_tokenizer_repo": args.caption_tokenizer_repo,
            "text_max_length": args.text_max_length,
            "caption_max_length": args.caption_max_length,
            "cfg_scale_text": args.cfg_scale_text,
            "cfg_scale_caption": args.cfg_scale_caption,
            "cfg_scale_speaker": args.cfg_scale_speaker,
            "cfg_guidance_mode": args.cfg_guidance_mode,
            "cfg_min_t": args.cfg_min_t,
            "cfg_max_t": args.cfg_max_t,
        }
    )
    for key in ("text", "caption", "seconds", "num_steps", "seed", "checkpoint", "reference_wav"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    if args.no_reference:
        payload["no_reference"] = True
        payload["reference_wav"] = None
    if payload.get("reference_wav"):
        payload["no_reference"] = False
    if payload.get("checkpoint_family") == "voicedesign" and not payload.get("caption"):
        raise ValueError("VoiceDesign scenarios require caption text")
    if not payload.get("no_reference") and not payload.get("reference_wav"):
        raise ValueError("scenario must set no_reference=true or provide reference_wav")
    return Scenario(**payload)


def _upstream_command(scenario: Scenario, output_wav: Path) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "infer.py",
        "--hf-checkpoint",
        scenario.checkpoint,
        "--text",
        scenario.text,
        "--output-wav",
        str(output_wav),
        "--model-device",
        scenario.model_device,
        "--codec-device",
        scenario.codec_device,
        "--model-precision",
        "fp32",
        "--codec-precision",
        "fp32",
        "--num-steps",
        str(scenario.num_steps),
        "--seed",
        str(scenario.seed),
        "--show-timings",
    ]
    command.extend(["--no-ref"] if scenario.no_reference else ["--ref-wav", str(scenario.reference_wav)])
    if scenario.caption:
        command.extend(["--caption", scenario.caption])
    return command


def _mlx_command(scenario: Scenario, args: argparse.Namespace, output_wav: Path, metadata_json: Path) -> list[str]:
    weights = args.mlx_weights or "/path/to/converted-mlx-weights.npz"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_wav.py"),
        "--weights",
        str(Path(weights).expanduser()) if args.mlx_weights else weights,
        "--text",
        scenario.text,
        "--output",
        str(output_wav),
        "--num-steps",
        str(scenario.num_steps),
        "--seed",
        str(scenario.seed),
        "--duration-scale",
        str(scenario.duration_scale),
        "--codec-repo",
        scenario.codec_repo,
        "--codec-device",
        scenario.codec_device,
        "--cfg-scale-text",
        str(scenario.cfg_scale_text),
        "--cfg-scale-caption",
        str(scenario.cfg_scale_caption),
        "--cfg-scale-speaker",
        str(scenario.cfg_scale_speaker),
        "--cfg-guidance-mode",
        scenario.cfg_guidance_mode,
        "--cfg-min-t",
        str(scenario.cfg_min_t),
        "--cfg-max-t",
        str(scenario.cfg_max_t),
        "--text-max-length",
        str(scenario.text_max_length),
        "--metadata-json",
        str(metadata_json),
        "--json",
    ]
    model_config = args.mlx_model_config_json or ("/path/to/model-config.json" if not args.mlx_weights else None)
    if model_config:
        command.extend(["--model-config-json", str(Path(model_config).expanduser()) if args.mlx_model_config_json else model_config])
    command.extend(["--no-reference"] if scenario.no_reference else ["--reference-wav", str(scenario.reference_wav)])
    if scenario.caption:
        command.extend(["--caption", scenario.caption])
    if scenario.seconds is not None:
        command.extend(["--seconds", str(scenario.seconds)])
    if scenario.text_tokenizer_repo:
        command.extend(["--text-tokenizer-repo", scenario.text_tokenizer_repo])
    if scenario.caption_tokenizer_repo:
        command.extend(["--caption-tokenizer-repo", scenario.caption_tokenizer_repo])
    if scenario.caption_max_length is not None:
        command.extend(["--caption-max-length", str(scenario.caption_max_length)])
    return command


def wav_properties(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with wave.open(str(path), "rb") as fh:
            frames = fh.getnframes()
            sample_rate = fh.getframerate()
            channels = fh.getnchannels()
            sample_width = fh.getsampwidth()
    except (OSError, wave.Error):
        return {"path": str(path), "bytes": path.stat().st_size, "readable": False}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "readable": True,
        "sample_rate": sample_rate,
        "samples": frames,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "duration_seconds": frames / sample_rate if sample_rate else None,
    }


def _run(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_seconds, check=False)
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_excerpt": completed.stdout[-4000:],
        "stderr_excerpt": completed.stderr[-4000:],
    }


def _fixture_side(name: str, output_wav: Path, command: list[str], *, duration_mode: str, seed: int) -> dict[str, Any]:
    samples = 48000 if duration_mode == "manual" else 36000
    return {
        "name": name,
        "status": "fixture",
        "command": {"argv": command, "shell": shlex.join(command)},
        "audio": {
            "path": str(output_wav),
            "bytes": 96044,
            "readable": True,
            "sample_rate": 24000,
            "samples": samples,
            "channels": 1,
            "sample_width_bytes": 2,
            "duration_seconds": samples / 24000,
        },
        "metadata": {"result": {"duration_mode": duration_mode, "resolved_seconds": samples / 24000, "samples": samples, "seed": seed}},
    }


def _metadata_axes(scenario: Scenario) -> dict[str, Any]:
    return {
        "tokenizer": {
            "text_tokenizer_repo": scenario.text_tokenizer_repo,
            "caption_tokenizer_repo": scenario.caption_tokenizer_repo,
            "text_max_length": scenario.text_max_length,
            "caption_max_length": scenario.caption_max_length,
            "caption_enabled": bool(scenario.caption),
        },
        "duration": {
            "seconds": scenario.seconds,
            "duration_scale": scenario.duration_scale,
            "expected_mode": "manual" if scenario.seconds is not None else "predicted_or_upstream_default",
        },
        "sampling": {
            "num_steps": scenario.num_steps,
            "seed": scenario.seed,
            "cfg_scale_text": scenario.cfg_scale_text,
            "cfg_scale_caption": scenario.cfg_scale_caption,
            "cfg_scale_speaker": scenario.cfg_scale_speaker,
            "cfg_guidance_mode": scenario.cfg_guidance_mode,
            "cfg_min_t": scenario.cfg_min_t,
            "cfg_max_t": scenario.cfg_max_t,
        },
        "codec": {
            "codec_repo": scenario.codec_repo,
            "codec_device": scenario.codec_device,
            "reference_wav": scenario.reference_wav,
            "no_reference": scenario.no_reference,
        },
    }


def _classify(upstream: dict[str, Any], mlx: dict[str, Any]) -> dict[str, Any]:
    if upstream["status"] not in {"passed", "fixture"} or mlx["status"] not in {"passed", "fixture"}:
        return {"status": "not_comparable", "reason": "both sides must pass, or both must be deterministic fixtures"}
    up_audio = upstream.get("audio") or {}
    mlx_audio = mlx.get("audio") or {}
    if not up_audio or not mlx_audio:
        return {"status": "not_comparable", "reason": "missing audio properties"}
    sample_rate_match = up_audio.get("sample_rate") == mlx_audio.get("sample_rate")
    duration_delta = None
    if up_audio.get("duration_seconds") is not None and mlx_audio.get("duration_seconds") is not None:
        duration_delta = float(mlx_audio["duration_seconds"]) - float(up_audio["duration_seconds"])
    status = "expected_drift"
    reasons = ["audio waveforms are not expected to be bit-identical across PyTorch and MLX generation"]
    if not sample_rate_match:
        status = "regression"
        reasons.append("sample rates differ")
    if duration_delta is not None and abs(duration_delta) > 0.25:
        status = "regression"
        reasons.append("duration differs by more than 250 ms")
    return {"status": status, "sample_rate_match": sample_rate_match, "duration_delta_seconds": duration_delta, "reasons": reasons}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scenario = build_scenario(args)
    if args.run_mlx and not args.mlx_weights:
        raise ValueError("--mlx-weights is required when --run-mlx is set")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_upstream or args.run_mlx:
        output_dir = output_dir.resolve()
    upstream_wav = output_dir / f"{scenario.name}.upstream.wav"
    mlx_wav = output_dir / f"{scenario.name}.mlx.wav"
    mlx_metadata = output_dir / f"{scenario.name}.mlx.metadata.json"
    upstream_command = _upstream_command(scenario, upstream_wav)
    mlx_command = _mlx_command(scenario, args, mlx_wav, mlx_metadata)
    duration_mode = "manual" if scenario.seconds is not None else "predicted"

    if args.fixture:
        upstream = _fixture_side("upstream", upstream_wav, upstream_command, duration_mode=duration_mode, seed=scenario.seed)
        mlx = _fixture_side("mlx", mlx_wav, mlx_command, duration_mode=duration_mode, seed=scenario.seed)
    else:
        upstream = {"name": "upstream", "status": "not_run", "command": {"argv": upstream_command, "shell": shlex.join(upstream_command)}, "audio": wav_properties(upstream_wav), "metadata": {}}
        mlx = {"name": "mlx", "status": "not_run", "command": {"argv": mlx_command, "shell": shlex.join(mlx_command)}, "audio": wav_properties(mlx_wav), "metadata": {}}
        if args.run_upstream:
            if not args.upstream_root:
                raise ValueError("--upstream-root is required when --run-upstream is set")
            upstream.update(_run(upstream_command, cwd=Path(args.upstream_root).expanduser().resolve(), timeout_seconds=args.timeout_seconds))
            upstream["audio"] = wav_properties(upstream_wav)
        if args.run_mlx:
            mlx.update(_run(mlx_command, cwd=ROOT, timeout_seconds=args.timeout_seconds))
            mlx["audio"] = wav_properties(mlx_wav)
            if mlx_metadata.exists():
                mlx["metadata"] = json.loads(mlx_metadata.read_text(encoding="utf-8"))

    return {
        "schema_version": 1,
        "created_by": "scripts/run_upstream_parity.py",
        "environment": {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd()},
        "scenario": asdict(scenario),
        "metadata_axes": _metadata_axes(scenario),
        "artifacts": {"output_dir": str(output_dir), "commit_note": "Generated WAVs, upstream checkouts, and checkpoint caches must stay outside git."},
        "upstream": upstream,
        "mlx": mlx,
        "comparison": _classify(upstream, mlx),
        "deferred_scope": [
            "full VoiceDesign real-checkpoint baseline matrix",
            "full v3 real-checkpoint baseline matrix",
            "intermediate tensor and perceptual audio metrics beyond WAV properties",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible upstream PyTorch vs MLX generation parity report.")
    parser.add_argument("--scenario", choices=tuple(_scenario_presets()), default="v3-no-reference")
    parser.add_argument("--scenario-name")
    parser.add_argument("--scenario-json")
    parser.add_argument("--output-dir", default="parity-runs")
    parser.add_argument("--report-json")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--run-upstream", action="store_true")
    parser.add_argument("--run-mlx", action="store_true")
    parser.add_argument("--upstream-root")
    parser.add_argument("--mlx-weights")
    parser.add_argument("--mlx-model-config-json")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--checkpoint")
    parser.add_argument("--text")
    parser.add_argument("--caption")
    parser.add_argument("--reference-wav")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-device", default="mps")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--text-tokenizer-repo")
    parser.add_argument("--caption-tokenizer-repo")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--caption-max-length", type=int)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--cfg-guidance-mode", choices=("independent", "joint", "reduced"), default="independent")
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    report_path = Path(args.report_json).expanduser() if args.report_json else Path(args.output_dir).expanduser() / f"{report['scenario']['name']}.parity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(f"wrote parity report: {report_path}")
        print(f"comparison: {report['comparison']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
