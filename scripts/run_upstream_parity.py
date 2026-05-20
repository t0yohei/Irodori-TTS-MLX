#!/usr/bin/env python3
"""Run or plan upstream PyTorch vs MLX generation parity checks."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import struct
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "こんにちは。今日は良い天気です。"
V3_REFERENCE_TEXT = "音声参照を使ったv3予測時間の確認です。短く自然に読み上げます。"
V3_REFERENCE_WAV = "tests/fixtures/v3-reference.wav"
DEFAULT_CAPTION = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"
VOICEDESIGN_CONTRAST_TEXT = "新しい音声デザインの確認です。短い案内文をはっきり読み上げます。"
VOICEDESIGN_CONTRAST_CAPTION = "低めの落ち着いた男性の声で、遠くから響くようにゆっくり読み上げてください。"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
SCHEMA_VERSION = 1
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


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
        "v3-reference-predicted": {
            "checkpoint_family": "v3",
            "checkpoint": "Aratako/Irodori-TTS-500M-v3",
            "text": V3_REFERENCE_TEXT,
            "no_reference": False,
            "reference_wav": V3_REFERENCE_WAV,
            "caption": None,
            "seconds": None,
            "duration_scale": 1.0,
            "num_steps": 8,
            "seed": 20260519,
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
        "voicedesign-contrastive-caption": {
            "checkpoint_family": "voicedesign",
            "checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
            "text": VOICEDESIGN_CONTRAST_TEXT,
            "no_reference": True,
            "reference_wav": None,
            "caption": VOICEDESIGN_CONTRAST_CAPTION,
            "seconds": 2.0,
            "duration_scale": 1.0,
            "num_steps": 12,
            "seed": 20260518,
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


def _reference_wav_path(scenario: Scenario) -> Path | None:
    if not scenario.reference_wav:
        return None
    path = Path(scenario.reference_wav).expanduser()
    return path if path.is_absolute() else ROOT / path


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
    reference_wav = _reference_wav_path(scenario)
    command.extend(["--no-ref"] if scenario.no_reference else ["--ref-wav", str(reference_wav)])
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
    if args.codec_runtime_mode:
        command.extend(["--codec-runtime-mode", args.codec_runtime_mode])
    if args.codec_path:
        command.extend(["--codec-path", str(Path(args.codec_path).expanduser().resolve())])
    if args.codec_artifact_dir:
        command.extend(["--codec-artifact-dir", str(Path(args.codec_artifact_dir).expanduser().resolve())])
    if args.codec_artifact_repo:
        command.extend(["--codec-artifact-repo", args.codec_artifact_repo])
    if args.codec_artifact_revision and args.codec_artifact_repo:
        command.extend(["--codec-artifact-revision", args.codec_artifact_revision])
    model_config = args.mlx_model_config_json or ("/path/to/model-config.json" if not args.mlx_weights else None)
    if model_config:
        command.extend(["--model-config-json", str(Path(model_config).expanduser()) if args.mlx_model_config_json else model_config])
    reference_wav = _reference_wav_path(scenario)
    command.extend(["--no-reference"] if scenario.no_reference else ["--reference-wav", str(reference_wav)])
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


def _wav_format_name(format_tag: int) -> str:
    if format_tag == WAVE_FORMAT_PCM:
        return "pcm"
    if format_tag == WAVE_FORMAT_IEEE_FLOAT:
        return "ieee_float"
    return f"unknown_{format_tag}"


def _read_wav_chunks(path: Path) -> dict[str, Any] | None:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    fmt: bytes | None = None
    payload: bytes | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            return None
        if chunk_id == b"fmt ":
            fmt = data[chunk_start:chunk_end]
        elif chunk_id == b"data":
            payload = data[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size % 2)
    if fmt is None or payload is None or len(fmt) < 16:
        return None
    format_tag, channels, sample_rate, _byte_rate, block_align, bits_per_sample = struct.unpack_from("<HHIIHH", fmt, 0)
    if format_tag == WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 40:
        format_tag = struct.unpack_from("<H", fmt, 24)[0]
    sample_width = max(1, int(bits_per_sample + 7) // 8)
    frames = len(payload) // block_align if block_align else 0
    return {
        "format_tag": int(format_tag),
        "format": _wav_format_name(int(format_tag)),
        "channels": int(channels),
        "sample_rate": int(sample_rate),
        "sample_width": int(sample_width),
        "bits_per_sample": int(bits_per_sample),
        "block_align": int(block_align),
        "frames": int(frames),
        "data": payload[: frames * block_align] if block_align else b"",
    }


def _pcm_frame_values(raw: bytes, *, sample_width: int, channels: int) -> list[float]:
    if not raw:
        return []
    values: list[float] = []
    frame_bytes = int(sample_width) * int(channels)
    if frame_bytes <= 0:
        return values
    for offset in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        channel_values: list[float] = []
        for channel in range(channels):
            start = offset + channel * sample_width
            sample = raw[start : start + sample_width]
            if sample_width == 1:
                value = int(sample[0]) - 128
                scale = 128.0
            elif sample_width == 2:
                value = int.from_bytes(sample, byteorder="little", signed=True)
                scale = 32768.0
            elif sample_width == 3:
                sign_byte = b"\xff" if sample[-1] & 0x80 else b"\x00"
                value = int.from_bytes(sample + sign_byte, byteorder="little", signed=True)
                scale = 8388608.0
            elif sample_width == 4:
                value = int.from_bytes(sample, byteorder="little", signed=True)
                scale = 2147483648.0
            else:
                return []
            channel_values.append(max(-1.0, min(1.0, float(value) / scale)))
        values.append(mean(channel_values))
    return values


def _float_frame_values(raw: bytes, *, sample_width: int, channels: int) -> list[float]:
    if sample_width not in {4, 8}:
        return []
    values: list[float] = []
    frame_bytes = int(sample_width) * int(channels)
    if frame_bytes <= 0:
        return values
    fmt = "<f" if sample_width == 4 else "<d"
    for offset in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        channel_values: list[float] = []
        for channel in range(channels):
            start = offset + channel * sample_width
            value = float(struct.unpack(fmt, raw[start : start + sample_width])[0])
            channel_values.append(value if math.isfinite(value) else 0.0)
        values.append(mean(channel_values))
    return values


def _wav_frame_values(raw: bytes, *, sample_width: int, channels: int, format_tag: int) -> list[float]:
    if format_tag == WAVE_FORMAT_PCM:
        return _pcm_frame_values(raw, sample_width=sample_width, channels=channels)
    if format_tag == WAVE_FORMAT_IEEE_FLOAT:
        return _float_frame_values(raw, sample_width=sample_width, channels=channels)
    return []


def _audio_metrics(samples: list[float], *, sample_rate: int, tail_seconds: float = 0.25) -> dict[str, Any]:
    if not samples:
        return {
            "peak_abs": 0.0,
            "rms": 0.0,
            "mean_abs": 0.0,
            "silence_ratio": 1.0,
            "leading_silence_seconds": 0.0,
            "tail_rms": 0.0,
            "tail_silence_ratio": 1.0,
            "zero_crossing_rate": 0.0,
        }
    abs_values = [abs(value) for value in samples]
    silence_threshold = 1.0e-4
    tail_count = min(len(samples), max(1, int(float(sample_rate) * tail_seconds))) if sample_rate else len(samples)
    tail = samples[-tail_count:]
    leading_silent = 0
    for value in abs_values:
        if value > silence_threshold:
            break
        leading_silent += 1
    crossings = sum(
        1
        for previous, current in zip(samples, samples[1:])
        if (previous < 0.0 <= current) or (previous >= 0.0 > current)
    )
    return {
        "peak_abs": max(abs_values),
        "rms": math.sqrt(sum(value * value for value in samples) / len(samples)),
        "mean_abs": sum(abs_values) / len(samples),
        "silence_ratio": sum(1 for value in abs_values if value <= silence_threshold) / len(samples),
        "leading_silence_seconds": leading_silent / sample_rate if sample_rate else None,
        "tail_rms": math.sqrt(sum(value * value for value in tail) / len(tail)) if tail else 0.0,
        "tail_silence_ratio": sum(1 for value in tail if abs(value) <= silence_threshold) / len(tail) if tail else 1.0,
        "zero_crossing_rate": crossings / max(1, len(samples) - 1),
    }


def wav_properties(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    parsed = _read_wav_chunks(path)
    if parsed is not None:
        samples = _wav_frame_values(
            parsed["data"],
            sample_width=parsed["sample_width"],
            channels=parsed["channels"],
            format_tag=parsed["format_tag"],
        )
        metrics = _audio_metrics(samples, sample_rate=parsed["sample_rate"]) if samples else None
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "readable": True,
            "format": parsed["format"],
            "format_tag": parsed["format_tag"],
            "sample_rate": parsed["sample_rate"],
            "samples": parsed["frames"],
            "channels": parsed["channels"],
            "sample_width_bytes": parsed["sample_width"],
            "bits_per_sample": parsed["bits_per_sample"],
            "duration_seconds": parsed["frames"] / parsed["sample_rate"] if parsed["sample_rate"] else None,
            "metrics": metrics,
            "metrics_status": "computed" if metrics is not None else "unsupported_format",
        }
    try:
        with wave.open(str(path), "rb") as fh:
            frames = fh.getnframes()
            sample_rate = fh.getframerate()
            channels = fh.getnchannels()
            sample_width = fh.getsampwidth()
            raw = fh.readframes(frames)
    except (OSError, wave.Error):
        return {"path": str(path), "bytes": path.stat().st_size, "readable": False}
    samples = _pcm_frame_values(raw, sample_width=sample_width, channels=channels)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "readable": True,
        "sample_rate": sample_rate,
        "samples": frames,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "duration_seconds": frames / sample_rate if sample_rate else None,
        "metrics": _audio_metrics(samples, sample_rate=sample_rate),
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


def _side_report(
    name: str,
    status: str,
    command: list[str],
    output_wav: Path,
    *,
    reason: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    availability = {"state": status, "reason": reason, "detail": detail}
    return {
        "name": name,
        "status": status,
        "command": {"argv": command, "shell": shlex.join(command)},
        "availability": availability,
        "audio": None,
        "metadata": {},
        "intermediates": {},
    }


def _fixture_audio_metrics(*, duration_mode: str, name: str) -> dict[str, Any]:
    if duration_mode == "manual":
        return {
            "peak_abs": 0.28 if name == "upstream" else 0.27,
            "rms": 0.075 if name == "upstream" else 0.073,
            "mean_abs": 0.052 if name == "upstream" else 0.051,
            "silence_ratio": 0.04,
            "leading_silence_seconds": 0.0,
            "tail_rms": 0.012 if name == "upstream" else 0.013,
            "tail_silence_ratio": 0.18,
            "zero_crossing_rate": 0.026 if name == "upstream" else 0.027,
        }
    return {
        "peak_abs": 0.25 if name == "upstream" else 0.24,
        "rms": 0.068 if name == "upstream" else 0.066,
        "mean_abs": 0.047 if name == "upstream" else 0.046,
        "silence_ratio": 0.05,
        "leading_silence_seconds": 0.0,
        "tail_rms": 0.014 if name == "upstream" else 0.015,
        "tail_silence_ratio": 0.2,
        "zero_crossing_rate": 0.024 if name == "upstream" else 0.025,
    }


def _fixture_side(name: str, output_wav: Path, command: list[str], *, duration_mode: str, seed: int) -> dict[str, Any]:
    samples = 48000 if duration_mode == "manual" else 36000
    latent_steps = samples // 480
    result_metadata = {
        "duration_mode": duration_mode,
        "resolved_seconds": samples / 24000,
        "samples": samples,
        "latent_steps": latent_steps,
        "patched_steps": latent_steps,
        "seed": seed,
    }
    if duration_mode == "predicted":
        result_metadata.update(
            {
                "requested_seconds": None,
                "predicted_duration": {"source": "fixture", "predicted_seconds": samples / 24000, "duration_scale": 1.0},
                "messages": ["predicted duration active (fixture)"],
            }
        )
    return {
        "name": name,
        "status": "fixture",
        "command": {"argv": command, "shell": shlex.join(command)},
        "availability": {"state": "fixture", "reason": None, "detail": "deterministic fixture; no external artifacts required"},
        "audio": {
            "path": str(output_wav),
            "bytes": 96044,
            "readable": True,
            "sample_rate": 24000,
            "samples": samples,
            "channels": 1,
            "sample_width_bytes": 2,
            "duration_seconds": samples / 24000,
            "metrics": _fixture_audio_metrics(duration_mode=duration_mode, name=name),
        },
        "metadata": {"result": result_metadata},
        "intermediates": {
            "tokenizer": {"text_token_count": 12, "text_mask_true": 12},
            "duration": {"mode": duration_mode, "resolved_seconds": samples / 24000, "latent_steps": latent_steps},
            "sampling": {
                "seed": seed,
                "latent_shape": [1, latent_steps, 32],
                "latent_mean": 0.001 if name == "upstream" else 0.002,
                "latent_std": 0.82 if name == "upstream" else 0.83,
            },
        },
    }


def _missing_reference_report(name: str, command: list[str], output_wav: Path, reference_wav: Path) -> dict[str, Any]:
    return _side_report(
        name,
        "unavailable",
        command,
        output_wav,
        reason="missing_reference_wav",
        detail=f"Reference WAV does not exist: {reference_wav}",
    )


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
    metric_deltas = _audio_metric_deltas(up_audio, mlx_audio)
    intermediate_comparisons = _compare_intermediates(upstream.get("intermediates") or {}, mlx.get("intermediates") or {})
    status = "expected_drift"
    reasons = ["audio waveforms are not expected to be bit-identical across PyTorch and MLX generation"]
    if not sample_rate_match:
        status = "regression"
        reasons.append("sample rates differ")
    if duration_delta is not None and abs(duration_delta) > 0.25:
        status = "regression"
        reasons.append("duration differs by more than 250 ms")
    return {
        "status": status,
        "sample_rate_match": sample_rate_match,
        "duration_delta_seconds": duration_delta,
        "audio_metric_deltas": metric_deltas,
        "intermediate_comparisons": intermediate_comparisons,
        "reasons": reasons,
    }


def _audio_metric_deltas(up_audio: dict[str, Any], mlx_audio: dict[str, Any]) -> dict[str, Any]:
    up_metrics = up_audio.get("metrics") or {}
    mlx_metrics = mlx_audio.get("metrics") or {}
    deltas: dict[str, Any] = {}
    for key in ("peak_abs", "rms", "mean_abs", "tail_rms", "silence_ratio", "tail_silence_ratio", "zero_crossing_rate"):
        if up_metrics.get(key) is not None and mlx_metrics.get(key) is not None:
            deltas[f"{key}_delta"] = float(mlx_metrics[key]) - float(up_metrics[key])
    if up_metrics.get("rms") and mlx_metrics.get("rms") is not None:
        deltas["rms_ratio"] = float(mlx_metrics.get("rms", 0.0)) / float(up_metrics["rms"])
    return deltas


def _compare_intermediates(upstream: dict[str, Any], mlx: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for path in (
        ("tokenizer", "text_token_count"),
        ("tokenizer", "text_mask_true"),
        ("duration", "mode"),
        ("duration", "latent_steps"),
        ("sampling", "latent_shape"),
    ):
        up_value = upstream
        mlx_value = mlx
        for key in path:
            up_value = up_value.get(key) if isinstance(up_value, dict) else None
            mlx_value = mlx_value.get(key) if isinstance(mlx_value, dict) else None
        label = ".".join(path)
        if up_value is not None or mlx_value is not None:
            comparisons[label] = {"upstream": up_value, "mlx": mlx_value, "match": up_value == mlx_value}
    return comparisons


def _mlx_intermediates_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = metadata.get("result") if isinstance(metadata.get("result"), dict) else {}
    request = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
    boundaries = metadata.get("boundaries") if isinstance(metadata.get("boundaries"), dict) else {}
    codec = boundaries.get("codec") if isinstance(boundaries.get("codec"), dict) else {}
    config = boundaries.get("config") if isinstance(boundaries.get("config"), dict) else {}
    model_config = config.get("model_config") if isinstance(config.get("model_config"), dict) else {}
    latent_steps = result.get("latent_steps")
    latent_dim = codec.get("latent_dim") or codec.get("decode_latent_dim") or model_config.get("latent_dim")
    sampling: dict[str, Any] = {
        "seed": result.get("seed") or request.get("seed"),
        "latent_steps": latent_steps,
        "patched_steps": result.get("patched_steps"),
    }
    if latent_steps is not None and latent_dim is not None:
        sampling["latent_shape"] = [1, int(latent_steps), int(latent_dim)]
    return {
        "tokenizer": {
            "text_max_length": request.get("text_max_length"),
            "caption_max_length": request.get("caption_max_length"),
            "caption_enabled": bool(request.get("caption")),
        },
        "duration": {
            "mode": result.get("duration_mode"),
            "requested_seconds": result.get("requested_seconds"),
            "resolved_seconds": result.get("resolved_seconds"),
            "latent_steps": latent_steps,
        },
        "sampling": sampling,
    }


def _report_status(upstream: dict[str, Any], mlx: dict[str, Any]) -> str:
    statuses = {upstream["status"], mlx["status"]}
    if statuses <= {"passed", "fixture"}:
        return "complete"
    if "failed" in statuses:
        return "failed"
    return "partial"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scenario = build_scenario(args)
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
    reference_wav = _reference_wav_path(scenario)

    if args.fixture:
        upstream = _fixture_side("upstream", upstream_wav, upstream_command, duration_mode=duration_mode, seed=scenario.seed)
        mlx = _fixture_side("mlx", mlx_wav, mlx_command, duration_mode=duration_mode, seed=scenario.seed)
    else:
        upstream = _side_report("upstream", "not_run", upstream_command, upstream_wav, reason="not_requested")
        mlx = _side_report("mlx", "not_run", mlx_command, mlx_wav, reason="not_requested")
        if args.run_upstream:
            if reference_wav and not reference_wav.exists():
                upstream = _missing_reference_report("upstream", upstream_command, upstream_wav, reference_wav)
            elif not args.upstream_root:
                upstream = _side_report(
                    "upstream",
                    "unavailable",
                    upstream_command,
                    upstream_wav,
                    reason="missing_upstream_root",
                    detail="Pass --upstream-root pointing at an Irodori-TTS checkout to execute the upstream side.",
                )
            else:
                upstream.update(_run(upstream_command, cwd=Path(args.upstream_root).expanduser().resolve(), timeout_seconds=args.timeout_seconds))
                upstream["availability"] = {"state": upstream["status"], "reason": None, "detail": None}
                upstream["audio"] = wav_properties(upstream_wav)
        if args.run_mlx:
            if reference_wav and not reference_wav.exists():
                mlx = _missing_reference_report("mlx", mlx_command, mlx_wav, reference_wav)
            elif not args.mlx_weights:
                mlx = _side_report(
                    "mlx",
                    "unavailable",
                    mlx_command,
                    mlx_wav,
                    reason="missing_mlx_weights",
                    detail="Pass --mlx-weights, and usually --mlx-model-config-json, to execute the MLX side.",
                )
            else:
                mlx.update(_run(mlx_command, cwd=ROOT, timeout_seconds=args.timeout_seconds))
                mlx["availability"] = {"state": mlx["status"], "reason": None, "detail": None}
                mlx["audio"] = wav_properties(mlx_wav)
                if mlx_metadata.exists():
                    mlx["metadata"] = json.loads(mlx_metadata.read_text(encoding="utf-8"))
                    mlx["intermediates"] = _mlx_intermediates_from_metadata(mlx["metadata"])

    return {
        "schema_version": SCHEMA_VERSION,
        "created_by": "scripts/run_upstream_parity.py",
        "environment": {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd()},
        "report_status": _report_status(upstream, mlx),
        "scenario": asdict(scenario),
        "metadata_axes": _metadata_axes(scenario),
        "artifacts": {"output_dir": str(output_dir), "commit_note": "Generated WAVs, upstream checkouts, and checkpoint caches must stay outside git."},
        "upstream": upstream,
        "mlx": mlx,
        "comparison": _classify(upstream, mlx),
        "deferred_scope": [
            "full VoiceDesign real-checkpoint baseline matrix",
            "full v3 real-checkpoint baseline matrix",
            "perceptual audio metrics that require heavyweight DSP dependencies",
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
    parser.add_argument("--codec-runtime-mode", choices=("mlx",))
    parser.add_argument("--codec-path")
    parser.add_argument("--codec-artifact-dir")
    parser.add_argument("--codec-artifact-repo")
    parser.add_argument("--codec-artifact-revision")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--text-tokenizer-repo")
    parser.add_argument("--caption-tokenizer-repo")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--caption-max-length", type=int)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--cfg-guidance-mode", choices=("independent", "joint", "alternating", "reduced"), default="independent")
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
