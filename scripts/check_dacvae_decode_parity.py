#!/usr/bin/env python3
"""Compare upstream PyTorch and MLX DACVAE decode outputs for fixed latents."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from irodori_mlx.runtime import (  # noqa: E402
    DACVAEBridgeConfig,
    MLXDACVAEBridge,
    PyTorchDACVAEBridge,
    _load_audio_numpy,
)


@dataclass(frozen=True)
class DecodeParityTolerances:
    max_abs: float = 5e-3
    mean_abs: float = 1e-3
    rmse: float = 2e-3
    min_cosine: float = 0.999


def _load_latents(path: str | Path):
    import mlx.core as mx

    latents = np.load(Path(path).expanduser()).astype("float32", copy=False)
    if latents.ndim != 3:
        raise ValueError(f"Expected latents shaped (B,T,D), got {latents.shape}: {path}")
    if int(latents.shape[0]) != 1:
        raise ValueError(f"Decode parity currently expects batch size 1, got {latents.shape[0]}: {path}")
    return mx.array(latents)


def _audio_stats(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float32)
    return {
        "sample_rate": int(sample_rate),
        "samples": int(samples.shape[0]),
        "dtype": "float32",
        "min": float(samples.min()) if samples.size else 0.0,
        "max": float(samples.max()) if samples.size else 0.0,
        "peak_abs": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0,
        "finite": bool(np.isfinite(samples).all()),
        "within_unit_range": bool(samples.size == 0 or np.max(np.abs(samples)) <= 1.0001),
    }


def compare_audio(
    upstream: np.ndarray,
    mlx: np.ndarray,
    *,
    sample_rate: int,
    tolerances: DecodeParityTolerances,
) -> dict[str, Any]:
    upstream = np.asarray(upstream, dtype=np.float32).reshape(-1)
    mlx = np.asarray(mlx, dtype=np.float32).reshape(-1)
    shape_match = upstream.shape == mlx.shape
    n = min(int(upstream.shape[0]), int(mlx.shape[0]))
    if n == 0:
        diff = np.array([], dtype=np.float32)
        cosine = 1.0 if shape_match else 0.0
    else:
        upstream_cmp = upstream[:n]
        mlx_cmp = mlx[:n]
        diff = mlx_cmp - upstream_cmp
        denom = float(np.linalg.norm(upstream_cmp) * np.linalg.norm(mlx_cmp))
        if denom == 0.0:
            cosine = 1.0 if np.linalg.norm(diff) == 0.0 else 0.0
        else:
            cosine = float(np.dot(upstream_cmp, mlx_cmp) / denom)
    metrics = {
        "shape_match": bool(shape_match),
        "compared_samples": int(n),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
        "cosine": cosine,
        "upstream": _audio_stats(upstream, sample_rate),
        "mlx": _audio_stats(mlx, sample_rate),
    }
    checks = {
        "shape": bool(shape_match),
        "finite": bool(metrics["upstream"]["finite"] and metrics["mlx"]["finite"]),
        "range": bool(metrics["upstream"]["within_unit_range"] and metrics["mlx"]["within_unit_range"]),
        "max_abs": metrics["max_abs"] <= tolerances.max_abs,
        "mean_abs": metrics["mean_abs"] <= tolerances.mean_abs,
        "rmse": metrics["rmse"] <= tolerances.rmse,
        "cosine": metrics["cosine"] >= tolerances.min_cosine,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "sample_rate": int(sample_rate),
        "tolerances": asdict(tolerances),
        "checks": checks,
        "metrics": metrics,
    }


def decode_pair(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latents = _load_latents(args.latents_npy)
    max_samples = None if args.max_samples is None else int(args.max_samples)

    upstream_config = DACVAEBridgeConfig(
        codec_repo=args.codec_repo,
        codec_device=args.codec_device,
        runtime_mode="persistent",
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
        normalize_db=None,
    )
    mlx_config = DACVAEBridgeConfig(
        codec_repo=args.codec_repo,
        codec_path=str(Path(args.codec_path).expanduser()),
        codec_device=args.codec_device,
        runtime_mode="mlx-decode",
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
        normalize_db=None,
    )
    upstream_bridge = PyTorchDACVAEBridge(config=upstream_config)
    mlx_bridge = MLXDACVAEBridge(config=mlx_config, require_encode=False)
    if int(upstream_bridge.sample_rate) != int(mlx_bridge.sample_rate):
        raise ValueError(f"sample_rate mismatch: upstream={upstream_bridge.sample_rate}, mlx={mlx_bridge.sample_rate}")
    if int(upstream_bridge.hop_length) != int(mlx_bridge.hop_length):
        raise ValueError(f"hop_length mismatch: upstream={upstream_bridge.hop_length}, mlx={mlx_bridge.hop_length}")
    if int(upstream_bridge.latent_dim) != int(mlx_bridge.latent_dim):
        raise ValueError(f"latent_dim mismatch: upstream={upstream_bridge.latent_dim}, mlx={mlx_bridge.latent_dim}")

    upstream_wav = output_dir / "upstream-decode.wav"
    mlx_wav = output_dir / "mlx-decode.wav"
    upstream_bridge.decode_to_wav(latents, upstream_wav, max_samples=max_samples)
    mlx_bridge.decode_to_wav(latents, mlx_wav, max_samples=max_samples)
    upstream_audio, upstream_sr = _load_audio_numpy(upstream_wav)
    mlx_audio, mlx_sr = _load_audio_numpy(mlx_wav)
    if int(upstream_sr) != int(mlx_sr):
        raise ValueError(f"decoded WAV sample_rate mismatch: upstream={upstream_sr}, mlx={mlx_sr}")
    tolerances = DecodeParityTolerances(
        max_abs=float(args.max_abs_tolerance),
        mean_abs=float(args.mean_abs_tolerance),
        rmse=float(args.rmse_tolerance),
        min_cosine=float(args.min_cosine),
    )
    comparison = compare_audio(upstream_audio, mlx_audio, sample_rate=int(upstream_sr), tolerances=tolerances)
    return {
        "schema_version": 1,
        "source_issue": "https://github.com/t0yohei/Irodori-TTS-MLX/issues/113",
        "parent_epic": "https://github.com/t0yohei/Irodori-TTS-MLX/issues/123",
        "latents": {
            "path": str(Path(args.latents_npy).expanduser()),
            "shape": [int(dim) for dim in latents.shape],
            "dtype": "float32",
        },
        "codec": {
            "repo": args.codec_repo,
            "device": args.codec_device,
            "mlx_codec_path": str(Path(args.codec_path).expanduser()),
            "sample_rate": int(upstream_bridge.sample_rate),
            "hop_length": int(upstream_bridge.hop_length),
            "latent_dim": int(upstream_bridge.latent_dim),
            "watermark": "disabled",
        },
        "outputs": {
            "upstream_wav": str(upstream_wav),
            "mlx_wav": str(mlx_wav),
        },
        "comparison": comparison,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-npy", required=True, help="Fixed runtime-layout latents shaped (1,T,32).")
    parser.add_argument("--codec-path", required=True, help="Converted MLX DACVAE decode artifact .npz.")
    parser.add_argument("--output-dir", required=True, help="Directory for decoded WAVs and parity report.")
    parser.add_argument("--report-json", help="Report path. Defaults to <output-dir>/dacvae-decode-parity.json.")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-abs-tolerance", type=float, default=DecodeParityTolerances.max_abs)
    parser.add_argument("--mean-abs-tolerance", type=float, default=DecodeParityTolerances.mean_abs)
    parser.add_argument("--rmse-tolerance", type=float, default=DecodeParityTolerances.rmse)
    parser.add_argument("--min-cosine", type=float, default=DecodeParityTolerances.min_cosine)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = decode_pair(args)
    except Exception as exc:
        print(f"DACVAE decode parity failed before comparison: {exc}", file=sys.stderr)
        return 2
    report_path = (
        Path(args.report_json).expanduser()
        if args.report_json
        else Path(args.output_dir).expanduser() / "dacvae-decode-parity.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["comparison"]["status"], "report": str(report_path)}, sort_keys=True))
    return 0 if report["comparison"]["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
