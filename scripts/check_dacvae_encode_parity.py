#!/usr/bin/env python3
"""Compare upstream PyTorch and MLX DACVAE encode outputs for fixed audio."""

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
class EncodeParityTolerances:
    max_abs: float = 1e-3
    mean_abs: float = 2e-4
    rmse: float = 5e-4
    min_cosine: float = 0.999


def _latent_stats(latents: np.ndarray) -> dict[str, Any]:
    latents = np.asarray(latents, dtype=np.float32)
    return {
        "shape": [int(dim) for dim in latents.shape],
        "dtype": "float32",
        "finite": bool(np.isfinite(latents).all()),
        "min": float(latents.min()) if latents.size else 0.0,
        "max": float(latents.max()) if latents.size else 0.0,
        "mean": float(latents.mean()) if latents.size else 0.0,
        "std": float(latents.std()) if latents.size else 0.0,
    }


def _audio_stats(path: str | Path) -> dict[str, Any]:
    samples, sample_rate = _load_audio_numpy(path)
    return {
        "path": str(Path(path).expanduser()),
        "sample_rate": int(sample_rate),
        "samples": int(samples.shape[0]),
        "duration_seconds": float(samples.shape[0] / sample_rate) if sample_rate else 0.0,
        "peak_abs": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) if samples.size else 0.0,
        "finite": bool(np.isfinite(samples).all()),
    }


def compare_latents(
    upstream: np.ndarray,
    mlx: np.ndarray,
    *,
    hop_length: int,
    tolerances: EncodeParityTolerances,
) -> dict[str, Any]:
    upstream = np.asarray(upstream, dtype=np.float32)
    mlx = np.asarray(mlx, dtype=np.float32)
    shape_match = upstream.shape == mlx.shape
    batch_match = upstream.ndim == 3 and mlx.ndim == 3 and int(upstream.shape[0]) == int(mlx.shape[0])
    latent_dim_match = upstream.ndim == 3 and mlx.ndim == 3 and int(upstream.shape[2]) == int(mlx.shape[2])
    latent_steps_match = upstream.ndim == 3 and mlx.ndim == 3 and int(upstream.shape[1]) == int(mlx.shape[1])
    n = min(int(upstream.size), int(mlx.size))
    if n == 0:
        diff = np.array([], dtype=np.float32)
        cosine = 1.0 if shape_match else 0.0
    else:
        upstream_cmp = upstream.reshape(-1)[:n]
        mlx_cmp = mlx.reshape(-1)[:n]
        diff = mlx_cmp - upstream_cmp
        denom = float(np.linalg.norm(upstream_cmp) * np.linalg.norm(mlx_cmp))
        if denom == 0.0:
            cosine = 1.0 if np.linalg.norm(diff) == 0.0 else 0.0
        else:
            cosine = float(np.dot(upstream_cmp, mlx_cmp) / denom)
    metrics = {
        "shape_match": bool(shape_match),
        "compared_values": int(n),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
        "cosine": cosine,
        "upstream": _latent_stats(upstream),
        "mlx": _latent_stats(mlx),
    }
    length_contract = {
        "hop_length": int(hop_length),
        "latent_steps_upstream": int(upstream.shape[1]) if upstream.ndim == 3 else None,
        "latent_steps_mlx": int(mlx.shape[1]) if mlx.ndim == 3 else None,
        "speaker_mask_true_count_upstream": int(upstream.shape[1]) if upstream.ndim == 3 else None,
        "speaker_mask_true_count_mlx": int(mlx.shape[1]) if mlx.ndim == 3 else None,
    }
    checks = {
        "shape": bool(shape_match),
        "batch": bool(batch_match),
        "latent_steps": bool(latent_steps_match),
        "latent_dim": bool(latent_dim_match),
        "finite": bool(metrics["upstream"]["finite"] and metrics["mlx"]["finite"]),
        "max_abs": metrics["max_abs"] <= tolerances.max_abs,
        "mean_abs": metrics["mean_abs"] <= tolerances.mean_abs,
        "rmse": metrics["rmse"] <= tolerances.rmse,
        "cosine": metrics["cosine"] >= tolerances.min_cosine,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "tolerances": asdict(tolerances),
        "checks": checks,
        "metrics": metrics,
        "length_contract": length_contract,
    }


def encode_pair(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(args.audio_wav).expanduser()
    normalize_db = None if args.normalize_db is None else float(args.normalize_db)
    ensure_max = bool(args.ensure_max)

    upstream_config = DACVAEBridgeConfig(
        codec_repo=args.codec_repo,
        codec_device=args.codec_device,
        runtime_mode="persistent",
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
        normalize_db=normalize_db,
    )
    mlx_config = DACVAEBridgeConfig(
        codec_repo=args.codec_repo,
        codec_path=str(Path(args.codec_path).expanduser()),
        codec_device=args.codec_device,
        runtime_mode="mlx",
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
        normalize_db=normalize_db,
    )
    upstream_bridge = PyTorchDACVAEBridge(config=upstream_config)
    mlx_bridge = MLXDACVAEBridge(config=mlx_config)
    if int(upstream_bridge.sample_rate) != int(mlx_bridge.sample_rate):
        raise ValueError(f"sample_rate mismatch: upstream={upstream_bridge.sample_rate}, mlx={mlx_bridge.sample_rate}")
    if int(upstream_bridge.hop_length) != int(mlx_bridge.hop_length):
        raise ValueError(f"hop_length mismatch: upstream={upstream_bridge.hop_length}, mlx={mlx_bridge.hop_length}")
    if int(upstream_bridge.latent_dim) != int(mlx_bridge.latent_dim):
        raise ValueError(f"latent_dim mismatch: upstream={upstream_bridge.latent_dim}, mlx={mlx_bridge.latent_dim}")

    upstream_latents = upstream_bridge.encode_reference(
        audio_path,
        max_seconds=args.max_seconds,
        normalize_db=normalize_db,
        ensure_max=ensure_max,
    )
    mlx_latents = mlx_bridge.encode_reference(
        audio_path,
        max_seconds=args.max_seconds,
        normalize_db=normalize_db,
        ensure_max=ensure_max,
    )
    upstream_np = np.asarray(upstream_latents, dtype=np.float32)
    mlx_np = np.asarray(mlx_latents, dtype=np.float32)
    upstream_path = output_dir / "upstream-encode-latents.npy"
    mlx_path = output_dir / "mlx-encode-latents.npy"
    np.save(upstream_path, upstream_np)
    np.save(mlx_path, mlx_np)
    tolerances = EncodeParityTolerances(
        max_abs=float(args.max_abs_tolerance),
        mean_abs=float(args.mean_abs_tolerance),
        rmse=float(args.rmse_tolerance),
        min_cosine=float(args.min_cosine),
    )
    comparison = compare_latents(
        upstream_np,
        mlx_np,
        hop_length=int(upstream_bridge.hop_length),
        tolerances=tolerances,
    )
    return {
        "schema_version": 1,
        "source_issue": "https://github.com/t0yohei/Irodori-TTS-MLX/issues/115",
        "parent_epic": "https://github.com/t0yohei/Irodori-TTS-MLX/issues/123",
        "audio": _audio_stats(audio_path),
        "codec": {
            "repo": args.codec_repo,
            "device": args.codec_device,
            "mlx_codec_path": str(Path(args.codec_path).expanduser()),
            "sample_rate": int(upstream_bridge.sample_rate),
            "hop_length": int(upstream_bridge.hop_length),
            "latent_dim": int(upstream_bridge.latent_dim),
            "watermark": "disabled",
            "normalize_db": normalize_db,
            "ensure_max": ensure_max,
            "max_seconds": args.max_seconds,
        },
        "outputs": {
            "upstream_latents_npy": str(upstream_path),
            "mlx_latents_npy": str(mlx_path),
        },
        "comparison": comparison,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-wav", required=True, help="Fixed reference audio WAV for DACVAE encode.")
    parser.add_argument("--codec-path", required=True, help="Converted MLX DACVAE codec artifact .npz with encode tensors.")
    parser.add_argument("--output-dir", required=True, help="Directory for encoded latent fixtures and parity report.")
    parser.add_argument("--report-json", help="Report path. Defaults to <output-dir>/dacvae-encode-parity.json.")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--normalize-db", type=float, default=None)
    parser.add_argument("--ensure-max", action="store_true")
    parser.add_argument("--max-abs-tolerance", type=float, default=EncodeParityTolerances.max_abs)
    parser.add_argument("--mean-abs-tolerance", type=float, default=EncodeParityTolerances.mean_abs)
    parser.add_argument("--rmse-tolerance", type=float, default=EncodeParityTolerances.rmse)
    parser.add_argument("--min-cosine", type=float, default=EncodeParityTolerances.min_cosine)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = encode_pair(args)
    except Exception as exc:
        print(f"DACVAE encode parity failed before comparison: {exc}", file=sys.stderr)
        return 2
    report_path = (
        Path(args.report_json).expanduser()
        if args.report_json
        else Path(args.output_dir).expanduser() / "dacvae-encode-parity.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["comparison"]["status"], "report": str(report_path)}, sort_keys=True))
    return 0 if report["comparison"]["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
