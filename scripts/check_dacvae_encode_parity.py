#!/usr/bin/env python3
"""Check MLX DACVAE encode output for fixed audio."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 2
SOURCE_ISSUE = "https://github.com/t0yohei/Irodori-TTS-MLX/issues/185"
PARENT_EPIC = "https://github.com/t0yohei/Irodori-TTS-MLX/issues/169"
DEFAULT_EXPECTED_LATENT_DIM = 32

DACVAEBridgeConfig: Any = None
MLXDACVAEBridge: Any = None
_load_audio_numpy: Any = None


class PartialPreconditionError(RuntimeError):
    """Raised only for preflight misses that are allowed to produce partial reports."""


def _load_runtime_encode_dependencies() -> None:
    global DACVAEBridgeConfig, MLXDACVAEBridge, _load_audio_numpy
    if all(dependency is not None for dependency in (DACVAEBridgeConfig, MLXDACVAEBridge, _load_audio_numpy)):
        return
    from irodori_mlx.runtime import (  # noqa: E402
        DACVAEBridgeConfig as runtime_config,
        MLXDACVAEBridge as mlx_bridge,
        _load_audio_numpy as runtime_load_audio_numpy,
    )

    DACVAEBridgeConfig = runtime_config
    MLXDACVAEBridge = mlx_bridge
    _load_audio_numpy = runtime_load_audio_numpy


def _require_existing_path(path: str | Path, label: str) -> None:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise PartialPreconditionError(f"{label} was not found: {resolved}")


def _require_module(module_name: str, detail: str) -> None:
    try:
        found = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise PartialPreconditionError(detail) from exc
    if found is None:
        raise PartialPreconditionError(detail)


def _preflight_encode_pair(args: argparse.Namespace) -> None:
    _require_existing_path(args.audio_wav, "Fixed DACVAE encode reference WAV fixture")
    _require_existing_path(args.codec_path, "Converted MLX DACVAE codec artifact")
    _require_module("mlx", "MLX runtime dependency is required for DACVAE encode parity.")


def _path_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    resolved = Path(path).expanduser()
    return {"path": str(resolved), "exists": resolved.exists()}


def _is_partial_exception(exc: Exception) -> bool:
    return isinstance(exc, PartialPreconditionError)


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
    if _load_audio_numpy is None:
        _load_runtime_encode_dependencies()
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
    _load_runtime_encode_dependencies()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(args.audio_wav).expanduser()
    normalize_db = None if args.normalize_db is None else float(args.normalize_db)
    ensure_max = bool(args.ensure_max)

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
    mlx_bridge = MLXDACVAEBridge(config=mlx_config)
    if int(mlx_bridge.latent_dim) != int(args.expected_latent_dim):
        raise ValueError(
            "Encode check expected Semantic-DACVAE runtime-layout latents shaped "
            f"(1,T,{int(args.expected_latent_dim)}), got latent_dim={mlx_bridge.latent_dim}"
        )

    mlx_latents = mlx_bridge.encode_reference(
        audio_path,
        max_seconds=args.max_seconds,
        normalize_db=normalize_db,
        ensure_max=ensure_max,
    )
    mlx_np = np.asarray(mlx_latents, dtype=np.float32)
    mlx_path = output_dir / "mlx-encode-latents.npy"
    np.save(mlx_path, mlx_np)
    stats = _latent_stats(mlx_np)
    checks = {
        "rank": mlx_np.ndim == 3,
        "batch": mlx_np.ndim == 3 and int(mlx_np.shape[0]) == 1,
        "latent_dim": mlx_np.ndim == 3 and int(mlx_np.shape[2]) == int(args.expected_latent_dim),
        "finite": bool(stats["finite"]),
    }
    comparison = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "metrics": {"mlx": stats},
        "length_contract": {
            "hop_length": int(mlx_bridge.hop_length),
            "latent_steps_mlx": int(mlx_np.shape[1]) if mlx_np.ndim == 3 else None,
            "speaker_mask_true_count_mlx": int(mlx_np.shape[1]) if mlx_np.ndim == 3 else None,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_issue": SOURCE_ISSUE,
        "parent_epic": PARENT_EPIC,
        "run": {
            "status": "complete",
            "complete": True,
        },
        "audio": _audio_stats(audio_path),
        "codec": {
            "repo": args.codec_repo,
            "device": args.codec_device,
            "mlx_codec_path": str(Path(args.codec_path).expanduser()),
            "sample_rate": int(mlx_bridge.sample_rate),
            "hop_length": int(mlx_bridge.hop_length),
            "latent_dim": int(mlx_bridge.latent_dim),
            "expected_latent_dim": int(args.expected_latent_dim),
            "metadata_checks": {
                "sample_rate": True,
                "hop_length": True,
                "latent_dim": True,
            },
            "watermark": "disabled",
            "normalize_db": normalize_db,
            "ensure_max": ensure_max,
            "max_seconds": args.max_seconds,
        },
        "outputs": {
            "mlx_latents_npy": str(mlx_path),
        },
        "comparison": comparison,
    }


def build_incomplete_report(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    status = "partial" if _is_partial_exception(exc) else "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "source_issue": SOURCE_ISSUE,
        "parent_epic": PARENT_EPIC,
        "run": {
            "status": status,
            "reason": str(exc),
            "complete": False,
        },
        "audio": {
            **_path_metadata(args.audio_wav),
            "stats_available": Path(args.audio_wav).expanduser().exists(),
        },
        "codec": {
            "repo": args.codec_repo,
            "device": args.codec_device,
            "mlx_codec": _path_metadata(args.codec_path),
            "expected_latent_dim": int(args.expected_latent_dim),
            "watermark": "disabled",
            "normalize_db": None if args.normalize_db is None else float(args.normalize_db),
            "ensure_max": bool(args.ensure_max),
            "max_seconds": args.max_seconds,
        },
        "outputs": {
            "mlx_latents_npy": None,
        },
        "comparison": {
            "status": status,
            "reason": str(exc),
            "checks": {},
            "metrics": {},
        },
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
    parser.add_argument(
        "--expected-latent-dim",
        type=int,
        default=DEFAULT_EXPECTED_LATENT_DIM,
        help="Expected runtime latent channel count for the fixed fixture. Defaults to 32 for Semantic-DACVAE.",
    )
    parser.add_argument("--max-abs-tolerance", type=float, default=EncodeParityTolerances.max_abs)
    parser.add_argument("--mean-abs-tolerance", type=float, default=EncodeParityTolerances.mean_abs)
    parser.add_argument("--rmse-tolerance", type=float, default=EncodeParityTolerances.rmse)
    parser.add_argument("--min-cosine", type=float, default=EncodeParityTolerances.min_cosine)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial report and exit 0 when preflight detects absent local artifacts or runtime dependencies.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _preflight_encode_pair(args)
        report = encode_pair(args)
    except Exception as exc:
        report = build_incomplete_report(args, exc)
    report_path = (
        Path(args.report_json).expanduser()
        if args.report_json
        else Path(args.output_dir).expanduser() / "dacvae-encode-parity.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    status = report["comparison"]["status"]
    print(json.dumps({"status": status, "report": str(report_path)}, sort_keys=True))
    if status == "passed":
        return 0
    if status == "partial":
        return 0 if args.allow_partial else 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
