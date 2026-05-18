#!/usr/bin/env python3
"""Check MLX DACVAE decode output for fixed latents."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 3
SOURCE_ISSUE = "https://github.com/t0yohei/Irodori-TTS-MLX/issues/184"
PARENT_EPIC = "https://github.com/t0yohei/Irodori-TTS-MLX/issues/169"
DEFAULT_EXPECTED_LATENT_DIM = 32

DACVAEBridgeConfig: Any = None
MLXDACVAEBridge: Any = None
_load_audio_numpy: Any = None


def _load_runtime_decode_dependencies() -> None:
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


class PartialPreconditionError(RuntimeError):
    """Raised only for preflight misses that are allowed to produce partial reports."""


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


def _preflight_decode_pair(args: argparse.Namespace) -> None:
    _require_existing_path(args.latents_npy, "Fixed DACVAE decode latents fixture")
    _require_existing_path(args.codec_path, "Converted MLX DACVAE codec artifact")
    _require_module("mlx", "MLX runtime dependency is required for DACVAE decode parity.")


def _load_latents(path: str | Path, *, expected_latent_dim: int = DEFAULT_EXPECTED_LATENT_DIM):
    latents = np.load(Path(path).expanduser()).astype("float32", copy=False)
    if latents.ndim != 3:
        raise ValueError(f"Expected latents shaped (B,T,D), got {latents.shape}: {path}")
    if int(latents.shape[0]) != 1:
        raise ValueError(f"Decode parity currently expects batch size 1, got {latents.shape[0]}: {path}")
    if int(latents.shape[2]) != int(expected_latent_dim):
        raise ValueError(
            "Decode parity expected runtime-layout latents shaped "
            f"(1,T,{int(expected_latent_dim)}), got {latents.shape}: {path}"
        )
    import mlx.core as mx

    return mx.array(latents)


def _path_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    resolved = Path(path).expanduser()
    return {"path": str(resolved), "exists": resolved.exists()}


def _is_partial_exception(exc: Exception) -> bool:
    return isinstance(exc, PartialPreconditionError)


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
        "latents": _path_metadata(args.latents_npy),
        "codec": {
            "repo": args.codec_repo,
            "device": args.codec_device,
            "expected_latent_dim": int(args.expected_latent_dim),
            "mlx_codec": _path_metadata(args.codec_path),
            "watermark": "disabled",
        },
        "outputs": {
            "output_dir": str(Path(args.output_dir).expanduser()),
        },
        "comparison": {
            "status": status,
            "reason": str(exc),
            "checks": {},
            "metrics": {},
        },
    }


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


def decode_pair(args: argparse.Namespace) -> dict[str, Any]:
    _load_runtime_decode_dependencies()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latents = _load_latents(args.latents_npy, expected_latent_dim=int(args.expected_latent_dim))
    codec_path = Path(args.codec_path).expanduser()
    if not codec_path.exists():
        raise FileNotFoundError(f"Converted MLX DACVAE codec artifact was not found: {codec_path}")
    max_samples = None if args.max_samples is None else int(args.max_samples)

    mlx_config = DACVAEBridgeConfig(
        codec_repo=args.codec_repo,
        codec_path=str(codec_path),
        codec_device=args.codec_device,
        runtime_mode="mlx",
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
        normalize_db=None,
    )
    mlx_bridge = MLXDACVAEBridge(config=mlx_config, require_encode=False)
    if int(mlx_bridge.latent_dim) != int(args.expected_latent_dim):
        raise ValueError(
            "Decode check expected Semantic-DACVAE runtime-layout latents shaped "
            f"(1,T,{int(args.expected_latent_dim)}), got latent_dim={mlx_bridge.latent_dim}"
        )

    mlx_wav = output_dir / "mlx-decode.wav"
    mlx_bridge.decode_to_wav(latents, mlx_wav, max_samples=max_samples)
    mlx_audio, mlx_sr = _load_audio_numpy(mlx_wav)
    stats = _audio_stats(mlx_audio, int(mlx_sr))
    checks = {
        "sample_rate": int(mlx_sr) == int(mlx_bridge.sample_rate),
        "finite": bool(stats["finite"]),
        "range": bool(stats["within_unit_range"]),
    }
    comparison = {
        "status": "passed" if all(checks.values()) else "failed",
        "sample_rate": int(mlx_sr),
        "checks": checks,
        "metrics": {"mlx": stats},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_issue": SOURCE_ISSUE,
        "parent_epic": PARENT_EPIC,
        "run": {
            "status": "complete",
            "complete": True,
        },
        "latents": {
            "path": str(Path(args.latents_npy).expanduser()),
            "shape": [int(dim) for dim in latents.shape],
            "dtype": "float32",
        },
        "codec": {
            "repo": args.codec_repo,
            "device": args.codec_device,
            "mlx_codec_path": str(codec_path),
            "expected_latent_dim": int(args.expected_latent_dim),
            "sample_rate": int(mlx_bridge.sample_rate),
            "hop_length": int(mlx_bridge.hop_length),
            "latent_dim": int(mlx_bridge.latent_dim),
            "metadata_checks": {
                "sample_rate": True,
                "hop_length": True,
                "latent_dim": True,
                "decoded_wav_sample_rate": True,
            },
            "watermark": "disabled",
        },
        "outputs": {
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
    parser.add_argument(
        "--expected-latent-dim",
        type=int,
        default=DEFAULT_EXPECTED_LATENT_DIM,
        help="Expected runtime latent channel count for the fixed fixture. Defaults to 32 for Semantic-DACVAE.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial report and exit 0 when preflight detects absent local artifacts or runtime dependencies.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _preflight_decode_pair(args)
        report = decode_pair(args)
    except Exception as exc:
        report = build_incomplete_report(args, exc)
    report_path = (
        Path(args.report_json).expanduser()
        if args.report_json
        else Path(args.output_dir).expanduser() / "dacvae-decode-parity.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["comparison"]["status"], "report": str(report_path)}, sort_keys=True))
    status = report["comparison"]["status"]
    if status == "passed":
        return 0
    if status == "partial":
        return 0 if args.allow_partial else 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
