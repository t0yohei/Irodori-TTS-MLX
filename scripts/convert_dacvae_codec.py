#!/usr/bin/env python3
"""Inspect Semantic-DACVAE checkpoints for the MLX codec artifact boundary.

The current public runtime can exercise MLX encode/decode with the small
projection fixture contract, but it does not yet implement the real DACVAE
conv/residual/VAEBottleneck stack. This script makes that blocker explicit and
machine-readable so local conversion attempts do not silently emit unusable
codec artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_SOURCE_FILE = "weights.pth"
EXPECTED_SAMPLE_RATE = 48000
EXPECTED_HOP_LENGTH = 512
EXPECTED_LATENT_DIM = 32
REQUIRED_ENCODE_GROUPS = {
    "encoder": ("encoder.",),
    "quantizer_in_proj": ("quantizer.in_proj.",),
}
REQUIRED_DECODE_GROUPS = {
    "quantizer_out_proj": ("quantizer.out_proj.",),
    "decoder": ("decoder.",),
}
BLOCKER = (
    "Full Semantic-DACVAE MLX conversion is blocked because the public runtime "
    "does not yet include MLX implementations of the DACVAE encoder conv stack, "
    "VAEBottleneck mean projection, decoder stack, or watermark-bypass decode path. "
    "The current dacvae-codec.npz contract only supports linear fixture tensors "
    "(encode_basis/decode_basis) for routing and parity harness smoke tests."
)


class DACVAEConversionError(RuntimeError):
    """Raised when the DACVAE checkpoint cannot be inspected."""


def _state_dict_from_loaded(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "codec", "net"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                return value
        return obj
    raise DACVAEConversionError("PyTorch checkpoint did not contain a mapping state_dict")


def load_state_dict_keys(source: str | Path) -> list[str]:
    """Load a local PyTorch DACVAE weights.pth and return sorted state_dict keys."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps.
        raise DACVAEConversionError("torch is required to inspect DACVAE weights.pth") from exc

    try:
        loaded = torch.load(str(Path(source).expanduser()), map_location="cpu")
    except Exception as exc:  # pragma: no cover - depends on user checkpoint.
        raise DACVAEConversionError(f"could not load DACVAE checkpoint {source}: {exc}") from exc
    state_dict = _state_dict_from_loaded(loaded)
    return sorted(str(key) for key in state_dict.keys())


def _group_presence(keys: Iterable[str], groups: Mapping[str, tuple[str, ...]]) -> dict[str, bool]:
    key_list = list(keys)
    return {
        group: any(any(key.startswith(prefix) for prefix in prefixes) for key in key_list)
        for group, prefixes in groups.items()
    }


def build_blocked_conversion_report(
    *,
    source: str | Path,
    output: str | Path,
    state_keys: Iterable[str],
) -> dict[str, Any]:
    """Build the current explicit blocker report for real Semantic-DACVAE conversion."""

    keys = sorted(str(key) for key in state_keys)
    encode_groups = _group_presence(keys, REQUIRED_ENCODE_GROUPS)
    decode_groups = _group_presence(keys, REQUIRED_DECODE_GROUPS)
    return {
        "status": "blocked",
        "source": str(Path(source).expanduser()),
        "requested_output": str(Path(output).expanduser()),
        "codec": {
            "repo_id": CODEC_REPO,
            "source_file": CODEC_SOURCE_FILE,
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "hop_length": EXPECTED_HOP_LENGTH,
            "latent_dim": EXPECTED_LATENT_DIM,
        },
        "state_dict": {
            "key_count": len(keys),
            "encode_groups_present": encode_groups,
            "decode_groups_present": decode_groups,
            "encode_contract_present": all(encode_groups.values()),
            "decode_contract_present": all(decode_groups.values()),
        },
        "artifact": {
            "format": "irodori-tts-mlx-dacvae-codec",
            "format_version": "0.2",
            "would_require": [
                "semantic_encoder_manifest_json",
                "semantic encoder conv/residual tensors",
                "quantizer.in_proj mean/scale tensors",
                "quantizer.out_proj tensors",
                "semantic decoder conv/residual tensors",
                "watermark-bypass metadata",
            ],
            "not_written": True,
        },
        "blocker": BLOCKER,
        "next_steps": [
            "Implement MLX DACVAE conv/residual/VAEBottleneck modules.",
            "Map the inspected PyTorch logical tensors into that module layout.",
            "Validate fixed encode/decode parity fixtures before publishing artifacts.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Semantic-DACVAE weights.pth and report the current MLX conversion blocker."
    )
    parser.add_argument("source", help="Local Semantic-DACVAE weights.pth checkpoint.")
    parser.add_argument("output", help="Requested output dacvae-codec.npz path. It is not written while blocked.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Print the blocker report as JSON.")
    parser.add_argument("--report-json", help="Optional path to write the blocker report JSON.")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Return success after writing/printing the blocker report instead of failing conversion.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        keys = load_state_dict_keys(args.source)
        report = build_blocked_conversion_report(source=args.source, output=args.output, state_keys=keys)
    except DACVAEConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.report_json:
        Path(args.report_json).expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Semantic-DACVAE MLX conversion blocked: {BLOCKER}")
    return 0 if args.inspect_only else 2


def cli_main() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(cli_main())
