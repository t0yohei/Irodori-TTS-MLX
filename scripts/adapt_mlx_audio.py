#!/usr/bin/env python3
"""Adapt unquantized mlx-audio Irodori artifacts into the hosted MLX weights layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from irodori_mlx.mlx_audio_adapter import MlxAudioAdapterError, adapt_mlx_audio_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an unquantized mlx-audio Irodori directory or Hugging Face snapshot "
            "(config.json + model.safetensors) into the Irodori-TTS-MLX hosted weights layout."
        )
    )
    parser.add_argument("source_dir", help="Local mlx-audio artifact directory or downloaded HF snapshot.")
    parser.add_argument("output_dir", help="Output hosted-layout directory to create/update.")
    parser.add_argument("--source-repo", help="Optional upstream Hugging Face repo id for provenance.")
    parser.add_argument("--source-revision", help="Optional upstream revision/commit for provenance.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = adapt_mlx_audio_layout(
        Path(args.source_dir),
        Path(args.output_dir),
        source_repo=args.source_repo,
        source_revision=args.source_revision,
    )
    print(f"wrote hosted layout: {result.output_dir}")
    print(f"checkpoint_family: {result.checkpoint_family}")
    print(f"weights: {result.weights_path}")
    print(f"manifest: {result.manifest_path}")
    return 0


def cli_main() -> int:
    try:
        return main()
    except MlxAudioAdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
