#!/usr/bin/env python3
"""Generate a WAV with MLX RF-DiT latents and the PyTorch DACVAE bridge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from irodori_mlx.runtime import (  # noqa: E402
    DACVAEBridgeConfig,
    GenerationRequest,
    MLXDACVAERuntime,
    MLXRuntimeConfig,
    iter_messages,
    load_model_config_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a WAV using MLX Irodori-TTS RF-DiT and upstream/PyTorch DACVAE."
    )
    parser.add_argument("--weights", required=True, help="Converted MLX .npz RF-DiT weights.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--text", required=True, help="Text prompt to synthesize.")
    parser.add_argument("--reference-wav", help="Speaker/reference audio path for DACVAE encoding.")
    parser.add_argument("--no-reference", action="store_true", help="Use an unconditional speaker path instead of reference audio.")
    parser.add_argument("--caption", help="Optional caption/style text for caption-conditioned checkpoints.")
    parser.add_argument("--model-config-json", help="JSON object for ModelConfig. Defaults to base v2 config.")
    parser.add_argument("--text-tokenizer-repo", help="Override text tokenizer repo from ModelConfig.")
    parser.add_argument("--caption-tokenizer-repo", help="Override caption tokenizer repo from ModelConfig.")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--caption-max-length", type=int)
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--codec-device", default="cpu", help="PyTorch codec device: cpu, mps, or cuda.")
    parser.add_argument("--disable-codec-normalize", action="store_true")
    parser.add_argument("--enable-watermark", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--cfg-guidance-mode", default="independent", choices=("independent", "joint", "reduced"))
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-reference-seconds", type=float, default=30.0)
    parser.add_argument("--no-context-kv-cache", action="store_true")
    parser.add_argument("--print-boundaries", action="store_true", help="Print JSON boundary/config metadata before generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_reference and not args.reference_wav:
        raise SystemExit("error: specify --reference-wav or --no-reference")

    model_config = load_model_config_json(args.model_config_json)
    runtime_config = MLXRuntimeConfig(
        model_config=model_config,
        weights_path=args.weights,
        text_tokenizer_repo=args.text_tokenizer_repo,
        caption_tokenizer_repo=args.caption_tokenizer_repo,
        text_max_length=int(args.text_max_length),
        caption_max_length=args.caption_max_length,
        codec=DACVAEBridgeConfig(
            codec_repo=args.codec_repo,
            codec_device=args.codec_device,
            enable_watermark=bool(args.enable_watermark),
            normalize_db=None if args.disable_codec_normalize else -16.0,
        ),
    )
    runtime = MLXDACVAERuntime(config=runtime_config)
    if args.print_boundaries:
        print(json.dumps(runtime.describe_boundaries(), indent=2, sort_keys=True, default=str), flush=True)

    result = runtime.generate(
        GenerationRequest(
            text=args.text,
            output_wav=args.output,
            reference_wav=args.reference_wav,
            no_reference=bool(args.no_reference),
            caption=args.caption,
            seconds=float(args.seconds),
            num_steps=int(args.num_steps),
            cfg_scale_text=float(args.cfg_scale_text),
            cfg_scale_caption=float(args.cfg_scale_caption),
            cfg_scale_speaker=float(args.cfg_scale_speaker),
            cfg_guidance_mode=args.cfg_guidance_mode,
            cfg_min_t=float(args.cfg_min_t),
            cfg_max_t=float(args.cfg_max_t),
            seed=int(args.seed),
            max_reference_seconds=args.max_reference_seconds,
            use_context_kv_cache=not bool(args.no_context_kv_cache),
        )
    )
    for line in iter_messages(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
