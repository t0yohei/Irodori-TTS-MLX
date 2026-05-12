from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .runtime import DACVAEBridgeConfig, PyTorchDACVAEBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short-lived PyTorch DACVAE helper process.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    describe = subparsers.add_parser("describe")
    describe.add_argument("--config-json", required=True)

    encode = subparsers.add_parser("encode")
    encode.add_argument("--config-json", required=True)
    encode.add_argument("--reference-wav", required=True)
    encode.add_argument("--output-latents", required=True)
    encode.add_argument("--max-seconds", type=float)
    encode.add_argument("--normalize-db", type=float)
    encode.add_argument("--ensure-max", action="store_true")

    decode = subparsers.add_parser("decode")
    decode.add_argument("--config-json", required=True)
    decode.add_argument("--input-latents", required=True)
    decode.add_argument("--output-wav", required=True)
    decode.add_argument("--max-samples", type=int)
    return parser.parse_args()


def load_config(raw: str) -> DACVAEBridgeConfig:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("codec worker config must be a JSON object")
    return DACVAEBridgeConfig(**payload)


def main() -> int:
    args = parse_args()
    bridge = PyTorchDACVAEBridge(config=load_config(args.config_json))
    if args.action == "describe":
        print(
            json.dumps(
                {
                    "sample_rate": int(bridge.sample_rate),
                    "latent_dim": int(bridge.latent_dim),
                    "hop_length": int(bridge.hop_length),
                }
            )
        )
        return 0
    if args.action == "encode":
        latents = bridge.encode_reference(
            args.reference_wav,
            max_seconds=args.max_seconds,
            normalize_db=args.normalize_db,
            ensure_max=bool(args.ensure_max),
        )
        out = Path(args.output_latents)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.array(latents, dtype=np.float32))
        return 0
    if args.action == "decode":
        latents = np.load(args.input_latents)
        bridge.decode_to_wav(mx.array(latents, dtype=mx.float32), args.output_wav, max_samples=args.max_samples)
        return 0
    raise ValueError(f"unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
