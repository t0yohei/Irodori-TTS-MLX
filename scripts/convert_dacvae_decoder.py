#!/usr/bin/env python3
"""Convert a real Semantic-DACVAE decoder checkpoint into a local MLX artifact.

This converter intentionally keeps the heavyweight upstream weights outside the
repository. It loads the public PyTorch `weights.pth`, extracts the logical
decoder-side tensors required for the v0.2 MLX port, and writes a deterministic
`.npz` artifact with provenance metadata. The current runtime can inspect this
real artifact contract without importing PyTorch; full MLX execution of the
DACVAE convolution stack is tracked separately from this manifest conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_SOURCE_FILE = "weights.pth"
ARTIFACT_FORMAT = "irodori-tts-mlx-dacvae-codec"
ARTIFACT_FORMAT_VERSION = "0.2"
ARTIFACT_KIND = "real_semantic_dacvae_decoder"
SAMPLE_RATE = 48000
HOP_LENGTH = 512
LATENT_DIM = 32
TENSOR_PREFIX = "dacvae_decoder/"

DECODE_REQUIRED_PREFIXES = (
    "quantizer.out_proj.",
    "decoder.",
)


class DACVAEDecoderConversionError(RuntimeError):
    """Raised when the Semantic-DACVAE decoder artifact cannot be converted."""


@dataclass(frozen=True)
class DecoderTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    array: Any

    @property
    def parameter_count(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Aratako/Semantic-DACVAE-Japanese-32dim weights.pth into a "
            "local MLX DACVAE decoder artifact contract."
        )
    )
    parser.add_argument("source", help="Local upstream weights.pth path.")
    parser.add_argument("output", help="Output dacvae-codec .npz path.")
    parser.add_argument("--source-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--source-revision", required=True, help="Exact upstream Hugging Face revision or commit.")
    parser.add_argument("--source-file", default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--dacvae-revision", required=True, help="Exact dacvae package/repository revision used.")
    parser.add_argument("--converter-commit", help="Converter git commit. Defaults to current repository HEAD.")
    parser.add_argument(
        "--license-review-status",
        choices=("pending", "approved", "rejected"),
        default="pending",
    )
    parser.add_argument("--license-review-ref", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing the artifact.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit a machine-readable report.")
    return parser.parse_args(argv)


def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency.
        raise DACVAEDecoderConversionError("numpy is required to write the DACVAE decoder artifact.") from exc
    return np


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local real-weight validation env.
        raise DACVAEDecoderConversionError(
            "torch is required to read upstream Semantic-DACVAE weights.pth. "
            "Install the runtime extras, then rerun this converter."
        ) from exc
    return torch


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "detach") or hasattr(value, "cpu") or hasattr(value, "numpy") or hasattr(value, "shape")


def unwrap_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "codec", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping) and any(_looks_like_tensor(item) for item in value.values()):
                return value
        if any(_looks_like_tensor(item) for item in checkpoint.values()):
            return checkpoint
    raise DACVAEDecoderConversionError(
        "Could not find a tensor state_dict in the checkpoint. Expected a mapping or a mapping under "
        "state_dict/model/model_state_dict/codec/net."
    )


def _strip_known_prefix(name: str) -> str:
    for prefix in ("module.", "model.", "codec."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _to_numpy_array(value: Any) -> Any:
    np = import_numpy()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not array.dtype.kind in {"f", "i", "u", "b"}:
        raise DACVAEDecoderConversionError(f"Unsupported tensor dtype for DACVAE decoder conversion: {array.dtype}")
    if array.dtype.kind == "f" and array.dtype.itemsize > 4:
        array = array.astype("float32")
    return array


def extract_decoder_tensors(state_dict: Mapping[str, Any]) -> dict[str, DecoderTensor]:
    tensors: dict[str, DecoderTensor] = {}
    for raw_name, raw_value in state_dict.items():
        name = _strip_known_prefix(str(raw_name))
        if not name.startswith(DECODE_REQUIRED_PREFIXES):
            continue
        array = _to_numpy_array(raw_value)
        tensors[name] = DecoderTensor(
            name=name,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            array=array,
        )

    missing_groups = [
        prefix.rstrip(".")
        for prefix in DECODE_REQUIRED_PREFIXES
        if not any(name.startswith(prefix) for name in tensors)
    ]
    if missing_groups:
        raise DACVAEDecoderConversionError(
            "Checkpoint does not contain required Semantic-DACVAE decoder tensor groups: "
            + ", ".join(missing_groups)
        )

    quantizer_candidates = [
        tensor
        for name, tensor in tensors.items()
        if name.startswith("quantizer.out_proj.") and len(tensor.shape) >= 2 and LATENT_DIM in tensor.shape
    ]
    if not quantizer_candidates:
        raise DACVAEDecoderConversionError(
            "quantizer.out_proj tensors must include the 32-channel latent decode projection."
        )

    decoder_output_candidates = [
        tensor
        for name, tensor in tensors.items()
        if name.startswith("decoder.") and len(tensor.shape) >= 2 and 1 in tensor.shape
    ]
    if not decoder_output_candidates:
        raise DACVAEDecoderConversionError(
            "decoder tensors must include a mono waveform projection with one input or output channel."
        )

    return dict(sorted(tensors.items()))


def tensor_manifest(tensors: Mapping[str, DecoderTensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": tensor.name,
            "artifact_key": TENSOR_PREFIX + tensor.name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "parameter_count": tensor.parameter_count,
        }
        for tensor in tensors.values()
    ]


def manifest_digest(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_metadata(args: argparse.Namespace, tensors: Mapping[str, DecoderTensor]) -> dict[str, Any]:
    manifest = tensor_manifest(tensors)
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "source_file": args.source_file,
        "converter_commit": args.converter_commit or current_git_commit(),
        "dacvae_revision": args.dacvae_revision,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "latent_dim": LATENT_DIM,
        "decode_present": True,
        "encode_present": False,
        "watermark_bypass": {
            "decoder_alpha": 0.0,
            "watermark_replacement": "decoder.wm_model.encoder_block.forward_no_conv when present",
        },
        "license_review_status": args.license_review_status,
        "license_review_ref": args.license_review_ref,
        "tensor_count": len(tensors),
        "total_parameters": sum(tensor.parameter_count for tensor in tensors.values()),
        "tensor_manifest_sha256": manifest_digest(manifest),
        "tensors": manifest,
        "runtime_status": {
            "mlx_decoder_execution": "blocked",
            "blocker": (
                "The artifact contains the real Semantic-DACVAE decoder tensors and provenance, "
                "but the MLX convolutional DACVAE executor is not implemented in this repository yet."
            ),
        },
    }


def build_report(source: Path, output: Path, metadata: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    return {
        "source": str(source),
        "output": str(output),
        "dry_run": dry_run,
        "artifact_format": metadata["artifact_format"],
        "artifact_kind": metadata["artifact_kind"],
        "source_repo": metadata["source_repo"],
        "source_revision": metadata["source_revision"],
        "sample_rate": metadata["sample_rate"],
        "hop_length": metadata["hop_length"],
        "latent_dim": metadata["latent_dim"],
        "tensor_count": metadata["tensor_count"],
        "total_parameters": metadata["total_parameters"],
        "tensor_manifest_sha256": metadata["tensor_manifest_sha256"],
        "license_review_status": metadata["license_review_status"],
        "runtime_status": metadata["runtime_status"],
    }


def validate_source_path(source: Path) -> None:
    if not source.exists():
        raise DACVAEDecoderConversionError(f"Source checkpoint does not exist: {source}")
    if not source.is_file():
        raise DACVAEDecoderConversionError(f"Source checkpoint is not a file: {source}")
    if source.name != DEFAULT_SOURCE_FILE and source.suffix != ".pth":
        raise DACVAEDecoderConversionError("Expected the upstream Semantic-DACVAE weights.pth file.")


def write_npz_atomic(output: Path, tensors: Mapping[str, DecoderTensor], metadata: Mapping[str, Any]) -> None:
    np = import_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "sample_rate": np.array(SAMPLE_RATE, dtype=np.int64),
        "hop_length": np.array(HOP_LENGTH, dtype=np.int64),
        "latent_dim": np.array(LATENT_DIM, dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    arrays.update({TENSOR_PREFIX + name: tensor.array for name, tensor in tensors.items()})

    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "wb") as file:
            np.savez(file, **arrays)
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def convert(source: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    validate_source_path(source)
    torch = import_torch()
    checkpoint = torch.load(str(source), map_location="cpu")
    tensors = extract_decoder_tensors(unwrap_state_dict(checkpoint))
    metadata = build_metadata(args, tensors)
    if not args.dry_run:
        write_npz_atomic(output, tensors, metadata)
    return build_report(source, output, metadata, dry_run=bool(args.dry_run))


def print_text_report(report: Mapping[str, Any]) -> None:
    print(f"source: {report['source']}")
    print(f"output: {report['output']}")
    print(f"dry_run: {report['dry_run']}")
    print(f"artifact_format: {report['artifact_format']}")
    print(f"artifact_kind: {report['artifact_kind']}")
    print(f"source_repo: {report['source_repo']}")
    print(f"source_revision: {report['source_revision']}")
    print(f"sample_rate: {report['sample_rate']}")
    print(f"hop_length: {report['hop_length']}")
    print(f"latent_dim: {report['latent_dim']}")
    print(f"tensor_count: {report['tensor_count']}")
    print(f"total_parameters: {report['total_parameters']:,}")
    print(f"tensor_manifest_sha256: {report['tensor_manifest_sha256']}")
    print(f"license_review_status: {report['license_review_status']}")
    print(f"runtime_status: {report['runtime_status']['mlx_decoder_execution']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser()
    try:
        report = convert(source, output, args)
    except DACVAEDecoderConversionError as exc:
        print(f"DACVAE decoder conversion failed: {exc}", file=sys.stderr)
        return 2
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
