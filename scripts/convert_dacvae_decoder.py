#!/usr/bin/env python3
"""Convert a real Semantic-DACVAE checkpoint into a local MLX artifact.

This converter intentionally keeps the heavyweight upstream weights outside the
repository. It loads the public PyTorch `weights.pth`, extracts the logical
encoder/decoder tensors required for the v0.2 MLX port, and writes a
deterministic `.npz` artifact with provenance metadata and runtime-ready MLX
tensor layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
HOP_LENGTH = 1920
LATENT_DIM = 32
DECODER_LATENT_DIM = 1024
DECODER_DIM = 1536
DECODER_RATES = (12, 10, 8, 2)
WM_RATES = (8, 5, 4, 2)
ENCODER_DIM = 64
ENCODER_RATES = (2, 8, 10, 12)
TENSOR_PREFIX = "dacvae_decoder/"
ENCODER_TENSOR_PREFIX = "dacvae_encoder/"
EXECUTABLE_TENSOR_PREFIX = "dacvae_decoder_exec/"
EXECUTABLE_ENCODER_TENSOR_PREFIX = "dacvae_encoder_exec/"

DECODE_REQUIRED_PREFIXES = (
    "quantizer.out_proj.",
    "decoder.",
)
ENCODE_REQUIRED_PREFIXES = (
    "encoder.",
    "quantizer.in_proj.",
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


@dataclass(frozen=True)
class ExecutableDecoderTensor:
    source_name: str
    target_name: str
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
            "local MLX DACVAE codec artifact contract."
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
        if not name.startswith(DECODE_REQUIRED_PREFIXES + ENCODE_REQUIRED_PREFIXES):
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


def semantic_dacvae_decoder_config_metadata() -> dict[str, object]:
    return {
        "latent_dim": DECODER_LATENT_DIM,
        "decoder_dim": DECODER_DIM,
        "decoder_rates": list(DECODER_RATES),
        "wm_rates": list(WM_RATES),
        "codebook_dim": LATENT_DIM,
        "output_channels": 1,
    }


def semantic_dacvae_encoder_config_metadata() -> dict[str, object]:
    return {
        "input_channels": 1,
        "encoder_dim": ENCODER_DIM,
        "encoder_rates": list(ENCODER_RATES),
        "latent_dim": DECODER_LATENT_DIM,
        "codebook_dim": LATENT_DIM,
    }


def _map_weight_norm_suffix(suffix: str) -> str | None:
    aliases = {
        "bias": "bias",
        "alpha": "alpha",
        "weight": "weight",
        "weight_g": "weight_g",
        "weight_v": "weight_v",
        "parametrizations.weight.original0": "weight_g",
        "parametrizations.weight.original1": "weight_v",
    }
    return aliases.get(suffix)


def _decoder_block_target(index: int, suffix: str) -> tuple[str, bool] | None:
    block_prefix = f"blocks.{index}"
    simple_map = {
        "0": (f"{block_prefix}.main_upsample.0", False),
        "1": (f"{block_prefix}.main_upsample.1", True),
    }
    for source_block, target in simple_map.items():
        prefix = f"{source_block}."
        if suffix.startswith(prefix):
            param = _map_weight_norm_suffix(suffix[len(prefix) :])
            return (f"{target[0]}.{param}", target[1]) if param else None

    residual_map = {
        "4": "0",
        "5": "1",
        "8": "2",
    }
    for source_block, residual_index in residual_map.items():
        prefix = f"{source_block}.block."
        if not suffix.startswith(prefix):
            continue
        inner = suffix[len(prefix) :]
        if inner.startswith("0."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.act1.{param}", False) if param else None
        if inner.startswith("1."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.conv1.{param}", False) if param else None
        if inner.startswith("2."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.act2.{param}", False) if param else None
        if inner.startswith("3."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.conv2.{param}", False) if param else None
    return None


def _encoder_block_target(index: int, suffix: str) -> tuple[str, bool] | None:
    block_prefix = f"blocks.{index}"
    residual_map = {
        "0": "0",
        "1": "1",
        "2": "2",
    }
    for source_block, residual_index in residual_map.items():
        prefix = f"{source_block}.block."
        if not suffix.startswith(prefix):
            continue
        inner = suffix[len(prefix) :]
        if inner.startswith("0."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.act1.{param}", False) if param else None
        if inner.startswith("1."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.conv1.{param}", False) if param else None
        if inner.startswith("2."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.act2.{param}", False) if param else None
        if inner.startswith("3."):
            param = _map_weight_norm_suffix(inner[2:])
            return (f"{block_prefix}.residuals.{residual_index}.conv2.{param}", False) if param else None

    for source_block, target in (("3", f"{block_prefix}.downsample_act"), ("4", f"{block_prefix}.downsample")):
        prefix = f"{source_block}."
        if suffix.startswith(prefix):
            param = _map_weight_norm_suffix(suffix[len(prefix) :])
            return (f"{target}.{param}", False) if param else None
    return None


def _executable_target_for_name(name: str) -> tuple[str, bool] | None:
    # Irodori's wrapper sets decoder.alpha=0 and replaces decoder.watermark with
    # wm_model.encoder_block.forward_no_conv, so this pre block is the final
    # non-watermarked output path for converted public checkpoints.
    prefixes: tuple[tuple[str, str, bool], ...] = (
        ("quantizer.out_proj.", "quantizer_out_proj.", False),
        ("decoder.model.0.", "conv_in.", False),
        ("decoder.wm_model.encoder_block.pre.0.", "snake_out.", False),
        ("decoder.wm_model.encoder_block.pre.1.", "conv_out.", False),
    )
    for source_prefix, target_prefix, is_transposed_conv in prefixes:
        if name.startswith(source_prefix):
            param = _map_weight_norm_suffix(name[len(source_prefix) :])
            return (target_prefix + param, is_transposed_conv) if param else None

    match = re.match(r"^decoder\.model\.(\d+)\.block\.(.+)$", name)
    if match:
        model_index = int(match.group(1))
        if model_index <= 0:
            return None
        block_index = model_index - 1
        if block_index >= len(DECODER_RATES):
            return None
        return _decoder_block_target(block_index, match.group(2))
    return None


def _executable_encoder_target_for_name(name: str) -> tuple[str, bool] | None:
    prefixes: tuple[tuple[str, str, bool], ...] = (
        ("encoder.block.0.", "conv_in.", False),
        ("encoder.block.5.", "snake_out.", False),
        ("encoder.block.6.", "conv_out.", False),
        ("quantizer.in_proj.", "quantizer_in_proj.", False),
    )
    for source_prefix, target_prefix, is_transposed_conv in prefixes:
        if name.startswith(source_prefix):
            param = _map_weight_norm_suffix(name[len(source_prefix) :])
            return (target_prefix + param, is_transposed_conv) if param else None

    match = re.match(r"^encoder\.block\.(\d+)\.block\.(.+)$", name)
    if match:
        model_index = int(match.group(1))
        block_index = model_index - 1
        if block_index < 0 or block_index >= len(ENCODER_RATES):
            return None
        return _encoder_block_target(block_index, match.group(2))
    return None


def _to_mlx_conv_weight(array: Any, *, is_transposed_conv: bool) -> Any:
    np = import_numpy()
    value = np.asarray(array)
    if value.ndim != 3:
        return value
    if is_transposed_conv:
        return np.transpose(value, (1, 2, 0)).astype(value.dtype, copy=False)
    return np.transpose(value, (0, 2, 1)).astype(value.dtype, copy=False)


def _to_mlx_snake_alpha(array: Any) -> Any:
    np = import_numpy()
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 1:
        return np.transpose(value, (0, 2, 1)).astype(value.dtype, copy=False)
    return value


def build_executable_decoder_tensors(
    tensors: Mapping[str, DecoderTensor],
) -> dict[str, ExecutableDecoderTensor]:
    executable: dict[str, ExecutableDecoderTensor] = {}
    for source_name, tensor in tensors.items():
        mapped = _executable_target_for_name(source_name)
        if mapped is None:
            continue
        target_name, is_transposed_conv = mapped
        if target_name.endswith(".alpha"):
            array = _to_mlx_snake_alpha(tensor.array)
        elif target_name.endswith((".weight", ".weight_g", ".weight_v")):
            array = _to_mlx_conv_weight(tensor.array, is_transposed_conv=is_transposed_conv)
        else:
            array = tensor.array
        executable[target_name] = ExecutableDecoderTensor(
            source_name=source_name,
            target_name=target_name,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            array=array,
        )

    if not executable:
        raise DACVAEDecoderConversionError(
            "No executable Semantic-DACVAE decoder tensors could be mapped from the checkpoint."
        )
    missing_pairs: list[str] = []
    for prefix in sorted({name.rsplit(".", 1)[0] for name in executable}):
        has_direct = f"{prefix}.weight" in executable
        has_weight_norm = f"{prefix}.weight_g" in executable and f"{prefix}.weight_v" in executable
        if any(name.startswith(prefix + ".weight") for name in executable) and not (has_direct or has_weight_norm):
            missing_pairs.append(prefix)
    if missing_pairs:
        raise DACVAEDecoderConversionError(
            "Incomplete weight-normalized decoder tensors for executable modules: " + ", ".join(missing_pairs[:8])
        )
    from irodori_mlx.dacvae import semantic_dacvae_decoder_expected_shapes

    expected_shapes = semantic_dacvae_decoder_expected_shapes()
    missing_required = tuple(name for name in expected_shapes if name not in executable)
    if missing_required:
        raise DACVAEDecoderConversionError(
            "Checkpoint is missing executable Semantic-DACVAE decoder tensors required by the MLX runtime: "
            + ", ".join(missing_required[:8])
        )
    shape_mismatches = [
        f"{name}: expected {expected_shapes[name]}, got {executable[name].shape}"
        for name in expected_shapes
        if tuple(executable[name].shape) != tuple(expected_shapes[name])
    ]
    if shape_mismatches:
        raise DACVAEDecoderConversionError(
            "Executable Semantic-DACVAE decoder tensor shapes do not match the MLX runtime contract: "
            + "; ".join(shape_mismatches[:8])
        )
    return dict(sorted(executable.items()))


def build_executable_encoder_tensors(
    tensors: Mapping[str, DecoderTensor],
) -> dict[str, ExecutableDecoderTensor]:
    executable: dict[str, ExecutableDecoderTensor] = {}
    for source_name, tensor in tensors.items():
        mapped = _executable_encoder_target_for_name(source_name)
        if mapped is None:
            continue
        target_name, is_transposed_conv = mapped
        if target_name.endswith(".alpha"):
            array = _to_mlx_snake_alpha(tensor.array)
        elif target_name.endswith((".weight", ".weight_g", ".weight_v")):
            array = _to_mlx_conv_weight(tensor.array, is_transposed_conv=is_transposed_conv)
        else:
            array = tensor.array
        executable[target_name] = ExecutableDecoderTensor(
            source_name=source_name,
            target_name=target_name,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            array=array,
        )

    if not executable:
        raise DACVAEDecoderConversionError(
            "No executable Semantic-DACVAE encoder tensors could be mapped from the checkpoint."
        )
    missing_pairs: list[str] = []
    for prefix in sorted({name.rsplit(".", 1)[0] for name in executable}):
        has_direct = f"{prefix}.weight" in executable
        has_weight_norm = f"{prefix}.weight_g" in executable and f"{prefix}.weight_v" in executable
        if any(name.startswith(prefix + ".weight") for name in executable) and not (has_direct or has_weight_norm):
            missing_pairs.append(prefix)
    if missing_pairs:
        raise DACVAEDecoderConversionError(
            "Incomplete weight-normalized encoder tensors for executable modules: " + ", ".join(missing_pairs[:8])
        )
    from irodori_mlx.dacvae import semantic_dacvae_encoder_expected_shapes

    expected_shapes = semantic_dacvae_encoder_expected_shapes()
    missing_required = tuple(name for name in expected_shapes if name not in executable)
    if missing_required:
        raise DACVAEDecoderConversionError(
            "Checkpoint is missing executable Semantic-DACVAE encoder tensors required by the MLX runtime: "
            + ", ".join(missing_required[:8])
        )
    shape_mismatches = [
        f"{name}: expected {expected_shapes[name]}, got {executable[name].shape}"
        for name in expected_shapes
        if tuple(executable[name].shape) != tuple(expected_shapes[name])
    ]
    if shape_mismatches:
        raise DACVAEDecoderConversionError(
            "Executable Semantic-DACVAE encoder tensor shapes do not match the MLX runtime contract: "
            + "; ".join(shape_mismatches[:8])
        )
    return dict(sorted(executable.items()))


def tensor_manifest(tensors: Mapping[str, DecoderTensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": tensor.name,
            "artifact_key": (ENCODER_TENSOR_PREFIX if tensor.name.startswith(ENCODE_REQUIRED_PREFIXES) else TENSOR_PREFIX)
            + tensor.name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "parameter_count": tensor.parameter_count,
        }
        for tensor in tensors.values()
    ]


def executable_tensor_manifest(tensors: Mapping[str, ExecutableDecoderTensor]) -> list[dict[str, Any]]:
    return [
        {
            "source_name": tensor.source_name,
            "target_name": tensor.target_name,
            "artifact_key": EXECUTABLE_TENSOR_PREFIX + tensor.target_name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "parameter_count": tensor.parameter_count,
        }
        for tensor in tensors.values()
    ]


def executable_encoder_tensor_manifest(tensors: Mapping[str, ExecutableDecoderTensor]) -> list[dict[str, Any]]:
    return [
        {
            "source_name": tensor.source_name,
            "target_name": tensor.target_name,
            "artifact_key": EXECUTABLE_ENCODER_TENSOR_PREFIX + tensor.target_name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "parameter_count": tensor.parameter_count,
        }
        for tensor in tensors.values()
    ]


def manifest_digest(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_metadata(
    args: argparse.Namespace,
    tensors: Mapping[str, DecoderTensor],
    executable_tensors: Mapping[str, ExecutableDecoderTensor],
    executable_encoder_tensors: Mapping[str, ExecutableDecoderTensor],
) -> dict[str, Any]:
    manifest = tensor_manifest(tensors)
    executable_manifest = executable_tensor_manifest(executable_tensors)
    executable_encoder_manifest = executable_encoder_tensor_manifest(executable_encoder_tensors)
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
        "semantic_dacvae_decoder_config": semantic_dacvae_decoder_config_metadata(),
        "semantic_dacvae_encoder_config": semantic_dacvae_encoder_config_metadata(),
        "decode_present": True,
        "encode_present": True,
        "watermark_bypass": {
            "decoder_alpha": 0.0,
            "watermark_replacement": "decoder.wm_model.encoder_block.forward_no_conv when present",
        },
        "license_review_status": args.license_review_status,
        "license_review_ref": args.license_review_ref,
        "tensor_count": len(tensors),
        "executable_tensor_count": len(executable_tensors),
        "executable_encoder_tensor_count": len(executable_encoder_tensors),
        "total_parameters": sum(tensor.parameter_count for tensor in tensors.values()),
        "executable_total_parameters": sum(tensor.parameter_count for tensor in executable_tensors.values()),
        "executable_encoder_total_parameters": sum(tensor.parameter_count for tensor in executable_encoder_tensors.values()),
        "tensor_manifest_sha256": manifest_digest(manifest),
        "executable_tensor_manifest_sha256": manifest_digest(executable_manifest),
        "executable_encoder_tensor_manifest_sha256": manifest_digest(executable_encoder_manifest),
        "tensors": manifest,
        "executable_tensors": executable_manifest,
        "executable_encoder_tensors": executable_encoder_manifest,
        "runtime_status": {
            "mlx_decoder_execution": "available_unvalidated",
            "mlx_encoder_execution": "available_unvalidated",
            "parity_status": "not_validated",
            "note": "Executable MLX encoder and decoder tensors are present, but acoustic parity is gated separately.",
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
        "executable_tensor_count": metadata["executable_tensor_count"],
        "executable_encoder_tensor_count": metadata["executable_encoder_tensor_count"],
        "total_parameters": metadata["total_parameters"],
        "executable_total_parameters": metadata["executable_total_parameters"],
        "executable_encoder_total_parameters": metadata["executable_encoder_total_parameters"],
        "tensor_manifest_sha256": metadata["tensor_manifest_sha256"],
        "executable_tensor_manifest_sha256": metadata["executable_tensor_manifest_sha256"],
        "executable_encoder_tensor_manifest_sha256": metadata["executable_encoder_tensor_manifest_sha256"],
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


def write_npz_atomic(
    output: Path,
    tensors: Mapping[str, DecoderTensor],
    executable_tensors: Mapping[str, ExecutableDecoderTensor],
    executable_encoder_tensors: Mapping[str, ExecutableDecoderTensor],
    metadata: Mapping[str, Any],
) -> None:
    np = import_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "sample_rate": np.array(SAMPLE_RATE, dtype=np.int64),
        "hop_length": np.array(HOP_LENGTH, dtype=np.int64),
        "latent_dim": np.array(LATENT_DIM, dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    arrays.update({TENSOR_PREFIX + name: tensor.array for name, tensor in tensors.items() if name.startswith(DECODE_REQUIRED_PREFIXES)})
    arrays.update({ENCODER_TENSOR_PREFIX + name: tensor.array for name, tensor in tensors.items() if name.startswith(ENCODE_REQUIRED_PREFIXES)})
    arrays.update({EXECUTABLE_TENSOR_PREFIX + name: tensor.array for name, tensor in executable_tensors.items()})
    arrays.update({EXECUTABLE_ENCODER_TENSOR_PREFIX + name: tensor.array for name, tensor in executable_encoder_tensors.items()})

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
    executable_tensors = build_executable_decoder_tensors(tensors)
    executable_encoder_tensors = build_executable_encoder_tensors(tensors)
    metadata = build_metadata(args, tensors, executable_tensors, executable_encoder_tensors)
    if not args.dry_run:
        write_npz_atomic(output, tensors, executable_tensors, executable_encoder_tensors, metadata)
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
    print(f"executable_tensor_count: {report['executable_tensor_count']}")
    print(f"executable_encoder_tensor_count: {report['executable_encoder_tensor_count']}")
    print(f"total_parameters: {report['total_parameters']:,}")
    print(f"executable_total_parameters: {report['executable_total_parameters']:,}")
    print(f"executable_encoder_total_parameters: {report['executable_encoder_total_parameters']:,}")
    print(f"tensor_manifest_sha256: {report['tensor_manifest_sha256']}")
    print(f"executable_tensor_manifest_sha256: {report['executable_tensor_manifest_sha256']}")
    print(f"executable_encoder_tensor_manifest_sha256: {report['executable_encoder_tensor_manifest_sha256']}")
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
