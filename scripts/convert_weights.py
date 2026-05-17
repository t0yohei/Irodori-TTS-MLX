#!/usr/bin/env python3
"""Convert upstream Irodori-TTS weights into an MLX-friendly archive.

The converter supports the base v2 checkpoint layout from
``Aratako/Irodori-TTS-500M-v2``, the VoiceDesign / caption-conditioned
layout from ``Aratako/Irodori-TTS-500M-v2-VoiceDesign``, and the v3
speaker-conditioned layout from ``Aratako/Irodori-TTS-500M-v3``. It validates
the expected key mapping before writing anything and copies tensors as-is into
a NumPy ``.npz`` archive, which can be loaded by MLX via ``mx.load``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from inspect_checkpoint import InspectionError, inspect_local_safetensors
except ImportError:  # pragma: no cover - fallback for unusual invocation paths
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inspect_checkpoint import InspectionError, inspect_local_safetensors

SUPPORTED_SOURCE_SUFFIX = ".safetensors"
SUPPORTED_FLOAT_NAMES = {
    "F16",
    "F32",
    "float16",
    "float32",
    "torch.float16",
    "torch.float32",
    "dtype('float16')",
    "dtype('float32')",
}
CHECKPOINT_FAMILY_BASE = "base_v2"
CHECKPOINT_FAMILY_VOICEDESIGN = "voicedesign"
CHECKPOINT_FAMILY_V3 = "v3"
SUPPORTED_CHECKPOINTS = {
    CHECKPOINT_FAMILY_BASE: "Aratako/Irodori-TTS-500M-v2",
    CHECKPOINT_FAMILY_VOICEDESIGN: "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
    CHECKPOINT_FAMILY_V3: "Aratako/Irodori-TTS-500M-v3",
}
EXPECTED_TENSOR_COUNTS = {
    CHECKPOINT_FAMILY_BASE: 613,
    CHECKPOINT_FAMILY_VOICEDESIGN: 636,
    CHECKPOINT_FAMILY_V3: 637,
}
CAPTION_PREFIXES = (
    "caption_encoder.",
    "caption_norm.",
)
CAPTION_FRAGMENTS = (
    ".attention.wk_caption.weight",
    ".attention.wv_caption.weight",
)
SPEAKER_PREFIXES = (
    "speaker_encoder.",
    "speaker_norm.",
)
SPEAKER_FRAGMENTS = (
    ".attention.wk_speaker.weight",
    ".attention.wv_speaker.weight",
)
DURATION_PREFIXES = (
    "duration_predictor.",
)


class ConversionError(RuntimeError):
    """Raised when conversion cannot continue safely."""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    array: Any | None = None

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape) if self.shape else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local Irodori-TTS safetensors checkpoint into an MLX-friendly .npz archive."
    )
    parser.add_argument("source", nargs="?", help="Local .safetensors checkpoint path.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .npz path. Required unless --dry-run or --self-test is used.",
    )
    parser.add_argument(
        "--format",
        choices=("npz",),
        default="npz",
        help="Output format. Only npz is supported.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing output.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON report.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-light converter self-tests and exit. The source argument is ignored.",
    )
    return parser.parse_args()


def build_expected_shapes(*, family: str) -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}

    def add(name: str, shape: tuple[int, ...]) -> None:
        if name in expected:
            raise AssertionError(f"duplicate expected tensor: {name}")
        expected[name] = shape

    for i in range(12):
        block = f"blocks.{i}"
        for proj in ("gate", "wk", "wo", "wq", "wv"):
            add(f"{block}.attention.{proj}.weight", (1280, 1280))
        for proj in ("wk_text", "wv_text"):
            add(f"{block}.attention.{proj}.weight", (1280, 512))
        if family in (CHECKPOINT_FAMILY_BASE, CHECKPOINT_FAMILY_V3):
            for proj in ("wk_speaker", "wv_speaker"):
                add(f"{block}.attention.{proj}.weight", (1280, 768))
        elif family == CHECKPOINT_FAMILY_VOICEDESIGN:
            for proj in ("wk_caption", "wv_caption"):
                add(f"{block}.attention.{proj}.weight", (1280, 512))
        else:
            raise AssertionError(f"unknown checkpoint family: {family}")
        for norm in ("k_norm", "q_norm"):
            add(f"{block}.attention.{norm}.weight", (20, 64))
        add(f"{block}.mlp.w1.weight", (3680, 1280))
        add(f"{block}.mlp.w2.weight", (1280, 3680))
        add(f"{block}.mlp.w3.weight", (3680, 1280))
        for adaln in ("attention_adaln", "mlp_adaln"):
            for branch in ("gate", "scale", "shift"):
                add(f"{block}.{adaln}.{branch}_down.weight", (192, 1280))
                add(f"{block}.{adaln}.{branch}_up.weight", (1280, 192))
                add(f"{block}.{adaln}.{branch}_up.bias", (1280,))

    add("text_encoder.text_embedding.weight", (99574, 512))
    for i in range(10):
        block = f"text_encoder.blocks.{i}"
        for proj in ("gate", "wk", "wo", "wq", "wv"):
            add(f"{block}.attention.{proj}.weight", (512, 512))
        for norm in ("k_norm", "q_norm"):
            add(f"{block}.attention.{norm}.weight", (8, 64))
        for norm in ("attention_norm", "mlp_norm"):
            add(f"{block}.{norm}.weight", (512,))
        add(f"{block}.mlp.w1.weight", (1331, 512))
        add(f"{block}.mlp.w2.weight", (512, 1331))
        add(f"{block}.mlp.w3.weight", (1331, 512))
    add("text_norm.weight", (512,))

    if family in (CHECKPOINT_FAMILY_BASE, CHECKPOINT_FAMILY_V3):
        add("speaker_encoder.in_proj.weight", (768, 32))
        add("speaker_encoder.in_proj.bias", (768,))
        for i in range(8):
            block = f"speaker_encoder.blocks.{i}"
            for proj in ("gate", "wk", "wo", "wq", "wv"):
                add(f"{block}.attention.{proj}.weight", (768, 768))
            for norm in ("k_norm", "q_norm"):
                add(f"{block}.attention.{norm}.weight", (12, 64))
            for norm in ("attention_norm", "mlp_norm"):
                add(f"{block}.{norm}.weight", (768,))
            add(f"{block}.mlp.w1.weight", (1996, 768))
            add(f"{block}.mlp.w2.weight", (768, 1996))
            add(f"{block}.mlp.w3.weight", (1996, 768))
        add("speaker_norm.weight", (768,))
    else:
        add("caption_encoder.text_embedding.weight", (99574, 512))
        for i in range(10):
            block = f"caption_encoder.blocks.{i}"
            for proj in ("gate", "wk", "wo", "wq", "wv"):
                add(f"{block}.attention.{proj}.weight", (512, 512))
            for norm in ("k_norm", "q_norm"):
                add(f"{block}.attention.{norm}.weight", (8, 64))
            for norm in ("attention_norm", "mlp_norm"):
                add(f"{block}.{norm}.weight", (512,))
            add(f"{block}.mlp.w1.weight", (1331, 512))
            add(f"{block}.mlp.w2.weight", (512, 1331))
            add(f"{block}.mlp.w3.weight", (1331, 512))
        add("caption_norm.weight", (512,))

    add("cond_module.0.weight", (1280, 512))
    add("cond_module.2.weight", (1280, 1280))
    add("cond_module.4.weight", (3840, 1280))
    add("in_proj.weight", (1280, 32))
    add("in_proj.bias", (1280,))
    add("out_norm.weight", (1280,))
    add("out_proj.weight", (32, 1280))
    add("out_proj.bias", (32,))

    if family == CHECKPOINT_FAMILY_V3:
        add("duration_predictor.null_speaker", (768,))
        add("duration_predictor.token_input_proj.weight", (1024, 512))
        add("duration_predictor.token_input_proj.bias", (1024,))
        for i in range(3):
            block = f"duration_predictor.token_blocks.{i}"
            add(f"{block}.modulation.weight", (3072, 768))
            add(f"{block}.modulation.bias", (3072,))
            add(f"{block}.norm.weight", (1024,))
            add(f"{block}.mlp.w1.weight", (1024, 1024))
            add(f"{block}.mlp.w2.weight", (1024, 1024))
            add(f"{block}.mlp.w3.weight", (1024, 1024))
        add("duration_predictor.token_out_norm.weight", (1024,))
        add("duration_predictor.token_out_proj.weight", (1, 1024))
        add("duration_predictor.token_out_proj.bias", (1,))

    expected_count = EXPECTED_TENSOR_COUNTS[family]
    if len(expected) != expected_count:
        raise AssertionError(f"expected {expected_count} tensors for {family}, built {len(expected)}")
    return dict(sorted(expected.items()))


EXPECTED_SHAPES_BY_FAMILY = {
    family: build_expected_shapes(family=family)
    for family in (CHECKPOINT_FAMILY_BASE, CHECKPOINT_FAMILY_VOICEDESIGN, CHECKPOINT_FAMILY_V3)
}


def is_safetensors_path(path: Path) -> bool:
    return path.suffix == SUPPORTED_SOURCE_SUFFIX


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConversionError(
            "Writing converted weights requires the optional 'numpy' package. "
            "Install it or run --dry-run for header-only validation."
        ) from exc
    return np


def dtype_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).replace("numpy.", "")


def load_safetensors_header(path: Path) -> tuple[dict[str, Any] | None, dict[str, TensorRecord]]:
    inspection = inspect_local_safetensors(path)
    records = {
        tensor.name: TensorRecord(
            name=tensor.name,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        for tensor in inspection.tensors
    }
    return inspection.config, records


def load_safetensors_arrays(path: Path) -> dict[str, TensorRecord]:
    import_numpy()
    if not has_module("safetensors"):
        raise ConversionError(
            "Converting .safetensors requires the optional 'safetensors' package. "
            "Install it or run --dry-run for header-only validation."
        )
    from safetensors import safe_open  # type: ignore[import-not-found]

    records: dict[str, TensorRecord] = {}
    with safe_open(str(path), framework="np") as handle:
        for key in handle.keys():
            array = handle.get_tensor(key)
            records[key] = TensorRecord(
                name=key,
                shape=tuple(int(dim) for dim in array.shape),
                dtype=dtype_name(array.dtype),
                array=array,
            )
    return records


def validate_source_path(path: Path) -> None:
    if not path.exists():
        raise ConversionError(f"Source checkpoint does not exist: {path}")
    if not path.is_file():
        raise ConversionError(f"Source checkpoint is not a file: {path}")
    if not is_safetensors_path(path):
        raise ConversionError(
            f"Only local {SUPPORTED_SOURCE_SUFFIX} checkpoints are supported in the converter: {path}"
        )


def load_checkpoint(path: Path, *, load_arrays: bool) -> tuple[dict[str, Any] | None, dict[str, TensorRecord]]:
    validate_source_path(path)
    config, header_records = load_safetensors_header(path)
    if not load_arrays:
        return config, header_records
    return config, load_safetensors_arrays(path)


def _has_caption_tensors(records: Mapping[str, TensorRecord]) -> bool:
    return any(name.startswith(CAPTION_PREFIXES) or any(fragment in name for fragment in CAPTION_FRAGMENTS) for name in records)


def _has_speaker_tensors(records: Mapping[str, TensorRecord]) -> bool:
    return any(name.startswith(SPEAKER_PREFIXES) or any(fragment in name for fragment in SPEAKER_FRAGMENTS) for name in records)


def _has_duration_tensors(records: Mapping[str, TensorRecord]) -> bool:
    return any(name.startswith(DURATION_PREFIXES) for name in records)


def detect_checkpoint_family(
    config: dict[str, Any] | None,
    records: Mapping[str, TensorRecord],
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if config is None:
        errors.append("metadata.config_json is required to identify the checkpoint family")
        return None, errors

    caption_config = bool(config.get("use_caption_condition") is True)
    speaker_config = bool(config.get("use_speaker_condition") is True)
    duration_config = bool(config.get("use_duration_predictor") is True)
    caption_tensors = _has_caption_tensors(records)
    speaker_tensors = _has_speaker_tensors(records)
    duration_tensors = _has_duration_tensors(records)

    if caption_config and speaker_config and speaker_tensors:
        errors.append("config is ambiguous: caption conditioning is enabled while speaker-conditioned tensors are also present")
    if caption_tensors and speaker_tensors:
        errors.append("tensor layout is ambiguous: found both caption-conditioned and speaker-conditioned tensors")
    if caption_config and duration_config:
        errors.append("config is ambiguous: caption conditioning is enabled while duration predictor fields are also present")
    if caption_tensors and duration_tensors:
        errors.append("tensor layout is ambiguous: found both caption-conditioned and duration-predictor tensors")
    if errors:
        return None, errors

    if caption_config or caption_tensors:
        return CHECKPOINT_FAMILY_VOICEDESIGN, errors

    if duration_config or duration_tensors:
        if config.get("use_caption_condition") is True:
            errors.append("v3 checkpoints must not enable caption conditioning")
        if not (speaker_config or speaker_tensors):
            errors.append("v3 checkpoints must retain the base speaker-conditioned layout")
        return CHECKPOINT_FAMILY_V3, errors

    if speaker_config or speaker_tensors:
        if config.get("use_caption_condition") is True:
            errors.append("base checkpoints must not enable caption conditioning")
        if config.get("use_duration_predictor") is True:
            errors.append("base v2 checkpoints must not enable the duration predictor")
        return CHECKPOINT_FAMILY_BASE, errors

    errors.append("could not determine checkpoint family from metadata or tensor names")
    return None, errors


def validate_base_config(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return ["metadata.config_json is required to confirm the base v2 checkpoint identity"]
    errors: list[str] = []
    expected_values = {
        "latent_dim": 32,
        "model_dim": 1280,
        "num_layers": 12,
        "text_layers": 10,
        "speaker_layers": 8,
        "speaker_dim": 768,
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            errors.append(f"config {key}: expected {expected!r}, got {config.get(key)!r}")
    if config.get("use_caption_condition") is True:
        errors.append("base checkpoints must not enable caption conditioning")
    if config.get("use_duration_predictor") is True:
        errors.append("base checkpoints must not enable the duration predictor")
    has_speaker_fields = any(key.startswith("speaker_") for key in config)
    if config.get("use_speaker_condition") is False or not has_speaker_fields:
        errors.append("base speaker conditioning fields are missing or disabled")
    return errors


def validate_voicedesign_config(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return ["metadata.config_json is required to confirm the VoiceDesign checkpoint identity"]
    errors: list[str] = []
    expected_values = {
        "latent_dim": 32,
        "model_dim": 1280,
        "num_layers": 12,
        "text_layers": 10,
        "caption_layers": 10,
        "caption_dim": 512,
        "caption_heads": 8,
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            errors.append(f"config {key}: expected {expected!r}, got {config.get(key)!r}")
    if config.get("use_caption_condition") is not True:
        errors.append("VoiceDesign checkpoints must set use_caption_condition=true")
    if config.get("use_speaker_condition") is True:
        errors.append("VoiceDesign checkpoints must not enable speaker conditioning")
    return errors


def validate_v3_config(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return ["metadata.config_json is required to confirm the v3 checkpoint identity"]
    errors: list[str] = []
    expected_values = {
        "latent_dim": 32,
        "model_dim": 1280,
        "num_layers": 12,
        "text_layers": 10,
        "speaker_layers": 8,
        "speaker_dim": 768,
        "duration_aux_dim": 14,
        "duration_hidden_dim": 1024,
        "duration_layers": 3,
        "duration_dropout": 0.1,
        "duration_attention_heads": 8,
        "duration_architecture": "token_sum_adarn_zero_no_aux",
        "duration_token_init_frames": 9.0,
        "duration_speaker_fusion": "adarn_zero",
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            errors.append(f"config {key}: expected {expected!r}, got {config.get(key)!r}")
    if config.get("use_duration_predictor") is not True:
        errors.append("v3 checkpoints must set use_duration_predictor=true")
    if config.get("use_caption_condition") is True:
        errors.append("v3 checkpoints must not enable caption conditioning")
    has_speaker_fields = any(key.startswith("speaker_") for key in config)
    if config.get("use_speaker_condition") is False or not has_speaker_fields:
        errors.append("v3 speaker conditioning fields are missing or disabled")
    return errors


def validate_records(
    records: Mapping[str, TensorRecord], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    family, family_errors = detect_checkpoint_family(config, records)
    expected_shapes = EXPECTED_SHAPES_BY_FAMILY.get(family or CHECKPOINT_FAMILY_BASE, {})
    source_keys = set(records)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - source_keys)
    unexpected = sorted(source_keys - expected_keys)
    shape_mismatches = []
    dtype_mismatches = []

    for key in sorted(source_keys & expected_keys):
        record = records[key]
        expected_shape = expected_shapes[key]
        if record.shape != expected_shape:
            shape_mismatches.append(
                {"key": key, "expected": list(expected_shape), "actual": list(record.shape)}
            )
        if dtype_name(record.dtype) not in SUPPORTED_FLOAT_NAMES:
            dtype_mismatches.append({"key": key, "expected": "float16/F16 or float32/F32", "actual": record.dtype})

    config_errors = list(family_errors)
    if family == CHECKPOINT_FAMILY_BASE:
        config_errors.extend(validate_base_config(config))
    elif family == CHECKPOINT_FAMILY_VOICEDESIGN:
        config_errors.extend(validate_voicedesign_config(config))
    elif family == CHECKPOINT_FAMILY_V3:
        config_errors.extend(validate_v3_config(config))

    ok = bool(family) and not (missing or unexpected or shape_mismatches or dtype_mismatches or config_errors)
    return {
        "ok": ok,
        "checkpoint_family": family,
        "supported_checkpoint": SUPPORTED_CHECKPOINTS.get(family) if family else None,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "unsupported_keys": [],
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "config_errors": sorted(set(config_errors)),
    }


def validation_error_message(validation: Mapping[str, Any]) -> str:
    parts: list[str] = ["Checkpoint validation failed"]
    family = validation.get("checkpoint_family")
    if family:
        parts.append(f"checkpoint_family: {family}")
    else:
        supported = ", ".join(SUPPORTED_CHECKPOINTS.values())
        parts.append(f"supported v0.1 checkpoint families: {supported}")
    for label in (
        "config_errors",
        "missing_keys",
        "unexpected_keys",
        "unsupported_keys",
        "shape_mismatches",
        "dtype_mismatches",
    ):
        values = validation.get(label) or []
        if values:
            preview = values[:5]
            suffix = "" if len(values) <= 5 else f" ... +{len(values) - 5} more"
            parts.append(f"{label}: {preview}{suffix}")
    parts.append(
        "next steps: inspect the checkpoint with scripts/inspect_checkpoint.py, verify metadata.config_json "
        "matches one of the supported families, then rerun conversion. See README.md Quickstart and "
        "docs/checkpoint_support.md."
    )
    return "\n".join(parts)


def records_to_arrays(records: Mapping[str, TensorRecord], *, checkpoint_family: str) -> dict[str, Any]:
    np = import_numpy()
    arrays: dict[str, Any] = {}
    for key in sorted(EXPECTED_SHAPES_BY_FAMILY[checkpoint_family]):
        array = records[key].array
        if array is None:
            raise ConversionError(f"Tensor data was not loaded for {key!r}")
        arrays[key] = np.asarray(array)
    return arrays


def validate_output_target(source: Path, output: Path) -> None:
    source_resolved = source.resolve(strict=True)
    output_resolved = output.resolve(strict=False)
    if source_resolved == output_resolved:
        raise ConversionError(
            "Output path must not be the source checkpoint path; refusing to overwrite input weights"
        )


def write_npz_atomic(output: Path, arrays: Mapping[str, Any]) -> None:
    np = import_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
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


def build_report(
    source: Path,
    output: Path | None,
    records: Mapping[str, TensorRecord],
    validation: Mapping[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    total_parameters = 0
    for record in records.values():
        dtype_counts[record.dtype] = dtype_counts.get(record.dtype, 0) + 1
        total_parameters += record.parameter_count
    return {
        "source": str(source),
        "output": None if output is None else str(output),
        "format": "npz",
        "dry_run": dry_run,
        "checkpoint_family": validation.get("checkpoint_family"),
        "supported_checkpoint": validation.get("supported_checkpoint"),
        "tensor_count": len(records),
        "total_parameters": total_parameters,
        "dtypes": dict(sorted(dtype_counts.items())),
        "validation": validation,
    }


def print_text_report(report: Mapping[str, Any]) -> None:
    validation = report["validation"]
    status = "ok" if validation["ok"] else "failed"
    print(f"source: {report['source']}")
    print(f"output: {report['output'] or '(not written)'}")
    print(f"format: {report['format']}")
    print(f"dry_run: {report['dry_run']}")
    print(f"checkpoint_family: {report.get('checkpoint_family') or '(unknown)'}")
    print(f"supported_checkpoint: {report.get('supported_checkpoint') or '(unknown)'}")
    print(f"validation: {status}")
    print(f"tensor_count: {report['tensor_count']}")
    print(f"total_parameters: {report['total_parameters']:,}")
    print(f"dtypes: {report['dtypes']}")
    if not validation["ok"]:
        print()
        print(validation_error_message(validation))


def run_self_tests() -> None:
    base_shapes = EXPECTED_SHAPES_BY_FAMILY[CHECKPOINT_FAMILY_BASE]
    voice_shapes = EXPECTED_SHAPES_BY_FAMILY[CHECKPOINT_FAMILY_VOICEDESIGN]
    v3_shapes = EXPECTED_SHAPES_BY_FAMILY[CHECKPOINT_FAMILY_V3]
    assert len(base_shapes) == 613
    assert len(voice_shapes) == 636
    assert len(v3_shapes) == 637
    assert base_shapes["blocks.0.attention.wq.weight"] == (1280, 1280)
    assert base_shapes["blocks.11.attention.wk_speaker.weight"] == (1280, 768)
    assert voice_shapes["blocks.11.attention.wk_caption.weight"] == (1280, 512)
    assert voice_shapes["caption_encoder.blocks.9.mlp.w2.weight"] == (512, 1331)
    assert base_shapes["speaker_encoder.blocks.7.attention.q_norm.weight"] == (12, 64)
    assert voice_shapes["out_proj.bias"] == (32,)
    assert v3_shapes["duration_predictor.token_input_proj.weight"] == (1024, 512)
    assert v3_shapes["duration_predictor.token_blocks.2.modulation.weight"] == (3072, 768)
    assert v3_shapes["duration_predictor.token_out_proj.bias"] == (1,)

    base_records = {
        key: TensorRecord(name=key, shape=shape, dtype="F32")
        for key, shape in base_shapes.items()
    }
    base_validation = validate_records(
        base_records,
        {"latent_dim": 32, "model_dim": 1280, "num_layers": 12, "text_layers": 10, "speaker_layers": 8, "speaker_dim": 768},
    )
    assert base_validation["ok"], base_validation

    voice_records = {
        key: TensorRecord(name=key, shape=shape, dtype="F32")
        for key, shape in voice_shapes.items()
    }
    voice_validation = validate_records(
        voice_records,
        {
            "latent_dim": 32,
            "model_dim": 1280,
            "num_layers": 12,
            "text_layers": 10,
            "use_caption_condition": True,
            "caption_layers": 10,
            "caption_dim": 512,
            "caption_heads": 8,
            "caption_vocab_size": 99574,
        },
    )
    assert voice_validation["ok"], voice_validation

    v3_records = {
        key: TensorRecord(name=key, shape=shape, dtype="F32")
        for key, shape in v3_shapes.items()
    }
    v3_validation = validate_records(
        v3_records,
        {
            "latent_dim": 32,
            "model_dim": 1280,
            "num_layers": 12,
            "text_layers": 10,
            "speaker_layers": 8,
            "speaker_dim": 768,
            "use_duration_predictor": True,
            "duration_aux_dim": 14,
            "duration_hidden_dim": 1024,
            "duration_layers": 3,
            "duration_dropout": 0.1,
            "duration_attention_heads": 8,
            "duration_architecture": "token_sum_adarn_zero_no_aux",
            "duration_token_init_frames": 9.0,
            "duration_speaker_fusion": "adarn_zero",
        },
    )
    assert v3_validation["ok"], v3_validation

    missing_config = validate_records(base_records, None)
    assert not missing_config["ok"]
    assert missing_config["config_errors"]

    bad_voice = dict(voice_records)
    bad_voice["speaker_norm.weight"] = TensorRecord("speaker_norm.weight", (768,), "F32")
    bad_validation = validate_records(
        bad_voice,
        {
            "latent_dim": 32,
            "model_dim": 1280,
            "num_layers": 12,
            "text_layers": 10,
            "use_caption_condition": True,
            "caption_layers": 10,
            "caption_dim": 512,
            "caption_heads": 8,
        },
    )
    assert not bad_validation["ok"]
    assert any("ambiguous" in err for err in bad_validation["config_errors"])

    malformed_base = dict(base_records)
    malformed_base.pop("out_proj.bias")
    malformed_base["in_proj.weight"] = TensorRecord("in_proj.weight", (32, 1280), "F32")
    malformed = validate_records(
        malformed_base,
        {"latent_dim": 32, "model_dim": 1280, "num_layers": 12, "text_layers": 10, "speaker_layers": 8, "speaker_dim": 768},
    )
    assert not malformed["ok"]
    assert "out_proj.bias" in malformed["missing_keys"]
    assert malformed["shape_mismatches"][0]["key"] == "in_proj.weight"
    print("self-tests passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return 0

    if args.source is None:
        raise ConversionError("source path is required unless --self-test is used")

    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser() if args.output else None
    if output is None and not args.dry_run:
        raise ConversionError("output path is required unless --dry-run is used")

    validate_source_path(source)
    if output is not None and not args.dry_run:
        validate_output_target(source, output)

    config, records = load_checkpoint(source, load_arrays=not args.dry_run)
    validation = validate_records(records, config)
    report = build_report(source, output, records, validation, dry_run=args.dry_run)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)

    if not validation["ok"]:
        raise ConversionError(validation_error_message(validation))

    if not args.dry_run:
        assert output is not None
        checkpoint_family = validation["checkpoint_family"]
        if checkpoint_family is None:
            raise ConversionError("checkpoint family was not resolved after successful validation")
        write_npz_atomic(output, records_to_arrays(records, checkpoint_family=checkpoint_family))
        if not args.json_output:
            print(f"wrote: {output}")
    return 0


def cli_main() -> int:
    try:
        return main()
    except (ConversionError, InspectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
