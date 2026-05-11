#!/usr/bin/env python3
"""Convert upstream Irodori-TTS weights into an MLX-friendly archive.

The initial converter intentionally supports only the base v2 checkpoint layout
from ``Aratako/Irodori-TTS-500M-v2``. It validates the documented key mapping
before writing anything and copies tensors as-is into a NumPy ``.npz`` archive,
which can be loaded by MLX via ``mx.load``.
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

import numpy as np

try:
    from inspect_checkpoint import InspectionError, inspect_local_safetensors
except ImportError:  # pragma: no cover - fallback for unusual invocation paths
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inspect_checkpoint import InspectionError, inspect_local_safetensors

SUPPORTED_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"
SUPPORTED_SOURCE_SUFFIX = ".safetensors"
SUPPORTED_TENSOR_COUNT = 613
FLOAT32_NAMES = {"F32", "float32", "torch.float32", "dtype('float32')"}
UNSUPPORTED_CAPTION_PREFIXES = (
    "caption_encoder.",
    "caption_norm.",
)
UNSUPPORTED_CAPTION_FRAGMENTS = (
    ".attention.wk_caption.weight",
    ".attention.wv_caption.weight",
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
        description="Convert a local base Irodori-TTS safetensors checkpoint into an MLX-friendly .npz archive."
    )
    parser.add_argument("source", help="Local base-v2 .safetensors checkpoint path.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .npz path. Required unless --dry-run or --self-test is used.",
    )
    parser.add_argument(
        "--format",
        choices=("npz",),
        default="npz",
        help="Output format. Only npz is supported for the initial converter.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing output.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON report.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-light converter self-tests and exit. The source argument is ignored.",
    )
    return parser.parse_args()


def build_expected_shapes() -> dict[str, tuple[int, ...]]:
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
        for proj in ("wk_speaker", "wv_speaker"):
            add(f"{block}.attention.{proj}.weight", (1280, 768))
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

    add("cond_module.0.weight", (1280, 512))
    add("cond_module.2.weight", (1280, 1280))
    add("cond_module.4.weight", (3840, 1280))
    add("in_proj.weight", (1280, 32))
    add("in_proj.bias", (1280,))
    add("out_norm.weight", (1280,))
    add("out_proj.weight", (32, 1280))
    add("out_proj.bias", (32,))

    if len(expected) != SUPPORTED_TENSOR_COUNT:
        raise AssertionError(f"expected {SUPPORTED_TENSOR_COUNT} tensors, built {len(expected)}")
    return dict(sorted(expected.items()))


EXPECTED_SHAPES = build_expected_shapes()


def is_safetensors_path(path: Path) -> bool:
    return path.suffix == SUPPORTED_SOURCE_SUFFIX


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dtype_name(value: Any) -> str:
    if isinstance(value, np.dtype):
        return value.name
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


def load_checkpoint(path: Path, *, load_arrays: bool) -> tuple[dict[str, Any] | None, dict[str, TensorRecord]]:
    if not path.exists():
        raise ConversionError(f"Source checkpoint does not exist: {path}")
    if not path.is_file():
        raise ConversionError(f"Source checkpoint is not a file: {path}")
    if not is_safetensors_path(path):
        raise ConversionError(
            f"Only local {SUPPORTED_SOURCE_SUFFIX} checkpoints are supported in the initial converter: {path}"
        )
    config, header_records = load_safetensors_header(path)
    if not load_arrays:
        return config, header_records
    return config, load_safetensors_arrays(path)


def validate_base_config(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return []
    errors: list[str] = []
    expected_values = {
        "latent_dim": 32,
        "model_dim": 1280,
        "num_layers": 12,
        "text_layers": 10,
        "speaker_layers": 8,
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            errors.append(f"config {key}: expected {expected!r}, got {config.get(key)!r}")
    if config.get("use_caption_condition") is True:
        errors.append("VoiceDesign/caption checkpoints are not supported: use_caption_condition=true")
    has_speaker_fields = any(key.startswith("speaker_") for key in config)
    if config.get("use_speaker_condition") is False or not has_speaker_fields:
        errors.append("base speaker conditioning fields are missing or disabled")
    return errors


def is_unsupported_caption_key(name: str) -> bool:
    return name.startswith(UNSUPPORTED_CAPTION_PREFIXES) or any(
        fragment in name for fragment in UNSUPPORTED_CAPTION_FRAGMENTS
    )


def validate_records(
    records: Mapping[str, TensorRecord], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    source_keys = set(records)
    expected_keys = set(EXPECTED_SHAPES)
    missing = sorted(expected_keys - source_keys)
    unexpected = sorted(source_keys - expected_keys)
    unsupported = sorted(key for key in source_keys if is_unsupported_caption_key(key))
    shape_mismatches = []
    dtype_mismatches = []

    for key in sorted(source_keys & expected_keys):
        record = records[key]
        expected_shape = EXPECTED_SHAPES[key]
        if record.shape != expected_shape:
            shape_mismatches.append(
                {"key": key, "expected": list(expected_shape), "actual": list(record.shape)}
            )
        if dtype_name(record.dtype) not in FLOAT32_NAMES:
            dtype_mismatches.append({"key": key, "expected": "float32/F32", "actual": record.dtype})

    config_errors = validate_base_config(config)
    ok = not (missing or unexpected or unsupported or shape_mismatches or dtype_mismatches or config_errors)
    return {
        "ok": ok,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "unsupported_keys": unsupported,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "config_errors": config_errors,
    }


def validation_error_message(validation: Mapping[str, Any]) -> str:
    parts: list[str] = ["Checkpoint validation failed"]
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
    return "\n".join(parts)


def records_to_arrays(records: Mapping[str, TensorRecord]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in sorted(EXPECTED_SHAPES):
        array = records[key].array
        if array is None:
            raise ConversionError(f"Tensor data was not loaded for {key!r}")
        arrays[key] = np.asarray(array)
    return arrays


def write_npz_atomic(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
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
        "supported_checkpoint": SUPPORTED_CHECKPOINT,
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
    print(f"validation: {status}")
    print(f"tensor_count: {report['tensor_count']}")
    print(f"total_parameters: {report['total_parameters']:,}")
    print(f"dtypes: {report['dtypes']}")
    if not validation["ok"]:
        print()
        print(validation_error_message(validation))


def run_self_tests() -> None:
    assert len(EXPECTED_SHAPES) == 613
    assert EXPECTED_SHAPES["blocks.0.attention.wq.weight"] == (1280, 1280)
    assert EXPECTED_SHAPES["blocks.11.attention.wk_speaker.weight"] == (1280, 768)
    assert EXPECTED_SHAPES["text_encoder.blocks.9.mlp.w2.weight"] == (512, 1331)
    assert EXPECTED_SHAPES["speaker_encoder.blocks.7.attention.q_norm.weight"] == (12, 64)
    assert EXPECTED_SHAPES["out_proj.bias"] == (32,)

    records = {
        key: TensorRecord(name=key, shape=shape, dtype="F32")
        for key, shape in EXPECTED_SHAPES.items()
    }
    validation = validate_records(records, {"latent_dim": 32, "model_dim": 1280, "num_layers": 12, "text_layers": 10, "speaker_layers": 8, "speaker_dim": 768})
    assert validation["ok"], validation

    bad_records = dict(records)
    bad_records.pop("out_proj.bias")
    bad_records["caption_norm.weight"] = TensorRecord("caption_norm.weight", (512,), "F32")
    bad_records["in_proj.weight"] = TensorRecord("in_proj.weight", (32, 1280), "F32")
    bad = validate_records(bad_records, {"use_caption_condition": True})
    assert not bad["ok"]
    assert "out_proj.bias" in bad["missing_keys"]
    assert "caption_norm.weight" in bad["unexpected_keys"]
    assert "caption_norm.weight" in bad["unsupported_keys"]
    assert bad["shape_mismatches"][0]["key"] == "in_proj.weight"
    assert bad["config_errors"]
    print("self-tests passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return 0

    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser() if args.output else None
    if output is None and not args.dry_run:
        raise ConversionError("output path is required unless --dry-run is used")

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
        write_npz_atomic(output, records_to_arrays(records))
        if not args.json_output:
            print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConversionError, InspectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
