#!/usr/bin/env python3
"""Inspect Irodori-TTS checkpoint metadata without loading tensor payloads.

The primary target is Hugging Face or local ``model.safetensors`` files. For
safetensors files this script reads only the header, so it can inspect multi-GiB
checkpoints without downloading or allocating the tensor data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

DEFAULT_HF_FILE = "model.safetensors"
DEFAULT_REVISION = "main"
MAX_HEADER_BYTES = 64 * 1024 * 1024
USER_AGENT = "irodori-tts-mlx-checkpoint-inspector/0.1"


class InspectionError(RuntimeError):
    """Raised when a checkpoint cannot be inspected safely."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: list[int]
    data_offsets: list[int]

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def byte_count(self) -> int:
        if len(self.data_offsets) != 2:
            return 0
        return int(self.data_offsets[1]) - int(self.data_offsets[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "data_offsets": self.data_offsets,
            "parameter_count": self.parameter_count,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class CheckpointInspection:
    source: dict[str, Any]
    metadata: dict[str, Any]
    config: dict[str, Any] | None
    tensors: list[TensorInfo]

    @property
    def total_parameters(self) -> int:
        return sum(tensor.parameter_count for tensor in self.tensors)

    @property
    def tensor_payload_bytes(self) -> int:
        return sum(tensor.byte_count for tensor in self.tensors)

    @property
    def dtypes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tensor in self.tensors:
            counts[tensor.dtype] = counts.get(tensor.dtype, 0) + 1
        return dict(sorted(counts.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metadata": self.metadata,
            "config": self.config,
            "totals": {
                "tensor_count": len(self.tensors),
                "total_parameters": self.total_parameters,
                "tensor_payload_bytes": self.tensor_payload_bytes,
                "dtypes": self.dtypes,
            },
            "tensors": [tensor.to_json() for tensor in self.tensors],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a local safetensors checkpoint path or a Hugging Face repo id "
            "without loading tensor payloads."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Local checkpoint path or Hugging Face repo id, for example "
            "Aratako/Irodori-TTS-500M-v2."
        ),
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_HF_FILE,
        help=f"Checkpoint filename for Hugging Face repo sources (default: {DEFAULT_HF_FILE}).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Hugging Face revision for repo sources (default: {DEFAULT_REVISION}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON instead of the text summary.",
    )
    parser.add_argument(
        "--max-tensors",
        type=non_negative_int,
        default=40,
        help="Maximum tensors to print in text mode (default: 40). Use 0 for no tensor rows.",
    )
    parser.add_argument(
        "--all-tensors",
        action="store_true",
        help="Print all tensor rows in text mode.",
    )
    return parser.parse_args()


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def validate_header_length(header_length: int) -> None:
    if header_length <= 0:
        raise InspectionError("Invalid safetensors header length: must be positive")
    if header_length > MAX_HEADER_BYTES:
        raise InspectionError(
            f"Safetensors header is too large ({header_length:,} bytes); "
            f"refusing to read more than {MAX_HEADER_BYTES:,} bytes"
        )


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise InspectionError(f"Expected {size} bytes, got {len(data)} bytes")
    return data


def parse_safetensors_header(header_bytes: bytes) -> tuple[dict[str, Any], list[TensorInfo]]:
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InspectionError(f"Invalid safetensors header JSON: {exc}") from exc

    if not isinstance(header, dict):
        raise InspectionError("Safetensors header must be a JSON object")

    metadata = header.pop("__metadata__", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise InspectionError("Safetensors __metadata__ must be an object when present")

    tensors: list[TensorInfo] = []
    for name, raw_info in sorted(header.items()):
        if not isinstance(raw_info, dict):
            raise InspectionError(f"Tensor entry {name!r} is not an object")
        dtype = raw_info.get("dtype")
        shape = raw_info.get("shape")
        data_offsets = raw_info.get("data_offsets")
        if not isinstance(dtype, str):
            raise InspectionError(f"Tensor {name!r} is missing string dtype")
        if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
            raise InspectionError(f"Tensor {name!r} has invalid shape")
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or not all(isinstance(value, int) for value in data_offsets)
        ):
            raise InspectionError(f"Tensor {name!r} has invalid data_offsets")
        start, end = data_offsets
        if start < 0 or end < 0 or start > end:
            raise InspectionError(
                f"Tensor {name!r} has invalid data_offsets: offsets must be non-negative and monotonic"
            )
        tensors.append(TensorInfo(name=name, dtype=dtype, shape=shape, data_offsets=data_offsets))

    return metadata, tensors


def parse_config(metadata: dict[str, Any]) -> dict[str, Any] | None:
    config_json = metadata.get("config_json")
    if config_json is None:
        return None
    if not isinstance(config_json, str):
        raise InspectionError("metadata.config_json is present but is not a string")
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise InspectionError(f"metadata.config_json is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise InspectionError("metadata.config_json must decode to a JSON object")
    return config


def inspect_local_safetensors(path: Path) -> CheckpointInspection:
    try:
        with path.open("rb") as file:
            header_length = int.from_bytes(read_exact(file, 8), byteorder="little", signed=False)
            validate_header_length(header_length)
            header_bytes = read_exact(file, header_length)
    except OSError as exc:
        raise InspectionError(f"Could not read local checkpoint {path}: {exc}") from exc

    metadata, tensors = parse_safetensors_header(header_bytes)
    return CheckpointInspection(
        source={"type": "local", "path": str(path), "format": "safetensors"},
        metadata=metadata,
        config=parse_config(metadata),
        tensors=tensors,
    )


def quote_hf_path(value: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in value.split("/"))


def hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return "https://huggingface.co/{repo}/resolve/{revision}/{filename}".format(
        repo=quote_hf_path(repo_id),
        revision=urllib.parse.quote(revision, safe=""),
        filename=quote_hf_path(filename),
    )


def fetch_range(url: str, start: int, end: int) -> bytes:
    expected_size = end - start + 1
    if expected_size <= 0:
        raise InspectionError("Invalid byte range requested")
    if expected_size > MAX_HEADER_BYTES:
        raise InspectionError(
            f"Requested range is too large ({expected_size:,} bytes); "
            f"refusing to read more than {MAX_HEADER_BYTES:,} bytes"
        )

    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range")
            content_length = response.headers.get("Content-Length")
            if status != 206:
                raise InspectionError(
                    "Remote server did not honor the HTTP Range request; "
                    "refusing to risk a full checkpoint download"
                )
            if content_range and not content_range.startswith(f"bytes {start}-{end}/"):
                raise InspectionError(f"Unexpected Content-Range from {url}: {content_range}")
            if content_length is not None and int(content_length) != expected_size:
                raise InspectionError(
                    f"Unexpected Content-Length from {url}: expected {expected_size}, got {content_length}"
                )
            data = response.read(expected_size)
    except urllib.error.HTTPError as exc:
        raise InspectionError(f"HTTP {exc.code} while fetching {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise InspectionError(f"Could not fetch {url}: {exc.reason}") from exc

    if len(data) != expected_size:
        raise InspectionError(f"Expected {expected_size} bytes from {url}, got {len(data)}")
    return data


def inspect_hf_safetensors(repo_id: str, revision: str, filename: str) -> CheckpointInspection:
    if not filename.endswith(".safetensors"):
        raise InspectionError("Remote inspection currently supports safetensors files only")

    url = hf_resolve_url(repo_id, revision, filename)
    header_length_bytes = fetch_range(url, 0, 7)
    header_length = int.from_bytes(header_length_bytes, byteorder="little", signed=False)
    validate_header_length(header_length)
    header_bytes = fetch_range(url, 8, 8 + header_length - 1)

    metadata, tensors = parse_safetensors_header(header_bytes)
    return CheckpointInspection(
        source={
            "type": "huggingface",
            "repo_id": repo_id,
            "revision": revision,
            "file": filename,
            "url": url,
            "format": "safetensors",
            "header_bytes": header_length,
        },
        metadata=metadata,
        config=parse_config(metadata),
        tensors=tensors,
    )


def inspect_source(source: str, revision: str, filename: str) -> CheckpointInspection:
    path = Path(source).expanduser()
    if path.exists():
        if path.suffix != ".safetensors":
            raise InspectionError("Local inspection currently supports .safetensors files only")
        return inspect_local_safetensors(path)
    if path.suffix == ".safetensors" or source.startswith(('.', '/', '~')):
        raise InspectionError(f"Local checkpoint path does not exist: {path}")

    if source.startswith("http://") or source.startswith("https://"):
        raise InspectionError("Pass a Hugging Face repo id instead of a direct URL")

    return inspect_hf_safetensors(source, revision=revision, filename=filename)


def format_int(value: int) -> str:
    return f"{value:,}"


def format_bytes(value: int) -> str:
    gib = value / (1024**3)
    if gib >= 1:
        return f"{format_int(value)} bytes ({gib:.3f} GiB)"
    mib = value / (1024**2)
    return f"{format_int(value)} bytes ({mib:.3f} MiB)"


def print_text_report(inspection: CheckpointInspection, max_tensors: int, all_tensors: bool) -> None:
    source = inspection.source
    print("Checkpoint inspection")
    print("=====================")
    if source["type"] == "local":
        print(f"Source: local path {source['path']}")
    else:
        print(f"Source: Hugging Face {source['repo_id']} ({source['revision']}:{source['file']})")
    print(f"Format: {source['format']}")
    print()

    print("Totals")
    print("------")
    print(f"Tensors: {format_int(len(inspection.tensors))}")
    print(f"Parameters: {format_int(inspection.total_parameters)}")
    print(f"Tensor payload: {format_bytes(inspection.tensor_payload_bytes)}")
    print(f"Dtypes: {inspection.dtypes}")
    print()

    if inspection.metadata:
        print("Metadata")
        print("--------")
        for key in sorted(inspection.metadata):
            value = inspection.metadata[key]
            if key == "config_json" and isinstance(value, str):
                print(f"{key}: <JSON object, {len(value)} characters>")
            else:
                print(f"{key}: {value}")
        print()

    if inspection.config:
        print("Config")
        print("------")
        for key in sorted(inspection.config):
            print(f"{key}: {inspection.config[key]}")
        print()

    if max_tensors == 0 and not all_tensors:
        return

    tensors = inspection.tensors if all_tensors else inspection.tensors[:max_tensors]
    if tensors:
        print("Tensors")
        print("-------")
        name_width = min(max(len(tensor.name) for tensor in tensors), 72)
        for tensor in tensors:
            name = tensor.name
            if len(name) > name_width:
                name = name[: name_width - 1] + "…"
            print(
                f"{name:<{name_width}}  {tensor.dtype:<4}  "
                f"shape={tensor.shape}  params={format_int(tensor.parameter_count)}"
            )
        hidden = len(inspection.tensors) - len(tensors)
        if hidden > 0:
            print()
            print(textwrap.dedent(
                f"""\
                ... {format_int(hidden)} more tensors hidden.
                Use --all-tensors to print every tensor or --json for complete machine-readable output.
                """
            ).strip())


def main() -> int:
    args = parse_args()
    try:
        inspection = inspect_source(args.source, revision=args.revision, filename=args.file)
    except InspectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(inspection.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(inspection, max_tensors=args.max_tensors, all_tensors=args.all_tensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
