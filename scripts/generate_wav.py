#!/usr/bin/env python3
"""Generate a WAV with MLX RF-DiT latents and the PyTorch DACVAE bridge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DACVAEBridgeConfig = None
GenerationRequest = None
MLXDACVAERuntime = None
MLXRuntimeConfig = None
iter_messages = None
load_model_config_json = None


def _ensure_runtime_imports() -> None:
    """Import runtime dependencies lazily so --help works before optional setup is complete."""
    global DACVAEBridgeConfig, GenerationRequest, MLXDACVAERuntime, MLXRuntimeConfig, iter_messages, load_model_config_json
    from irodori_mlx import runtime as runtime_module

    if DACVAEBridgeConfig is None:
        DACVAEBridgeConfig = runtime_module.DACVAEBridgeConfig
    if GenerationRequest is None:
        GenerationRequest = runtime_module.GenerationRequest
    if MLXDACVAERuntime is None:
        MLXDACVAERuntime = runtime_module.MLXDACVAERuntime
    if MLXRuntimeConfig is None:
        MLXRuntimeConfig = runtime_module.MLXRuntimeConfig
    if iter_messages is None:
        iter_messages = runtime_module.iter_messages
    if load_model_config_json is None:
        load_model_config_json = runtime_module.load_model_config_json


CONFIG_KEYS = {
    "weights",
    "weights_dir",
    "weights_repo",
    "output",
    "output_wav",
    "text",
    "preset",
    "reference_wav",
    "no_reference",
    "caption",
    "model_config_json",
    "text_tokenizer_repo",
    "caption_tokenizer_repo",
    "text_max_length",
    "caption_max_length",
    "codec_repo",
    "codec_device",
    "codec_runtime_mode",
    "disable_codec_normalize",
    "enable_watermark",
    "seconds",
    "duration_scale",
    "num_steps",
    "cfg_scale_text",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_guidance_mode",
    "cfg_min_t",
    "cfg_max_t",
    "seed",
    "max_reference_seconds",
    "no_context_kv_cache",
    "print_boundaries",
    "metadata_json",
    "json_output",
    "requests_json",
}

REQUEST_KEYS = {
    "output",
    "output_wav",
    "text",
    "preset",
    "reference_wav",
    "no_reference",
    "caption",
    "seconds",
    "duration_scale",
    "num_steps",
    "cfg_scale_text",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_guidance_mode",
    "cfg_min_t",
    "cfg_max_t",
    "seed",
    "max_reference_seconds",
    "no_context_kv_cache",
}

REQUIRED_STRING_KEYS = {"weights", "output", "text"}
OPTIONAL_STRING_KEYS = {
    "weights_dir",
    "weights_repo",
    "reference_wav",
    "caption",
    "model_config_json",
    "text_tokenizer_repo",
    "caption_tokenizer_repo",
    "codec_repo",
    "codec_device",
    "metadata_json",
    "requests_json",
}
BOOL_KEYS = {
    "no_reference",
    "disable_codec_normalize",
    "enable_watermark",
    "no_context_kv_cache",
    "print_boundaries",
    "json_output",
}
INT_KEYS = {"text_max_length", "caption_max_length", "num_steps", "seed"}
FLOAT_KEYS = {
    "duration_scale",
    "cfg_scale_text",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_min_t",
    "cfg_max_t",
    "max_reference_seconds",
}
NULLABLE_FLOAT_KEYS = {"seconds"}
CHOICE_KEYS = {
    "preset": {"fast", "balanced", "quality"},
    "codec_runtime_mode": {"persistent", "subprocess"},
    "cfg_guidance_mode": {"independent", "joint", "reduced"},
}

PRESET_NUM_STEPS = {
    "fast": 12,
    "balanced": 24,
    "quality": 40,
}


def _load_json_value(value: str | None, *, label: str) -> Any:
    if value is None:
        return {}
    raw = str(value).strip()
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Inline {label} is invalid JSON.") from exc
    with Path(value).expanduser().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_json_object(value: str | None, *, label: str) -> dict[str, Any]:
    payload = _load_json_value(value, label=label)
    if not isinstance(payload, dict):
        source = f"inline {label}" if str(value or "").strip().startswith("{") else str(Path(value or "").expanduser())
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return payload


def _validate_generation_config(payload: dict[str, Any]) -> dict[str, Any]:
    for key in REQUIRED_STRING_KEYS:
        if key in payload:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"generation config field '{key}' must be a non-empty string")

    for key in OPTIONAL_STRING_KEYS:
        if key in payload and payload[key] is not None:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"generation config field '{key}' must be a string when provided")

    for key in BOOL_KEYS:
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"generation config field '{key}' must be a boolean")

    for key in INT_KEYS:
        if key in payload:
            value = payload[key]
            if value is None or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"generation config field '{key}' must be an integer")

    for key in FLOAT_KEYS:
        if key in payload:
            value = payload[key]
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"generation config field '{key}' must be a number")

    for key in NULLABLE_FLOAT_KEYS:
        if key in payload:
            value = payload[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"generation config field '{key}' must be a number or null")

    for key, choices in CHOICE_KEYS.items():
        if key in payload and payload[key] not in choices:
            allowed = ", ".join(sorted(choices))
            raise ValueError(f"generation config field '{key}' must be one of: {allowed}")

    return payload


def load_generation_config_json(value: str | None) -> dict[str, Any]:
    payload = _load_json_object(value, label="generation config")
    if "output_wav" in payload and "output" not in payload:
        payload["output"] = payload.pop("output_wav")
    unexpected = sorted(set(payload) - CONFIG_KEYS)
    if unexpected:
        raise ValueError("Unsupported generation config keys: " + ", ".join(unexpected))
    return _validate_generation_config(payload)


def load_generation_requests_json(value: str | None) -> list[dict[str, Any]]:
    payload = _load_json_value(value, label="generation requests")
    if isinstance(payload, dict) and "requests" in payload:
        payload = payload["requests"]
    if not isinstance(payload, list):
        raise ValueError("generation requests must be a JSON array or an object with a 'requests' array")
    requests: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"generation request #{index} must be a JSON object")
        request = dict(item)
        if "output_wav" in request and "output" not in request:
            request["output"] = request.pop("output_wav")
        unexpected = sorted(set(request) - REQUEST_KEYS)
        if unexpected:
            raise ValueError(f"Unsupported generation request #{index} keys: " + ", ".join(unexpected))
        _validate_generation_config(request)
        requests.append(request)
    if not requests:
        raise ValueError("generation requests must contain at least one request")
    return requests


def _default(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config.get(key, fallback)


def _has_cli_override(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def _resolve_num_steps(*, preset: str | None, current_num_steps: int, config: dict[str, Any], argv: list[str]) -> int:
    if not preset:
        return current_num_steps
    if _has_cli_override(argv, "--num-steps"):
        return current_num_steps
    if "num_steps" in config and not _has_cli_override(argv, "--preset"):
        return current_num_steps
    return PRESET_NUM_STEPS[preset]


def _add_configurable_bool(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    config: dict[str, Any],
    enable_flag: str,
    disable_flag: str,
    help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(enable_flag, dest=dest, action="store_true", default=config.get(dest, False), help=help_text)
    group.add_argument(disable_flag, dest=dest, action="store_false", help=argparse.SUPPRESS)


def build_parser(config: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    config = config or {}
    parser = argparse.ArgumentParser(
        description="Generate a WAV using MLX Irodori-TTS RF-DiT and the upstream/PyTorch DACVAE bridge.",
        epilog=(
            "Use --config-json with an inline JSON object or file path for repeatable runs, "
            "then override individual flags on the CLI when needed. Use --json or --metadata-json "
            "for automation-friendly output."
        ),
    )
    parser.add_argument("--config-json", help="Optional inline JSON object or path with common generation/runtime defaults.")
    parser.add_argument("--requests-json", default=config.get("requests_json"), help="Optional inline JSON array or path with repeated generation requests. Reuses one initialized runtime.")
    weights_group = parser.add_mutually_exclusive_group(required=not any(config.get(key) for key in ("weights", "weights_dir", "weights_repo")))
    weights_group.add_argument(
        "--weights",
        default=config.get("weights"),
        help=(
            "Converted local MLX .npz RF-DiT weights. This direct path remains the local-conversion "
            "fallback when a hosted/pre-converted layout is unavailable."
        ),
    )
    weights_group.add_argument(
        "--weights-dir",
        default=config.get("weights_dir"),
        help=(
            "Local directory using the hosted pre-converted weights layout "
            "(irodori_mlx_manifest.json, model_config.json, weights.npz, metadata)."
        ),
    )
    weights_group.add_argument(
        "--weights-repo",
        "--model",
        dest="weights_repo",
        default=config.get("weights_repo"),
        help=(
            "Hugging Face repo id with a pre-converted MLX weights layout, for example "
            "t0yohei/irodori-tts-mlx-v3-500m. Alias: --model. If resolution fails, "
            "use --weights with a locally converted .npz fallback."
        ),
    )
    parser.add_argument("--output", "--output-wav", dest="output", default=config.get("output"), help="Output WAV path for one-shot mode, or a default for batch requests.")
    parser.add_argument("--text", default=config.get("text"), help="Text prompt for one-shot mode, or a default for batch requests.")
    parser.add_argument(
        "--preset",
        default=config.get("preset"),
        choices=tuple(PRESET_NUM_STEPS),
        help=(
            "Local generation preset: fast=12 steps, balanced=24 steps, quality=40 steps. "
            "Explicit --num-steps still wins when provided."
        ),
    )
    parser.add_argument("--reference-wav", default=config.get("reference_wav"), help="Speaker/reference audio path for DACVAE encoding.")
    _add_configurable_bool(
        parser,
        dest="no_reference",
        config=config,
        enable_flag="--no-reference",
        disable_flag="--use-reference",
        help_text="Use an unconditional speaker path instead of reference audio.",
    )
    parser.add_argument("--caption", default=config.get("caption"), help="Optional caption/style text for caption-conditioned checkpoints.")
    parser.add_argument("--model-config-json", default=config.get("model_config_json"), help="Optional inline/path JSON for ModelConfig. Defaults to the base v2 config.")
    parser.add_argument("--text-tokenizer-repo", default=config.get("text_tokenizer_repo"), help="Override the text tokenizer repo from ModelConfig.")
    parser.add_argument("--caption-tokenizer-repo", default=config.get("caption_tokenizer_repo"), help="Override the caption tokenizer repo from ModelConfig.")
    parser.add_argument("--text-max-length", type=int, default=_default(config, "text_max_length", 256), help="Maximum text tokens to encode (default: 256).")
    parser.add_argument("--caption-max-length", type=int, default=config.get("caption_max_length"), help="Optional caption token limit. Defaults to text-max-length inside the runtime.")
    parser.add_argument("--codec-repo", default=_default(config, "codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim"), help="DACVAE codec repo id.")
    parser.add_argument("--codec-device", default=_default(config, "codec_device", "cpu"), help="PyTorch codec device: cpu, mps, or cuda.")
    parser.add_argument(
        "--codec-runtime-mode",
        default=_default(config, "codec_runtime_mode", "persistent"),
        choices=("persistent", "subprocess"),
        help="How to host the PyTorch DACVAE boundary: keep it in-process or isolate encode/decode in helper subprocesses.",
    )
    _add_configurable_bool(
        parser,
        dest="disable_codec_normalize",
        config=config,
        enable_flag="--disable-codec-normalize",
        disable_flag="--codec-normalize",
        help_text="Disable the default -16 dB codec normalization step.",
    )
    _add_configurable_bool(
        parser,
        dest="enable_watermark",
        config=config,
        enable_flag="--enable-watermark",
        disable_flag="--disable-watermark",
        help_text="Enable codec watermarking when the upstream codec supports it.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=config.get("seconds"),
        help="Manual output duration in seconds. Omit it to use predicted duration when the loaded model supports it.",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=float(_default(config, "duration_scale", 1.0)),
        help="Scale the predicted duration when --seconds is omitted (default: 1.0).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(_default(config, "num_steps", 40)),
        help="RF sampling steps. Defaults to 40 unless a preset supplies 12/24/40 first.",
    )
    parser.add_argument("--cfg-scale-text", type=float, default=float(_default(config, "cfg_scale_text", 3.0)), help="Classifier-free guidance scale for text conditioning.")
    parser.add_argument("--cfg-scale-caption", type=float, default=float(_default(config, "cfg_scale_caption", 3.0)), help="Classifier-free guidance scale for caption conditioning.")
    parser.add_argument("--cfg-scale-speaker", type=float, default=float(_default(config, "cfg_scale_speaker", 5.0)), help="Classifier-free guidance scale for speaker/reference conditioning.")
    parser.add_argument("--cfg-guidance-mode", default=_default(config, "cfg_guidance_mode", "independent"), choices=("independent", "joint", "reduced"), help="Guidance mixing strategy.")
    parser.add_argument("--cfg-min-t", type=float, default=float(_default(config, "cfg_min_t", 0.5)), help="Lower timestep bound for CFG application.")
    parser.add_argument("--cfg-max-t", type=float, default=float(_default(config, "cfg_max_t", 1.0)), help="Upper timestep bound for CFG application.")
    parser.add_argument("--seed", type=int, default=int(_default(config, "seed", 0)), help="Random seed for latent sampling.")
    parser.add_argument("--max-reference-seconds", type=float, default=float(_default(config, "max_reference_seconds", 30.0)), help="Trim reference audio to this many seconds before DACVAE encode.")
    _add_configurable_bool(
        parser,
        dest="no_context_kv_cache",
        config=config,
        enable_flag="--no-context-kv-cache",
        disable_flag="--context-kv-cache",
        help_text="Disable precomputed condition K/V caches in the RF-DiT sampler.",
    )
    _add_configurable_bool(
        parser,
        dest="print_boundaries",
        config=config,
        enable_flag="--print-boundaries",
        disable_flag="--no-print-boundaries",
        help_text="Print JSON boundary/config metadata before generation.",
    )
    parser.add_argument("--metadata-json", default=config.get("metadata_json"), help="Optional path to write generation metadata/timings as JSON.")
    _add_configurable_bool(
        parser,
        dest="json_output",
        config=config,
        enable_flag="--json",
        disable_flag="--no-json",
        help_text="Print generation metadata/timings as JSON instead of human-readable lines.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-json")
    pre_args, _ = pre.parse_known_args(argv)
    try:
        config = load_generation_config_json(pre_args.config_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        pre.error(str(exc))
    parser = build_parser(config)
    args = parser.parse_args(argv)
    args.model_config_json_cli_override = _has_cli_override(argv, "--model-config-json")
    args.num_steps = _resolve_num_steps(
        preset=args.preset,
        current_num_steps=int(args.num_steps),
        config=config,
        argv=argv,
    )
    if args.reference_wav and args.no_reference:
        parser.error("choose either --reference-wav or --no-reference, not both")
    if _has_cli_override(argv, "--weights"):
        args.weights_dir = None
        args.weights_repo = None
    elif _has_cli_override(argv, "--weights-dir"):
        args.weights = None
        args.weights_repo = None
    elif _has_cli_override(argv, "--weights-repo") or _has_cli_override(argv, "--model"):
        args.weights = None
        args.weights_dir = None
    selected_weights = [value for value in (args.weights, args.weights_dir, args.weights_repo) if value is not None and str(value).strip()]
    if not selected_weights:
        parser.error("choose one of --weights, --weights-dir, or --weights-repo/--model")
    if len(selected_weights) > 1:
        parser.error("choose only one of --weights, --weights-dir, or --weights-repo/--model")
    if not args.requests_json:
        if args.output is None or not str(args.output).strip():
            parser.error("--output must not be empty unless --requests-json supplies per-request outputs")
        if args.text is None or not str(args.text).strip():
            parser.error("--text must not be empty unless --requests-json supplies per-request text")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be > 0")
    if args.duration_scale <= 0:
        parser.error("--duration-scale must be > 0")
    if args.num_steps <= 0:
        parser.error("--num-steps must be > 0")
    if args.text_max_length <= 0:
        parser.error("--text-max-length must be > 0")
    if args.caption_max_length is not None and int(args.caption_max_length) <= 0:
        parser.error("--caption-max-length must be > 0 when provided")
    if args.max_reference_seconds <= 0:
        parser.error("--max-reference-seconds must be > 0")
    return args


HOSTED_WEIGHTS_REQUIRED_FILES = (
    "manifest",
    "weights",
    "model_config",
    "tokenizer_config",
    "conversion_metadata",
    "checksums",
)


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _hosted_manifest_relative_path(manifest_path: str, *, source_label: str, manifest_key: str) -> PurePosixPath:
    relative_path = PurePosixPath(manifest_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"{source_label} manifest file entry {manifest_key!r} must stay within the hosted weights layout: {manifest_path!r}"
        )
    return relative_path


def _resolve_hosted_layout_file(
    layout_dir: Path, manifest_path: str, *, source_label: str, manifest_key: str
) -> Path:
    relative_path = _hosted_manifest_relative_path(
        manifest_path, source_label=source_label, manifest_key=manifest_key
    )
    layout_root = layout_dir.resolve()
    layout_path = layout_root / Path(*relative_path.parts)
    if not layout_path.absolute().is_relative_to(layout_root):
        raise ValueError(
            f"{source_label} manifest file entry {manifest_key!r} escapes the hosted weights layout: {manifest_path!r}"
        )
    return layout_path


def _parse_checksum_filenames(checksum_text: str) -> set[str]:
    filenames: set[str] = set()
    for line in checksum_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        filename = parts[1].strip()
        if filename.startswith("*"):
            filename = filename[1:]
        if filename:
            filenames.add(filename)
    return filenames


def _load_hosted_weights_manifest(layout_dir: Path, *, source_label: str, require_approved_license: bool) -> dict[str, Any]:
    manifest_path = layout_dir / "irodori_mlx_manifest.json"
    manifest = _read_json_file(manifest_path, label="hosted weights manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{source_label} has unsupported irodori_mlx_manifest.json schema_version; expected 1")
    if manifest.get("format") != "irodori-tts-mlx-weights":
        raise ValueError(f"{source_label} is not an irodori-tts-mlx pre-converted weights layout")
    if manifest.get("format_version") != "0.2":
        raise ValueError(f"{source_label} has unsupported weights format_version {manifest.get('format_version')!r}; expected '0.2'")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{source_label} manifest must include a files object")
    files.setdefault("manifest", "irodori_mlx_manifest.json")
    missing_keys = [key for key in HOSTED_WEIGHTS_REQUIRED_FILES if not isinstance(files.get(key), str) or not files[key].strip()]
    if missing_keys:
        raise ValueError(f"{source_label} manifest is missing file entries: {', '.join(missing_keys)}")
    resolved_files = {
        key: _resolve_hosted_layout_file(layout_dir, files[key], source_label=source_label, manifest_key=key)
        for key in HOSTED_WEIGHTS_REQUIRED_FILES
    }
    missing_files = [files[key] for key in HOSTED_WEIGHTS_REQUIRED_FILES if not resolved_files[key].is_file()]
    if missing_files:
        raise ValueError(
            f"{source_label} pre-converted weights layout is missing required files: {', '.join(missing_files)}. "
            "Use --weights with a locally converted .npz fallback after running irodori-tts-convert or scripts/convert_weights.py."
        )
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{source_label} manifest must include runtime metadata")
    license_review = manifest.get("license_review")
    if not isinstance(license_review, dict):
        raise ValueError(f"{source_label} manifest must include license_review metadata")
    status = license_review.get("status")
    if require_approved_license and status != "approved":
        raise ValueError(
            f"{source_label} hosted weights license_review.status is {status!r}, expected 'approved'. "
            "Do not use unpublished or unapproved hosted weights; use --weights with a locally converted .npz fallback instead."
        )
    checksum_text = resolved_files["checksums"].read_text(encoding="utf-8")
    checksum_filenames = _parse_checksum_filenames(checksum_text)
    checksum_manifest_files = [key for key in HOSTED_WEIGHTS_REQUIRED_FILES if key != "checksums"]
    not_listed = [files[key] for key in checksum_manifest_files if files[key] not in checksum_filenames]
    if not_listed:
        raise ValueError(f"{source_label} checksums file does not list required files: {', '.join(not_listed)}")
    return manifest


def _download_weights_repo_snapshot(repo_id: str) -> Path:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional user setup.
        raise ValueError(
            "--weights-repo/--model requires huggingface_hub. Install it or use --weights-dir for a local "
            "pre-converted layout, or --weights with a locally converted .npz fallback."
        ) from exc
    try:
        revision = HfApi().model_info(repo_id=repo_id).sha
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"Could not determine a pinned revision for hosted pre-converted MLX weights repo {repo_id!r}")
        manifest_path = Path(hf_hub_download(repo_id=repo_id, filename="irodori_mlx_manifest.json", revision=revision))
        manifest = _read_json_file(manifest_path, label="hosted weights manifest")
        license_review = manifest.get("license_review")
        if not isinstance(license_review, dict):
            raise ValueError(f"{repo_id} manifest must include license_review metadata")
        status = license_review.get("status")
        if status != "approved":
            raise ValueError(
                f"{repo_id} hosted weights license_review.status is {status!r}, expected 'approved'. "
                "Do not download unpublished or unapproved hosted weights; use --weights with a locally converted .npz fallback instead."
            )
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"{repo_id} manifest must include a files object")
        files = {**files, "manifest": files.get("manifest", "irodori_mlx_manifest.json")}
        declared_paths = []
        for key in HOSTED_WEIGHTS_REQUIRED_FILES:
            value = files.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            declared_paths.append(str(_hosted_manifest_relative_path(value, source_label=repo_id, manifest_key=key)))
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=["README.md", "LICENSE.md", *sorted(set(declared_paths))],
            )
        )
    except Exception as exc:
        raise ValueError(
            f"Could not resolve hosted pre-converted MLX weights repo {repo_id!r}: {exc}. "
            "Check the repo id, network/cache access, and artifact license status. Fallback: run local conversion "
            "and pass --weights /path/to/weights.npz with --model-config-json."
        ) from exc


def resolve_preconverted_weights_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.weights_dir:
        layout_dir = Path(args.weights_dir).expanduser()
        manifest = _load_hosted_weights_manifest(layout_dir, source_label=str(layout_dir), require_approved_license=False)
    elif args.weights_repo:
        layout_dir = _download_weights_repo_snapshot(str(args.weights_repo))
        manifest = _load_hosted_weights_manifest(layout_dir, source_label=str(args.weights_repo), require_approved_license=True)
    else:
        return args
    files = manifest["files"]
    args.weights = str(
        _resolve_hosted_layout_file(
            layout_dir, files["weights"], source_label=str(layout_dir), manifest_key="weights"
        )
    )
    if not getattr(args, "model_config_json_cli_override", False):
        args.model_config_json = str(
            _resolve_hosted_layout_file(
                layout_dir, files["model_config"], source_label=str(layout_dir), manifest_key="model_config"
            )
        )
    return args


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return {
        "output_wav": result.output_wav,
        "sample_rate": result.sample_rate,
        "samples": result.samples,
        "latent_steps": result.latent_steps,
        "patched_steps": result.patched_steps,
        "seed": result.seed,
        "duration_mode": getattr(result, "duration_mode", None),
        "requested_seconds": getattr(result, "requested_seconds", None),
        "resolved_seconds": getattr(result, "resolved_seconds", None),
        "timings_ms": result.timings_ms,
        "messages": list(result.messages),
    }


def build_result_payload(
    *, result: Any, request: GenerationRequest, runtime: MLXDACVAERuntime, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "result": _result_to_dict(result),
        "request": asdict(request),
        "boundaries": runtime.describe_boundaries(),
        "cli": {
            "config_json": args.config_json,
            "requests_json": args.requests_json,
            "metadata_json": args.metadata_json,
            "json_output": bool(args.json_output),
            "print_boundaries": bool(args.print_boundaries),
            "preset": args.preset,
        },
    }


def build_runtime_config(args: argparse.Namespace, model_config: Any) -> MLXRuntimeConfig:
    _ensure_runtime_imports()
    return MLXRuntimeConfig(
        model_config=model_config,
        weights_path=args.weights,
        text_tokenizer_repo=args.text_tokenizer_repo,
        caption_tokenizer_repo=args.caption_tokenizer_repo,
        text_max_length=int(args.text_max_length),
        caption_max_length=args.caption_max_length,
        codec=DACVAEBridgeConfig(
            codec_repo=args.codec_repo,
            codec_device=args.codec_device,
            runtime_mode=args.codec_runtime_mode,
            enable_watermark=bool(args.enable_watermark),
            normalize_db=None if args.disable_codec_normalize else -16.0,
        ),
    )


def _merged_request_value(args: argparse.Namespace, overrides: dict[str, Any], key: str) -> Any:
    return overrides[key] if key in overrides else getattr(args, key)


def build_generation_request(args: argparse.Namespace, overrides: dict[str, Any] | None = None) -> GenerationRequest:
    _ensure_runtime_imports()
    overrides = dict(overrides or {})
    text = _merged_request_value(args, overrides, "text")
    output = _merged_request_value(args, overrides, "output")
    if text is None or not str(text).strip():
        raise ValueError("generation request requires a non-empty text field")
    if output is None or not str(output).strip():
        raise ValueError("generation request requires a non-empty output field")
    reference_wav = _merged_request_value(args, overrides, "reference_wav")
    no_reference = bool(_merged_request_value(args, overrides, "no_reference"))
    if reference_wav and no_reference:
        raise ValueError("generation request cannot set both reference_wav and no_reference")
    preset = _merged_request_value(args, overrides, "preset")
    num_steps = int(_merged_request_value(args, overrides, "num_steps"))
    if "preset" in overrides and "num_steps" not in overrides and preset:
        num_steps = PRESET_NUM_STEPS[preset]
    seconds = _merged_request_value(args, overrides, "seconds")
    duration_scale = float(_merged_request_value(args, overrides, "duration_scale"))
    max_reference_seconds = _merged_request_value(args, overrides, "max_reference_seconds")
    if seconds is not None and float(seconds) <= 0:
        raise ValueError("seconds must be > 0 when provided")
    if duration_scale <= 0:
        raise ValueError("duration_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if max_reference_seconds is not None and float(max_reference_seconds) <= 0:
        raise ValueError("max_reference_seconds must be > 0 when provided")
    return GenerationRequest(
        text=str(text),
        output_wav=str(output),
        reference_wav=reference_wav,
        no_reference=no_reference,
        caption=_merged_request_value(args, overrides, "caption"),
        seconds=None if seconds is None else float(seconds),
        duration_scale=duration_scale,
        num_steps=num_steps,
        cfg_scale_text=float(_merged_request_value(args, overrides, "cfg_scale_text")),
        cfg_scale_caption=float(_merged_request_value(args, overrides, "cfg_scale_caption")),
        cfg_scale_speaker=float(_merged_request_value(args, overrides, "cfg_scale_speaker")),
        cfg_guidance_mode=_merged_request_value(args, overrides, "cfg_guidance_mode"),
        cfg_min_t=float(_merged_request_value(args, overrides, "cfg_min_t")),
        cfg_max_t=float(_merged_request_value(args, overrides, "cfg_max_t")),
        seed=int(_merged_request_value(args, overrides, "seed")),
        max_reference_seconds=max_reference_seconds,
        use_context_kv_cache=not bool(_merged_request_value(args, overrides, "no_context_kv_cache")),
    )


def write_metadata_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolve_preconverted_weights_args(args)
    _ensure_runtime_imports()
    model_config = load_model_config_json(args.model_config_json)
    request_overrides = load_generation_requests_json(args.requests_json) if args.requests_json else [{}]
    if model_config.use_speaker_condition:
        for item in request_overrides:
            request_reference = item.get("reference_wav", args.reference_wav)
            request_no_reference = bool(item.get("no_reference", args.no_reference))
            if not request_no_reference and not request_reference:
                raise SystemExit(
                    "error: speaker-conditioned checkpoints require reference_wav unless no_reference is true"
                )

    runtime_config = build_runtime_config(args, model_config)
    runtime = MLXDACVAERuntime(config=runtime_config)
    if args.print_boundaries:
        print(
            json.dumps(runtime.describe_boundaries(), indent=2, sort_keys=True, default=str),
            flush=True,
            file=sys.stderr if args.json_output else sys.stdout,
        )

    payloads: list[dict[str, Any]] = []
    for index, overrides in enumerate(request_overrides, start=1):
        try:
            request = build_generation_request(args, overrides)
        except ValueError as exc:
            raise SystemExit(f"error: generation request #{index}: {exc}") from exc
        result = runtime.generate(request)
        payload = build_result_payload(result=result, request=request, runtime=runtime, args=args)
        payload["batch"] = {"index": index, "count": len(request_overrides), "overrides": dict(overrides)}
        payloads.append(payload)

    output_payload: dict[str, Any] | list[dict[str, Any]] = payloads[0] if len(payloads) == 1 and not args.requests_json else {
        "results": payloads,
        "boundaries": runtime.describe_boundaries(),
        "cli": payloads[0]["cli"],
        "batch": {"count": len(payloads)},
    }
    if args.metadata_json:
        write_metadata_json(args.metadata_json, output_payload)
    if args.json_output:
        print(json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    for index, payload in enumerate(payloads, start=1):
        if len(payloads) > 1:
            print(f"[request {index}/{len(payloads)}] {payload['request']['text']}")
        result_dict = payload["result"]
        for line in iter_messages(type("ResultView", (), result_dict)()):
            print(line)
    return 0


def cli_main() -> int:
    try:
        return main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
