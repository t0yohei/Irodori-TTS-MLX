#!/usr/bin/env python3
"""Generate a WAV with MLX RF-DiT latents and the PyTorch DACVAE bridge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DACVAEBridgeConfig = None
GenerationRequest = None
MLXDACVAERuntime = None
MLXRuntimeConfig = None
describe_codec_capabilities = None
iter_messages = None
load_model_config_json = None
release_mlx_runtime_memory = None
resolve_weights_layout_source = None


def _ensure_runtime_imports() -> None:
    """Import runtime dependencies lazily so --help works before optional setup is complete."""
    global DACVAEBridgeConfig, GenerationRequest, MLXDACVAERuntime, MLXRuntimeConfig, describe_codec_capabilities, iter_messages, load_model_config_json, release_mlx_runtime_memory, resolve_weights_layout_source
    from irodori_mlx import runtime as runtime_module
    from irodori_mlx.hosted_weights import resolve_weights_layout_source as resolve_layout

    if DACVAEBridgeConfig is None:
        DACVAEBridgeConfig = runtime_module.DACVAEBridgeConfig
    if GenerationRequest is None:
        GenerationRequest = runtime_module.GenerationRequest
    if MLXDACVAERuntime is None:
        MLXDACVAERuntime = runtime_module.MLXDACVAERuntime
    if MLXRuntimeConfig is None:
        MLXRuntimeConfig = runtime_module.MLXRuntimeConfig
    if describe_codec_capabilities is None:
        describe_codec_capabilities = runtime_module.describe_codec_capabilities
    if iter_messages is None:
        iter_messages = runtime_module.iter_messages
    if load_model_config_json is None:
        load_model_config_json = runtime_module.load_model_config_json
    if release_mlx_runtime_memory is None:
        release_mlx_runtime_memory = runtime_module.release_mlx_runtime_memory
    if resolve_weights_layout_source is None:
        resolve_weights_layout_source = resolve_layout


CONFIG_KEYS = {
    "weights",
    "weights_dir",
    "weights_repo",
    "weights_revision",
    "output",
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
    "codec_path",
    "codec_artifact_dir",
    "codec_artifact_repo",
    "codec_artifact_revision",
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
    "preflight",
    "metadata_json",
    "json_output",
    "requests_json",
    "cleanup_between_requests",
}

REQUEST_KEYS = {
    "output",
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
    "weights_revision",
    "reference_wav",
    "caption",
    "model_config_json",
    "text_tokenizer_repo",
    "caption_tokenizer_repo",
    "codec_repo",
    "codec_path",
    "codec_artifact_dir",
    "codec_artifact_repo",
    "codec_artifact_revision",
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
    "preflight",
    "json_output",
    "cleanup_between_requests",
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
    "codec_runtime_mode": {"persistent", "subprocess", "mlx", "mlx-decode", "mlx-decode-subprocess"},
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
    _add_configurable_bool(
        parser,
        dest="cleanup_between_requests",
        config=config,
        enable_flag="--cleanup-between-requests",
        disable_flag="--no-cleanup-between-requests",
        help_text="Synchronize MLX and clear reusable MLX cache memory after each --requests-json item.",
    )
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
            "Local directory or archive using the hosted pre-converted weights layout "
            "(irodori_mlx_manifest.json, model_config.json, weights.npz, metadata)."
        ),
    )
    weights_group.add_argument(
        "--weights-repo",
        dest="weights_repo",
        default=config.get("weights_repo"),
        help=(
            "Hugging Face repo id with a pre-converted MLX weights layout, for example "
            "t0yohei/Irodori-TTS-MLX-500M-v3. If resolution fails, use --weights "
            "with a locally converted .npz fallback."
        ),
    )
    parser.add_argument("--weights-revision", default=config.get("weights_revision"), help="Optional Hugging Face revision for --weights-repo.")
    parser.add_argument("--output", dest="output", default=config.get("output"), help="Output WAV path for one-shot mode, or a default for batch requests.")
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
    parser.add_argument(
        "--codec-path",
        default=config.get("codec_path"),
        help="Converted local MLX DACVAE codec .npz. Required when --codec-runtime-mode uses an MLX codec unless a codec artifact layout is provided.",
    )
    parser.add_argument(
        "--codec-artifact-dir",
        default=config.get("codec_artifact_dir"),
        help=(
            "Local directory or archive using the hosted DACVAE codec artifact layout "
            "(irodori_dacvae_codec_manifest.json, dacvae-codec.npz, codec_metadata.json, checksums)."
        ),
    )
    parser.add_argument(
        "--codec-artifact-repo",
        default=config.get("codec_artifact_repo"),
        help=(
            "Hugging Face repo id with an approved hosted DACVAE codec artifact layout. "
            "RF-DiT --weights-repo and DACVAE --codec-artifact-repo stay separate."
        ),
    )
    parser.add_argument("--codec-artifact-revision", default=config.get("codec_artifact_revision"), help="Optional Hugging Face revision for --codec-artifact-repo.")
    parser.add_argument("--codec-device", default=_default(config, "codec_device", "cpu"), help="PyTorch codec device: cpu, mps, or cuda.")
    parser.add_argument(
        "--codec-runtime-mode",
        default=_default(config, "codec_runtime_mode", "persistent"),
        choices=("persistent", "subprocess", "mlx", "mlx-decode", "mlx-decode-subprocess"),
        help=(
            "How to host DACVAE encode/decode: PyTorch in-process, PyTorch subprocesses, a converted local "
            "MLX fixture codec, or MLX decode-only with PyTorch encode fallback."
        ),
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
    _add_configurable_bool(
        parser,
        dest="preflight",
        config=config,
        enable_flag="--preflight",
        disable_flag="--no-preflight",
        help_text=(
            "Resolve weights, model config, tokenizer repos, codec runtime mode, and artifact paths, "
            "then exit before tokenizer loading, model loading, or WAV generation."
        ),
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
    explicit_weight_sources = {
        name
        for option, name in (
            ("--weights", "weights"),
            ("--weights-dir", "weights_dir"),
            ("--weights-repo", "weights_repo"),
        )
        if _has_cli_override(argv, option)
    }
    if explicit_weight_sources:
        for name in {"weights", "weights_dir", "weights_repo"} - explicit_weight_sources:
            setattr(args, name, None)
        if not _has_cli_override(argv, "--weights-revision"):
            args.weights_revision = None
        if {"weights_dir", "weights_repo"} & explicit_weight_sources:
            if not _has_cli_override(argv, "--model-config-json"):
                args.model_config_json = None
            if not _has_cli_override(argv, "--text-tokenizer-repo"):
                args.text_tokenizer_repo = None
            if not _has_cli_override(argv, "--caption-tokenizer-repo"):
                args.caption_tokenizer_repo = None
    explicit_codec_sources = {
        name
        for option, name in (
            ("--codec-path", "codec_path"),
            ("--codec-artifact-dir", "codec_artifact_dir"),
            ("--codec-artifact-repo", "codec_artifact_repo"),
        )
        if _has_cli_override(argv, option)
    }
    if explicit_codec_sources:
        for name in {"codec_path", "codec_artifact_dir", "codec_artifact_repo"} - explicit_codec_sources:
            setattr(args, name, None)
        if not _has_cli_override(argv, "--codec-artifact-revision"):
            args.codec_artifact_revision = None
    args.num_steps = _resolve_num_steps(
        preset=args.preset,
        current_num_steps=int(args.num_steps),
        config=config,
        argv=argv,
    )
    if args.reference_wav and args.no_reference:
        parser.error("choose either --reference-wav or --no-reference, not both")
    selected_weights = [value for value in (args.weights, args.weights_dir, args.weights_repo) if value is not None and str(value).strip()]
    if not selected_weights:
        parser.error("choose one of --weights, --weights-dir, or --weights-repo")
    if len(selected_weights) > 1:
        parser.error("choose only one of --weights, --weights-dir, or --weights-repo")
    if args.weights_revision and not args.weights_repo:
        parser.error("--weights-revision requires --weights-repo")
    selected_codec_artifacts = [
        value
        for value in (args.codec_path, args.codec_artifact_dir, args.codec_artifact_repo)
        if value is not None and str(value).strip()
    ]
    if len(selected_codec_artifacts) > 1:
        parser.error("choose only one of --codec-path, --codec-artifact-dir, or --codec-artifact-repo")
    if args.codec_artifact_revision and not args.codec_artifact_repo:
        parser.error("--codec-artifact-revision requires --codec-artifact-repo")
    args.model_config_json_cli_override = _has_cli_override(argv, "--model-config-json")
    if (args.weights_dir or args.weights_repo) and args.model_config_json:
        parser.error("--model-config-json is loaded from --weights-dir/--weights-repo layouts; use --weights for explicit .npz fallback")
    if not args.requests_json and not args.preflight:
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


def _download_weights_repo_snapshot(repo_id: str, *, revision: str | None = None):
    from irodori_mlx.hosted_weights import snapshot_weights_repo

    return snapshot_weights_repo(repo_id, revision=revision)


def resolve_weights_layout_source(
    *, weights_dir: str | Path | None = None, weights_repo: str | None = None, revision: str | None = None
):
    from irodori_mlx.hosted_weights import validate_weights_layout

    if weights_dir and weights_repo:
        raise ValueError("choose either weights_dir or weights_repo, not both")
    if weights_dir:
        return validate_weights_layout(weights_dir, source=str(weights_dir), source_kind="local")
    if weights_repo:
        snapshot = _download_weights_repo_snapshot(str(weights_repo), revision=revision)
        source = str(weights_repo) if revision is None else f"{weights_repo}@{revision}"
        try:
            return validate_weights_layout(snapshot, source=source, source_kind="repo")
        except ValueError as exc:
            raise ValueError(f"{exc}. Use --weights with a locally converted .npz fallback instead.") from exc
    return None


def resolve_preconverted_weights_args(args: argparse.Namespace) -> argparse.Namespace:
    """Compatibility adapter for the pre-v0.2 generate_wav tests/docs path."""

    layout = resolve_weights_layout_source(
        weights_dir=args.weights_dir,
        weights_repo=args.weights_repo,
        revision=getattr(args, "weights_revision", None),
    )
    if layout is None:
        return args
    args._resolved_weights_layout = layout
    args.weights = str(layout.weights_path)
    if not getattr(args, "model_config_json_cli_override", False):
        args.model_config_json = str(layout.model_config_path)
    return args


def _download_codec_repo_snapshot(repo_id: str, *, revision: str | None = None):
    from irodori_mlx.hosted_codec import snapshot_codec_repo

    return snapshot_codec_repo(repo_id, revision=revision)


def resolve_codec_artifact_source(
    *,
    codec_artifact_dir: str | Path | None = None,
    codec_artifact_repo: str | None = None,
    revision: str | None = None,
):
    from irodori_mlx.hosted_codec import validate_codec_artifact_layout

    if codec_artifact_dir and codec_artifact_repo:
        raise ValueError("choose either codec_artifact_dir or codec_artifact_repo, not both")
    if codec_artifact_dir:
        return validate_codec_artifact_layout(codec_artifact_dir, source=str(codec_artifact_dir), source_kind="local")
    if codec_artifact_repo:
        snapshot = _download_codec_repo_snapshot(str(codec_artifact_repo), revision=revision)
        source = str(codec_artifact_repo) if revision is None else f"{codec_artifact_repo}@{revision}"
        try:
            return validate_codec_artifact_layout(snapshot, source=source, source_kind="repo")
        except ValueError as exc:
            raise ValueError(f"{exc}. Use --codec-path or --codec-artifact-dir for a local DACVAE codec fallback instead.") from exc
    return None


def resolve_codec_artifact_args(args: argparse.Namespace):
    layout = resolve_codec_artifact_source(
        codec_artifact_dir=args.codec_artifact_dir,
        codec_artifact_repo=args.codec_artifact_repo,
        revision=getattr(args, "codec_artifact_revision", None),
    )
    if layout is None:
        return None
    args._resolved_codec_artifact = layout
    args.codec_path = str(layout.codec_path)
    return layout


def _preflight_source_payload(source: Any) -> dict[str, Any] | None:
    if source is None:
        return None
    payload = {
        "source": getattr(source, "source", None),
        "source_kind": getattr(source, "source_kind", None),
        "root": str(getattr(source, "root", "")) or None,
    }
    for attr in (
        "manifest_path",
        "weights_path",
        "model_config_path",
        "tokenizer_config_path",
        "conversion_metadata_path",
        "codec_path",
        "metadata_path",
        "checksums_path",
    ):
        value = getattr(source, attr, None)
        if value is not None:
            payload[attr] = str(value)
    return {key: value for key, value in payload.items() if value is not None}


def _preflight_request_count(args: argparse.Namespace, request_overrides: list[dict[str, Any]]) -> int:
    if request_overrides:
        return len(request_overrides)
    if args.text or args.output:
        return 1
    return 0


def load_model_config_json_for_preflight(value: str | Path | None) -> Any:
    """Load ModelConfig without importing the MLX runtime module."""

    from irodori_mlx.config import ModelConfig

    if value is None:
        return ModelConfig()
    raw = str(value).strip()
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Inline model config JSON is invalid.") from exc
    else:
        with Path(value).expanduser().open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Model config JSON must contain an object.")
    return ModelConfig(**payload)


def describe_codec_capabilities_for_preflight(*, args: argparse.Namespace, model_config: Any) -> dict[str, Any]:
    mode = args.codec_runtime_mode
    uses_speaker = bool(model_config.use_speaker_condition)
    needs_reference_encode = uses_speaker
    uses_pytorch_encode = mode in {"persistent", "subprocess", "mlx-decode", "mlx-decode-subprocess"}
    messages: list[str] = []
    report: dict[str, Any] = {
        "runtime_mode": mode,
        "checkpoint_family": model_config.checkpoint_family,
        "codec_path": args.codec_path,
        "mlx_decode_available": None,
        "mlx_encode_available": None,
        "requires_codec_artifact": mode in {"mlx", "mlx-decode", "mlx-decode-subprocess"},
        "requires_pytorch_decode": mode in {"persistent", "subprocess"},
        "requires_pytorch_encode": uses_pytorch_encode and needs_reference_encode,
        "reference_encode_policy": "not-required"
        if not uses_speaker
        else "pytorch-bridge"
        if mode != "mlx"
        else "mlx-artifact",
        "decode_policy": "mlx-artifact" if mode in {"mlx", "mlx-decode", "mlx-decode-subprocess"} else "pytorch-bridge",
    }
    if mode in {"persistent", "subprocess"}:
        messages.append(
            "PyTorch bridge required for DACVAE decode; reference-audio encode is not used."
            if not uses_speaker
            else "PyTorch bridge required for both DACVAE encode and decode."
        )
    elif mode in {"mlx-decode", "mlx-decode-subprocess"}:
        messages.append(
            "MLX codec artifact is used for decode; reference-audio encode is not used."
            if not uses_speaker
            else "MLX codec artifact is used for decode; reference-audio encode falls back to the PyTorch bridge."
        )
    elif mode == "mlx":
        messages.append("MLX codec artifact is used for both encode and decode; artifact must include encode tensors.")
        report["requires_pytorch_encode"] = False
    if report["requires_codec_artifact"] and not args.codec_path:
        messages.append(
            "MLX codec modes require --codec-path, --codec-artifact-dir, or --codec-artifact-repo."
        )
    report["messages"] = tuple(messages)
    return report


def build_preflight_payload(
    *,
    args: argparse.Namespace,
    model_config: Any,
    weights_layout: Any,
    codec_layout: Any,
    request_overrides: list[dict[str, Any]],
) -> dict[str, Any]:
    text_tokenizer_repo = args.text_tokenizer_repo or model_config.text_tokenizer_repo
    caption_tokenizer_repo = args.caption_tokenizer_repo or model_config.caption_tokenizer_repo
    weights_payload = _preflight_source_payload(weights_layout) or {
        "source_kind": "file",
        "weights_path": str(Path(args.weights).expanduser()) if args.weights else None,
        "model_config_json": args.model_config_json,
    }
    codec_payload = _preflight_source_payload(codec_layout) or {
        "source_kind": "file" if args.codec_path else "pytorch-bridge",
        "codec_path": str(Path(args.codec_path).expanduser()) if args.codec_path else None,
        "codec_repo": args.codec_repo,
    }
    payload = {
        "status": "ok",
        "preflight": {
            "generation_will_run": False,
            "request_count": _preflight_request_count(args, request_overrides),
            "checks": [
                "argument validation",
                "weights layout resolution",
                "model config loading",
                "codec artifact resolution",
                "runtime mode summary",
            ],
            "skipped": [
                "tokenizer download/load",
                "MLX RF-DiT weight loading",
                "DACVAE bridge construction",
                "WAV generation",
            ],
        },
        "runtime": {
            "checkpoint_family": model_config.checkpoint_family,
            "checkpoint_family_label": model_config.checkpoint_family_label,
            "checkpoint_capabilities": list(model_config.checkpoint_capabilities),
            "codec_runtime_mode": args.codec_runtime_mode,
            "text_tokenizer_repo": text_tokenizer_repo,
            "caption_tokenizer_repo": caption_tokenizer_repo if model_config.use_caption_condition else None,
            "text_max_length": int(args.text_max_length),
            "caption_max_length": args.caption_max_length,
        },
        "weights": weights_payload,
        "codec": codec_payload,
        "codec_capabilities": describe_codec_capabilities_for_preflight(args=args, model_config=model_config),
        "fallbacks": {
            "weights": "Use --weights with a locally converted .npz if --weights-repo or --weights-dir resolution fails.",
            "codec": "Use --codec-runtime-mode persistent for the upstream PyTorch DACVAE bridge, or --codec-path/--codec-artifact-dir for a local MLX codec artifact.",
            "tokenizer": "If tokenizer loading fails during generation, check network/cache access for the listed tokenizer repo.",
            "upstream": "Install upstream Irodori-TTS or set PYTHONPATH when using PyTorch bridge-backed codec modes.",
        },
    }
    return payload


def print_preflight_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return
    runtime = payload["runtime"]
    print("Preflight OK: generation was not run.")
    print(f"checkpoint: {runtime['checkpoint_family']} ({runtime['checkpoint_family_label']})")
    print(f"codec runtime mode: {runtime['codec_runtime_mode']}")
    print(f"text tokenizer: {runtime['text_tokenizer_repo']}")
    if runtime.get("caption_tokenizer_repo"):
        print(f"caption tokenizer: {runtime['caption_tokenizer_repo']}")
    weights = payload["weights"]
    print(f"weights source: {weights.get('source') or weights.get('weights_path')}")
    codec = payload["codec"]
    print(f"codec source: {codec.get('source') or codec.get('codec_path') or codec.get('codec_repo')}")
    for message in payload["codec_capabilities"].get("messages", ()):
        print(f"- {message}")


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
        "checkpoint_family": getattr(result, "checkpoint_family", None),
        "checkpoint_capabilities": list(getattr(result, "checkpoint_capabilities", ())),
        "codec_backend": getattr(result, "codec_backend", None),
        "codec_encode_backend": getattr(result, "codec_encode_backend", None),
        "codec_decode_backend": getattr(result, "codec_decode_backend", None),
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
            "cleanup_between_requests": bool(args.cleanup_between_requests),
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
            codec_path=args.codec_path,
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


def _request_value(args: argparse.Namespace, overrides: dict[str, Any], key: str) -> Any:
    return overrides[key] if key in overrides else getattr(args, key)


def validate_checkpoint_family_request(
    *, model_config: Any, args: argparse.Namespace, overrides: dict[str, Any], index: int
) -> None:
    family = model_config.checkpoint_family
    family_label = model_config.checkpoint_family_label
    capabilities = ", ".join(model_config.checkpoint_capabilities)
    reference_wav = _request_value(args, overrides, "reference_wav")
    no_reference = bool(_request_value(args, overrides, "no_reference"))
    caption = _request_value(args, overrides, "caption")
    seconds = _request_value(args, overrides, "seconds")

    if caption is not None and str(caption).strip() and not model_config.use_caption_condition:
        raise SystemExit(
            f"error: generation request #{index}: --caption is only supported by VoiceDesign v2 caption checkpoints; "
            f"selected family is {family} ({family_label}; capabilities: {capabilities})"
        )
    if model_config.use_caption_condition:
        if reference_wav:
            raise SystemExit(
                f"error: generation request #{index}: {family_label} is caption/no-reference only; "
                "remove --reference-wav and provide --caption"
            )
        if caption is None or not str(caption).strip():
            raise SystemExit(
                f"error: generation request #{index}: {family_label} requires --caption because speaker reference audio is not supported"
            )
    elif model_config.use_speaker_condition and not no_reference and not reference_wav:
        raise SystemExit(
            f"error: generation request #{index}: {family_label} requires --reference-wav unless --no-reference is true "
            f"(capabilities: {capabilities})"
        )

    if seconds is None and not model_config.use_duration_predictor:
        print(
            f"warning: generation request #{index}: {family_label} has no duration predictor; "
            "omitting --seconds uses the runtime fallback duration. Pass --seconds for explicit control.",
            file=sys.stderr,
        )


def write_metadata_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    preflight = bool(getattr(args, "preflight", False))
    if not preflight:
        _ensure_runtime_imports()
    layout = resolve_weights_layout_source(
        weights_dir=args.weights_dir,
        weights_repo=args.weights_repo,
        revision=args.weights_revision,
    )
    layout_runtime: dict[str, Any] | None = None
    if layout is not None:
        args.weights = str(layout.weights_path)
        model_config = layout.model_config
        layout_runtime = dict(layout.manifest.get("runtime", {}))
    else:
        if preflight and load_model_config_json is None:
            model_config = load_model_config_json_for_preflight(args.model_config_json)
        else:
            if load_model_config_json is None:
                _ensure_runtime_imports()
            model_config = load_model_config_json(args.model_config_json)
    codec_layout = resolve_codec_artifact_args(args)
    if args.requests_json:
        request_overrides = load_generation_requests_json(args.requests_json)
    elif getattr(args, "preflight", False) and not (args.text or args.output):
        request_overrides = []
    else:
        request_overrides = [{}]
    for index, item in enumerate(request_overrides, start=1):
        request_reference = item.get("reference_wav", args.reference_wav)
        request_no_reference = bool(item.get("no_reference", args.no_reference))
        request_caption = item.get("caption", args.caption)
        if layout_runtime is not None:
            if layout_runtime.get("requires_reference_audio") and not request_reference:
                raise SystemExit(
                    f"error: generation request #{index}: selected weights layout requires reference_wav"
                )
            if not layout_runtime.get("supports_no_reference", False) and request_no_reference:
                raise SystemExit(
                    f"error: generation request #{index}: selected weights layout does not support no_reference"
                )
            if (
                "supports_caption" in layout_runtime
                and not layout_runtime.get("supports_caption", False)
                and request_caption is not None
                and str(request_caption).strip()
            ):
                raise SystemExit(
                    f"error: generation request #{index}: selected weights layout does not support caption conditioning"
                )
        validate_checkpoint_family_request(model_config=model_config, args=args, overrides=item, index=index)

    if preflight:
        payload = build_preflight_payload(
            args=args,
            model_config=model_config,
            weights_layout=layout,
            codec_layout=codec_layout,
            request_overrides=request_overrides,
        )
        if args.metadata_json:
            write_metadata_json(args.metadata_json, payload)
        print_preflight_payload(payload, json_output=bool(args.json_output))
        return 0

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
        if args.cleanup_between_requests:
            assert release_mlx_runtime_memory is not None
            release_mlx_runtime_memory()

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
