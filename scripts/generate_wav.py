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

from irodori_mlx.runtime import (  # noqa: E402
    DACVAEBridgeConfig,
    GenerationRequest,
    MLXDACVAERuntime,
    MLXRuntimeConfig,
    iter_messages,
    load_model_config_json,
)


CONFIG_KEYS = {
    "weights",
    "output",
    "output_wav",
    "text",
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
}

REQUIRED_STRING_KEYS = {"weights", "output", "text"}
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
    "seconds",
    "cfg_scale_text",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_min_t",
    "cfg_max_t",
    "max_reference_seconds",
}
CHOICE_KEYS = {
    "codec_runtime_mode": {"persistent", "subprocess"},
    "cfg_guidance_mode": {"independent", "joint", "reduced"},
}


def _load_json_object(value: str | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    raw = str(value).strip()
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Inline {label} is invalid JSON.") from exc
        source = f"inline {label}"
    else:
        source = str(Path(value).expanduser())
        with Path(value).expanduser().open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return payload


def _validate_generation_config(payload: dict[str, Any]) -> dict[str, Any]:
    for key in REQUIRED_STRING_KEYS:
        if key in payload:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"generation config field '{key}' must be a non-empty string")

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


def _default(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config.get(key, fallback)


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
    parser.add_argument("--weights", required="weights" not in config, default=config.get("weights"), help="Converted MLX .npz RF-DiT weights.")
    parser.add_argument("--output", "--output-wav", dest="output", required="output" not in config, default=config.get("output"), help="Output WAV path.")
    parser.add_argument("--text", required="text" not in config, default=config.get("text"), help="Text prompt to synthesize.")
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
    parser.add_argument("--seconds", type=float, default=float(_default(config, "seconds", 5.0)), help="Target output duration in seconds (default: 5.0).")
    parser.add_argument("--num-steps", type=int, default=int(_default(config, "num_steps", 40)), help="RF sampling steps (default: 40).")
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
    if args.reference_wav and args.no_reference:
        parser.error("choose either --reference-wav or --no-reference, not both")
    if args.weights is None or not str(args.weights).strip():
        parser.error("--weights must not be empty")
    if args.output is None or not str(args.output).strip():
        parser.error("--output must not be empty")
    if args.text is None or not str(args.text).strip():
        parser.error("--text must not be empty")
    if args.seconds <= 0:
        parser.error("--seconds must be > 0")
    if args.num_steps <= 0:
        parser.error("--num-steps must be > 0")
    if args.text_max_length <= 0:
        parser.error("--text-max-length must be > 0")
    if args.caption_max_length is not None and int(args.caption_max_length) <= 0:
        parser.error("--caption-max-length must be > 0 when provided")
    if args.max_reference_seconds <= 0:
        parser.error("--max-reference-seconds must be > 0")
    return args


def build_result_payload(
    *, result: Any, request: GenerationRequest, runtime: MLXDACVAERuntime, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "result": asdict(result) if hasattr(result, "__dataclass_fields__") else {
            "output_wav": result.output_wav,
            "sample_rate": result.sample_rate,
            "samples": result.samples,
            "latent_steps": result.latent_steps,
            "patched_steps": result.patched_steps,
            "seed": result.seed,
            "timings_ms": result.timings_ms,
            "messages": list(result.messages),
        },
        "request": asdict(request),
        "boundaries": runtime.describe_boundaries(),
        "cli": {
            "config_json": args.config_json,
            "metadata_json": args.metadata_json,
            "json_output": bool(args.json_output),
            "print_boundaries": bool(args.print_boundaries),
        },
    }


def write_metadata_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_config = load_model_config_json(args.model_config_json)
    if model_config.use_speaker_condition and not args.no_reference and not args.reference_wav:
        raise SystemExit(
            "error: speaker-conditioned checkpoints require --reference-wav unless you explicitly pass --no-reference"
        )

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
            runtime_mode=args.codec_runtime_mode,
            enable_watermark=bool(args.enable_watermark),
            normalize_db=None if args.disable_codec_normalize else -16.0,
        ),
    )
    runtime = MLXDACVAERuntime(config=runtime_config)
    if args.print_boundaries:
        print(
            json.dumps(runtime.describe_boundaries(), indent=2, sort_keys=True, default=str),
            flush=True,
            file=sys.stderr if args.json_output else sys.stdout,
        )

    request = GenerationRequest(
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
    result = runtime.generate(request)
    payload = build_result_payload(result=result, request=request, runtime=runtime, args=args)
    if args.metadata_json:
        write_metadata_json(args.metadata_json, payload)
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    for line in iter_messages(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
