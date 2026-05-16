from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import CHECKPOINT_FAMILY_BASE_V2, CHECKPOINT_FAMILY_VOICEDESIGN_V2, ModelConfig
from .hosted_weights import MANIFEST_NAME, validate_weights_layout


class MlxAudioAdapterError(ValueError):
    """Raised when a mlx-audio artifact cannot be adapted safely."""


@dataclass(frozen=True)
class MlxAudioAdapterResult:
    output_dir: Path
    weights_path: Path
    model_config_path: Path
    manifest_path: Path
    checkpoint_family: str


WeightConverter = Callable[[Path, Path, Mapping[str, Any]], Mapping[str, Any]]

_MODEL_CONFIG_FIELDS = {field.name for field in fields(ModelConfig)}
_TENSOR_PREFIXES = (
    "model.",
    "dit.",
    "rf_dit.",
    "text_to_latent.",
)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MlxAudioAdapterError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MlxAudioAdapterError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MlxAudioAdapterError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_quantized_config(config: Mapping[str, Any]) -> None:
    quantization = config.get("quantization")
    dit = config.get("dit")
    dit_quantization = dit.get("quantization") if isinstance(dit, Mapping) else None
    if quantization or dit_quantization:
        bits = None
        if isinstance(quantization, Mapping):
            bits = quantization.get("bits")
        if bits is None and isinstance(dit_quantization, Mapping):
            bits = dit_quantization.get("bits")
        suffix = f" ({bits}-bit)" if bits is not None else ""
        raise MlxAudioAdapterError(
            "quantized mlx-audio Irodori artifacts are not supported by this adapter"
            f"{suffix}; use an unquantized fp16/f32 mlx-community Irodori layout until quantized MLX runtime "
            "support is designed."
        )


def translate_mlx_audio_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate mlx-audio's nested config.json.dit payload into ModelConfig fields."""

    if config.get("model_type") not in (None, "irodori_tts"):
        raise MlxAudioAdapterError(f"unsupported mlx-audio model_type: {config.get('model_type')!r}")
    _reject_quantized_config(config)
    dit = config.get("dit")
    if not isinstance(dit, Mapping):
        raise MlxAudioAdapterError("mlx-audio config.json must include a dit object")
    payload = {key: value for key, value in dit.items() if key in _MODEL_CONFIG_FIELDS}
    try:
        model_config = ModelConfig(**payload)
    except (TypeError, ValueError) as exc:
        raise MlxAudioAdapterError(f"mlx-audio config.json.dit is not supported by ModelConfig: {exc}") from exc
    if model_config.checkpoint_family not in (CHECKPOINT_FAMILY_BASE_V2, CHECKPOINT_FAMILY_VOICEDESIGN_V2):
        raise MlxAudioAdapterError(
            "mlx-audio adapter currently supports unquantized v2/base and VoiceDesign layouts only"
        )
    return asdict(model_config)


def remap_mlx_audio_tensor_name(name: str) -> str:
    """Normalize common mlx-audio RF-DiT key prefixes to this repo's converter key space."""

    for prefix in _TENSOR_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _default_weight_converter(source_safetensors: Path, output_npz: Path, model_config: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        from scripts import convert_weights
    except ImportError as exc:  # pragma: no cover - packaging edge case.
        raise MlxAudioAdapterError("scripts.convert_weights is required for mlx-audio weight adaptation") from exc

    try:
        _, records = convert_weights.load_checkpoint(source_safetensors, load_arrays=True)
    except convert_weights.ConversionError as exc:
        raise MlxAudioAdapterError(str(exc)) from exc
    remapped = {}
    for key, record in records.items():
        mapped = remap_mlx_audio_tensor_name(key)
        if mapped in remapped:
            raise MlxAudioAdapterError(f"multiple mlx-audio tensors map to {mapped!r}")
        remapped[mapped] = convert_weights.TensorRecord(
            name=mapped,
            shape=record.shape,
            dtype=record.dtype,
            array=record.array,
        )
    validation = convert_weights.validate_records(remapped, dict(model_config))
    if not validation["ok"]:
        raise MlxAudioAdapterError(convert_weights.validation_error_message(validation))
    family = validation["checkpoint_family"]
    if family is None:
        raise MlxAudioAdapterError("checkpoint family was not resolved after validation")
    try:
        arrays = convert_weights.records_to_arrays(remapped, checkpoint_family=family)
        convert_weights.write_npz_atomic(output_npz, arrays)
    except convert_weights.ConversionError as exc:
        raise MlxAudioAdapterError(str(exc)) from exc
    return validation


def _build_tokenizer_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    caption = bool(model_config.get("use_caption_condition"))
    return {
        "schema_version": 1,
        "text_tokenizer": {
            "source": "mlx-audio config.json.dit.text_tokenizer_repo",
            "repo": model_config.get("text_tokenizer_repo"),
            "normalization_contract": "docs/text_preprocessing.md",
            "padding": "right",
            "truncation": "family-defined",
        },
        "caption_tokenizer": {
            "source": "mlx-audio config.json.dit.caption_tokenizer_repo",
            "repo": model_config.get("caption_tokenizer_repo") or model_config.get("text_tokenizer_repo"),
            "normalization_contract": "docs/text_preprocessing.md",
            "padding": "right",
            "truncation": "family-defined",
        }
        if caption
        else None,
    }


def adapt_mlx_audio_layout(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    source_repo: str | None = None,
    source_revision: str | None = None,
    weight_converter: WeightConverter | None = None,
) -> MlxAudioAdapterResult:
    """Convert an unquantized mlx-audio Irodori directory/snapshot into the hosted weights layout."""

    source = Path(source_dir).expanduser()
    output = Path(output_dir).expanduser()
    config = _read_json_object(source / "config.json", label="mlx-audio config.json")
    model_config = translate_mlx_audio_config(config)
    source_weights = source / "model.safetensors"
    if not source_weights.is_file():
        raise MlxAudioAdapterError(f"mlx-audio model.safetensors is missing: {source_weights}")
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "weights.npz"
    converter = weight_converter or _default_weight_converter
    validation = converter(source_weights, weights_path, model_config)
    if not weights_path.is_file():
        raise MlxAudioAdapterError(f"adapter converter did not write {weights_path}")

    family = ModelConfig(**model_config).checkpoint_family
    if validation.get("checkpoint_family") and validation["checkpoint_family"] != family:
        raise MlxAudioAdapterError(
            f"converted tensor family {validation['checkpoint_family']!r} does not match config family {family!r}"
        )
    model_config_path = output / "model_config.json"
    tokenizer_config_path = output / "tokenizer_config.json"
    conversion_metadata_path = output / "conversion_metadata.json"
    manifest_path = output / MANIFEST_NAME
    checksums_path = output / "checksums.sha256"

    _write_json(model_config_path, model_config)
    _write_json(tokenizer_config_path, _build_tokenizer_config(model_config))
    _write_json(
        conversion_metadata_path,
        {
            "schema_version": 1,
            "converter": {
                "repository": "https://github.com/t0yohei/Irodori-TTS-MLX",
                "name": "mlx-audio-adapter",
            },
            "upstream": {
                "checkpoint_repo": source_repo or config.get("_name_or_path") or str(source),
                "checkpoint_revision": source_revision,
                "artifact_layout": "mlx-audio",
                "config_file": "config.json",
                "weights_file": "model.safetensors",
                "dacvae_present": (source / "dacvae").is_dir(),
            },
            "detected_family": family,
            "mlx_audio": {
                "model_type": config.get("model_type"),
                "sampler": config.get("sampler"),
                "dacvae_repo": config.get("dacvae_repo"),
            },
        },
    )
    caption = family == CHECKPOINT_FAMILY_VOICEDESIGN_V2
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "format": "irodori-tts-mlx-weights",
            "format_version": "0.2",
            "family": family,
            "upstream_checkpoint": source_repo or config.get("_name_or_path") or str(source),
            "files": {
                "weights": "weights.npz",
                "model_config": "model_config.json",
                "tokenizer_config": "tokenizer_config.json",
                "conversion_metadata": "conversion_metadata.json",
                "checksums": "checksums.sha256",
            },
            "runtime": {
                "minimum_irodori_tts_mlx_version": "0.2.0",
                "requires_upstream_dacvae_bridge": True,
                "requires_reference_audio": not caption,
                "supports_no_reference": caption,
                "supports_caption": caption,
                "supports_predicted_duration": False,
            },
            "license_review": {"status": "pending", "review_reference": "local-mlx-audio-adapter"},
        },
    )
    checksum_files = [
        "weights.npz",
        "model_config.json",
        "tokenizer_config.json",
        "conversion_metadata.json",
        MANIFEST_NAME,
    ]
    checksums_path.write_text(
        "".join(f"{_sha256(output / filename)}  {filename}\n" for filename in checksum_files),
        encoding="utf-8",
    )
    validate_weights_layout(output)
    return MlxAudioAdapterResult(
        output_dir=output,
        weights_path=weights_path,
        model_config_path=model_config_path,
        manifest_path=manifest_path,
        checkpoint_family=family,
    )
