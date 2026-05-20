from __future__ import annotations

import gc
import json
import math
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx

from .config import ModelConfig
from .dacvae import (
    EXECUTABLE_DECODER_PREFIX,
    EXECUTABLE_ENCODER_PREFIX,
    load_semantic_dacvae_decoder_artifact,
    load_semantic_dacvae_encoder_artifact,
    semantic_dacvae_decoder_config_from_metadata,
    semantic_dacvae_decoder_expected_shapes,
    semantic_dacvae_encoder_config_from_metadata,
    semantic_dacvae_encoder_expected_shapes,
)
from .duration import (
    build_duration_features,
    estimate_fallback_duration_seconds,
    estimate_voicedesign_duration_seconds,
    predicted_duration_overallocation_warning,
)
from .layers import unpatch_latents
from .model import TextToLatentRFDiT
from .sampling import sample_euler_rf_cfg
from .text_normalization import normalize_text
from .weights import assign_named_weights, load_npz_weights, rf_dit_required_keys


QUICKSTART_TROUBLESHOOTING = "See README.md 'If the quickstart fails' and docs/checkpoint_support.md."


@dataclass(frozen=True)
class DACVAEBridgeConfig:
    """Configuration for the DACVAE encode/decode boundary."""

    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    codec_path: str | None = None
    codec_device: str = "cpu"
    runtime_mode: str = "mlx"
    deterministic_encode: bool = True
    deterministic_decode: bool = True
    enable_watermark: bool = False
    normalize_db: float | None = -16.0


@dataclass(frozen=True)
class MLXRuntimeConfig:
    """Configuration for the end-to-end MLX runtime."""

    model_config: ModelConfig
    weights_path: str
    text_tokenizer_repo: str | None = None
    caption_tokenizer_repo: str | None = None
    max_text_len: int = 256
    max_caption_len: int | None = None
    codec: DACVAEBridgeConfig = DACVAEBridgeConfig()


@dataclass(frozen=True)
class SamplingRequest:
    text: str
    output_wav: str
    ref_wav: str | None = None
    ref_latent: str | None = None
    ref_embed: str | None = None
    no_ref: bool = False
    caption: str | None = None
    seconds: float | None = None
    duration_scale: float = 1.0
    max_auto_seconds: float | None = None
    max_auto_estimate_seconds: float | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    seed: int = 0
    max_ref_seconds: float | None = 30.0
    context_kv_cache: bool = True


@dataclass(frozen=True)
class SamplingResult:
    output_wav: str
    sample_rate: int
    samples: int
    latent_steps: int
    patched_steps: int
    seed: int
    duration_mode: str
    checkpoint_family: str
    checkpoint_capabilities: tuple[str, ...]
    speaker_condition_source: str = "none"
    codec_backend: str = "mlx"
    codec_encode_backend: str = "mlx"
    codec_decode_backend: str = "mlx"
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    requested_seconds: float | None = None
    resolved_seconds: float | None = None
    timings_ms: dict[str, float] | None = None
    messages: tuple[str, ...] = ()


def _as_numpy(value: mx.array):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is normally present with MLX.
        raise RuntimeError("numpy is required for MLX/PyTorch tensor conversion.") from exc
    return np.array(value)


def release_mlx_runtime_memory() -> None:
    """End pending MLX work and release reusable cache memory when supported."""

    try:
        if hasattr(mx, "synchronize"):
            mx.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass

def load_model_config_json(value: str | Path | None) -> ModelConfig:
    """Load `ModelConfig` from a JSON file path or an inline JSON object string."""

    if value is None:
        return ModelConfig()
    raw = str(value).strip()
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Inline model config JSON is invalid.") from exc
        source = "inline JSON"
    else:
        with Path(value).expanduser().open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        source = str(value)
    if not isinstance(payload, dict):
        raise ValueError(f"Model config JSON must contain an object: {source}")
    try:
        return ModelConfig(**payload)
    except TypeError as exc:
        raise ValueError(
            f"Model config JSON contains unsupported keys for irodori_mlx.config.ModelConfig: {source}. "
            "Use the config object emitted by scripts/inspect_checkpoint.py and keep only ModelConfig fields. "
            f"{QUICKSTART_TROUBLESHOOTING}"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"Model config JSON is not supported by the MLX runtime: {source}: {exc}. "
            "Confirm the checkpoint family and config match the converted weights. "
            f"{QUICKSTART_TROUBLESHOOTING}"
        ) from exc


def patch_latents_drop_tail(latents: mx.array, patch_size: int) -> mx.array:
    """Patch latent sequences and drop an incomplete tail like upstream DACVAE helpers."""

    if int(patch_size) <= 1:
        return latents
    if len(latents.shape) != 3:
        raise ValueError(f"Expected latents with shape (B,T,D), got {latents.shape}")
    bsz, seq_len, dim = latents.shape
    usable = (int(seq_len) // int(patch_size)) * int(patch_size)
    if usable <= 0:
        raise ValueError(f"Latent sequence too short for patch_size={patch_size}: seq_len={seq_len}")
    return latents[:, :usable].reshape(bsz, usable // int(patch_size), dim * int(patch_size))


SPEAKER_EMBED_TENSOR_KEYS = (
    "speaker_state",
    "speaker_embedding",
    "speaker_embed",
    "embedding",
    "speaker",
)

REFERENCE_LATENT_TENSOR_KEYS = (
    "reference_latent",
    "ref_latent",
    "latents",
)


def _select_speaker_embedding_tensor(tensors: dict[str, object], path: Path) -> tuple[str, object]:
    for key in SPEAKER_EMBED_TENSOR_KEYS:
        if key in tensors:
            return key, tensors[key]
    if len(tensors) == 1:
        key = next(iter(tensors))
        return key, tensors[key]
    allowed = ", ".join(SPEAKER_EMBED_TENSOR_KEYS)
    present = ", ".join(sorted(tensors)) or "<none>"
    raise ValueError(
        f"Speaker embedding {path} must contain one tensor or one of: {allowed}. "
        f"Found: {present}"
    )


def load_speaker_embedding_safetensors(path: str | Path, *, speaker_dim: int) -> tuple[mx.array, mx.array, dict[str, object]]:
    """Load an upstream Speaker Inversion embedding as a direct MLX speaker-state condition."""

    embed_path = Path(path).expanduser()
    try:
        from safetensors.numpy import load_file as load_safetensors_numpy
    except ImportError as exc:  # pragma: no cover - depends on optional runtime extra.
        raise RuntimeError(
            "safetensors is required for --ref-embed. Install the runtime extra with "
            "python -m pip install -e .[runtime] or install safetensors."
        ) from exc
    try:
        tensors = load_safetensors_numpy(str(embed_path))
    except Exception as exc:
        raise ValueError(f"Could not load speaker embedding safetensors file: {embed_path}: {exc}") from exc
    if not tensors:
        raise ValueError(f"Speaker embedding safetensors file has no tensors: {embed_path}")

    tensor_key, value = _select_speaker_embedding_tensor(tensors, embed_path)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a base dependency.
        raise RuntimeError("numpy is required to load speaker embeddings.") from exc

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if int(arr.shape[0]) != 1:
            raise ValueError(
                f"Speaker embedding batch dimension must be 1 for single-request generation, got shape={arr.shape}"
            )
    else:
        raise ValueError(
            f"Speaker embedding tensor must have shape (D), (S,D), or (1,S,D), got shape={arr.shape}"
        )
    if int(arr.shape[-1]) != int(speaker_dim):
        raise ValueError(
            f"Speaker embedding last dimension must match speaker_dim={speaker_dim}, got shape={arr.shape}"
        )
    if int(arr.shape[1]) <= 0:
        raise ValueError(f"Speaker embedding sequence length must be positive, got shape={arr.shape}")

    speaker_state = mx.array(arr, dtype=mx.float32)
    speaker_mask = mx.ones((1, int(arr.shape[1])), dtype=mx.bool_)
    mx.eval(speaker_state, speaker_mask)
    metadata: dict[str, object] = {
        "path": str(embed_path),
        "tensor_key": tensor_key,
        "shape": [int(dim) for dim in arr.shape],
    }
    return speaker_state, speaker_mask, metadata


def _select_reference_latent_tensor(tensors: dict[str, object], path: Path) -> tuple[str, object]:
    for key in REFERENCE_LATENT_TENSOR_KEYS:
        if key in tensors:
            return key, tensors[key]
    allowed = ", ".join(REFERENCE_LATENT_TENSOR_KEYS)
    present = ", ".join(sorted(tensors)) or "<none>"
    raise ValueError(
        f"Reference latent artifact {path} must contain one of: {allowed}. "
        f"Found: {present}"
    )


def load_reference_latent_npz(path: str | Path, *, latent_dim: int) -> tuple[mx.array, mx.array, dict[str, object]]:
    """Load an MLX cached reference latent artifact.

    The public artifact boundary is a NumPy .npz file containing DACVAE encoder
    latents before RF-DiT patching. Accepted tensor keys are intentionally small
    and explicit so the runtime does not grow arbitrary upstream .pt compatibility.
    """

    latent_path = Path(path).expanduser()
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a base dependency.
        raise RuntimeError("numpy is required to load reference latent artifacts.") from exc
    try:
        with np.load(latent_path, allow_pickle=False) as archive:
            tensors = {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise ValueError(f"Could not load reference latent .npz file: {latent_path}: {exc}") from exc
    if not tensors:
        raise ValueError(f"Reference latent .npz file has no arrays: {latent_path}")

    tensor_key, value = _select_reference_latent_tensor(tensors, latent_path)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if int(arr.shape[0]) != 1:
            raise ValueError(
                f"Reference latent batch dimension must be 1 for single-request generation, got shape={arr.shape}"
            )
    else:
        raise ValueError(f"Reference latent tensor must have shape (T,D) or (1,T,D), got shape={arr.shape}")
    if int(arr.shape[1]) <= 0:
        raise ValueError(f"Reference latent sequence length must be positive, got shape={arr.shape}")
    if int(arr.shape[-1]) != int(latent_dim):
        raise ValueError(f"Reference latent last dimension must match latent_dim={latent_dim}, got shape={arr.shape}")

    ref_latent = mx.array(arr, dtype=mx.float32)
    ref_mask = mx.ones((1, int(arr.shape[1])), dtype=mx.bool_)
    mx.eval(ref_latent, ref_mask)
    metadata: dict[str, object] = {
        "path": str(latent_path),
        "tensor_key": tensor_key,
        "shape": [int(dim) for dim in arr.shape],
    }
    return ref_latent, ref_mask, metadata


def load_mlx_model(config: ModelConfig, weights_path: str | Path) -> TextToLatentRFDiT:
    """Load a converted MLX RF-DiT model from an `.npz` archive."""

    model = TextToLatentRFDiT(config)
    try:
        weights = load_npz_weights(weights_path)
        assign_named_weights(
            model,
            weights,
            required=rf_dit_required_keys(config),
            strict=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load converted MLX weights from {Path(weights_path).expanduser()}: {exc}. "
            "Most first-run failures here are a weights/config family mismatch. Re-run the Quickstart: "
            "inspect the checkpoint, derive --model-config-json from metadata.config_json, then convert the "
            f"same checkpoint to .npz. {QUICKSTART_TROUBLESHOOTING}"
        ) from exc
    mx.eval(model.parameters())
    return model


class PretrainedTextTokenizer:
    """Small runtime tokenizer wrapper matching upstream right-padding semantics."""

    def __init__(self, tokenizer, *, add_bos: bool = True) -> None:
        self.tokenizer = tokenizer
        self.add_bos = bool(add_bos)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Tokenizer has no pad_token_id and no eos_token fallback.")
        if self.add_bos and self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no bos_token_id but BOS prepend was requested.")

    @classmethod
    def from_pretrained(cls, repo_id: str, *, add_bos: bool = True) -> "PretrainedTextTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError(
                "transformers is required for text tokenization. Install the runtime extra with "
                "`python -m pip install -e \".[runtime]\"` or install transformers and sentencepiece. "
                f"{QUICKSTART_TROUBLESHOOTING}"
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=False)
            return cls(tokenizer, add_bos=add_bos)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer {repo_id!r}: {exc}. "
                "Check network/cache access and make sure --text-tokenizer-repo / --caption-tokenizer-repo "
                "matches the checkpoint config. For VoiceDesign checkpoints, the caption tokenizer must also "
                f"be available. {QUICKSTART_TROUBLESHOOTING}"
            ) from exc

    def encode(self, text: str, *, max_length: int) -> tuple[mx.array, mx.array]:
        token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if self.add_bos:
            token_ids.insert(0, int(self.tokenizer.bos_token_id))
        n = min(int(max_length), len(token_ids))
        pad_id = int(self.tokenizer.pad_token_id)
        ids = [pad_id] * int(max_length)
        mask = [False] * int(max_length)
        if n > 0:
            ids[:n] = token_ids[:n]
            mask[:n] = [True] * n
        return mx.array([ids], dtype=mx.int32), mx.array([mask], dtype=mx.bool_)

def _load_npz_scalar_string(archive, name: str) -> str | None:
    if name not in archive.files:
        return None
    value = archive[name]
    if getattr(value, "shape", ()) == ():
        return str(value.item())
    if getattr(value, "shape", ()) == (1,):
        return str(value[0])
    raise ValueError(f"codec metadata field {name!r} must be a scalar string")


def _load_npz_scalar_int(archive, name: str) -> int:
    value = archive[name]
    if getattr(value, "shape", ()) == ():
        return int(value.item())
    if getattr(value, "shape", ()) == (1,):
        return int(value[0])
    raise ValueError(f"codec metadata field {name!r} must be a scalar integer")


def _npz_array_shape(archive, name: str) -> tuple[int, ...]:
    import numpy as np

    with archive.zip.open(name + ".npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(member)
        elif version == (3, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_3_0(member)
        else:
            raise ValueError(f"Unsupported .npy header version for {name!r}: {version}")
    return tuple(int(dim) for dim in shape)


def inspect_mlx_codec_artifact(path: str | Path) -> dict[str, object]:
    """Inspect the local MLX DACVAE codec artifact without importing PyTorch."""

    codec_path = Path(path).expanduser()
    try:
        import numpy as np

        with np.load(codec_path, allow_pickle=False) as archive:
            metadata_json = _load_npz_scalar_string(archive, "metadata_json")
            metadata = json.loads(metadata_json) if metadata_json else {}
            sample_rate = int(metadata.get("sample_rate", _load_npz_scalar_int(archive, "sample_rate")))
            hop_length = int(metadata.get("hop_length", _load_npz_scalar_int(archive, "hop_length")))
            latent_dim = int(metadata.get("latent_dim", _load_npz_scalar_int(archive, "latent_dim")))
            files = set(archive.files)
            has_linear_decode = {"decode_basis", "decode_bias"}.issubset(files)
            has_encode = {"encode_basis", "encode_bias"}.issubset(files)
            real_decode_tensors = sorted(name for name in files if name.startswith("dacvae_decoder/"))
            executable_decode_tensors = sorted(name for name in files if name.startswith(EXECUTABLE_DECODER_PREFIX))
            executable_decode_keys = {name[len(EXECUTABLE_DECODER_PREFIX) :] for name in executable_decode_tensors}
            executable_encode_tensors = sorted(name for name in files if name.startswith(EXECUTABLE_ENCODER_PREFIX))
            executable_encode_keys = {name[len(EXECUTABLE_ENCODER_PREFIX) :] for name in executable_encode_tensors}
            semantic_decoder_config = semantic_dacvae_decoder_config_from_metadata(metadata)
            expected_executable_shapes = semantic_dacvae_decoder_expected_shapes(semantic_decoder_config)
            required_executable_keys = set(expected_executable_shapes)
            executable_shape_mismatches = {}
            for key, expected in expected_executable_shapes.items():
                if key not in executable_decode_keys:
                    continue
                actual = _npz_array_shape(archive, EXECUTABLE_DECODER_PREFIX + key)
                if actual != expected:
                    executable_shape_mismatches[key] = {"expected": expected, "actual": actual}
            has_executable_decode = (
                required_executable_keys.issubset(executable_decode_keys)
                and not executable_shape_mismatches
                and int(latent_dim) == int(semantic_decoder_config.codebook_dim)
            )
            semantic_encoder_config = semantic_dacvae_encoder_config_from_metadata(metadata)
            expected_executable_encode_shapes = semantic_dacvae_encoder_expected_shapes(semantic_encoder_config)
            required_executable_encode_keys = set(expected_executable_encode_shapes)
            executable_encode_shape_mismatches = {}
            for key, expected in expected_executable_encode_shapes.items():
                if key not in executable_encode_keys:
                    continue
                actual = _npz_array_shape(archive, EXECUTABLE_ENCODER_PREFIX + key)
                if actual != expected:
                    executable_encode_shape_mismatches[key] = {"expected": expected, "actual": actual}
            has_executable_encode = (
                required_executable_encode_keys.issubset(executable_encode_keys)
                and not executable_encode_shape_mismatches
                and int(latent_dim) == int(semantic_encoder_config.codebook_dim)
            )
            has_decode = has_linear_decode or has_executable_decode
            semantic_encoder_tensors = sorted(name for name in files if name.startswith("dacvae_encoder/encoder."))
            semantic_in_proj_tensors = sorted(
                name
                for name in files
                if name.startswith("dacvae_encoder/quantizer.in_proj.")
                or name.startswith("dacvae_quantizer/quantizer.in_proj.")
            )
            raw_artifact_kind = str(metadata.get("artifact_kind") or "linear-fixture")
            has_semantic_encoder_manifest = "semantic_encoder_manifest_json" in files
            has_real_semantic_encode = has_executable_encode or (bool(semantic_encoder_tensors) and bool(semantic_in_proj_tensors))
            is_semantic_dacvae = (
                raw_artifact_kind == "semantic-dacvae"
                and has_semantic_encoder_manifest
                and has_real_semantic_encode
            )
            artifact_kind = "semantic-dacvae" if is_semantic_dacvae else raw_artifact_kind
            if raw_artifact_kind == "semantic-dacvae" and not is_semantic_dacvae:
                artifact_kind = "linear-fixture" if has_decode or has_encode else "unverified-semantic-dacvae"
            has_real_decode = artifact_kind == "real_semantic_dacvae_decoder" and bool(real_decode_tensors)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"MLX DACVAE codec artifact was not found: {codec_path}") from exc
    except KeyError as exc:
        raise ValueError(f"MLX DACVAE codec artifact {codec_path} is missing metadata field {exc}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"MLX DACVAE codec artifact {codec_path} has invalid metadata_json: {exc}") from exc

    return {
        "codec_path": str(codec_path),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "latent_dim": latent_dim,
        "has_mlx_decode": has_decode,
        "has_linear_mlx_decode": has_linear_decode,
        "has_executable_mlx_decode": has_executable_decode,
        "has_mlx_encode": has_encode or has_executable_encode,
        "has_linear_mlx_encode": has_encode,
        "has_executable_mlx_encode": has_executable_encode,
        "artifact_kind": artifact_kind,
        "is_semantic_dacvae": is_semantic_dacvae,
        "has_semantic_encoder_manifest": has_semantic_encoder_manifest,
        "has_real_semantic_encode": has_real_semantic_encode,
        "semantic_encoder_tensor_count": len(semantic_encoder_tensors),
        "semantic_in_proj_tensor_count": len(semantic_in_proj_tensors),
        "has_real_dacvae_decode": has_real_decode,
        "real_dacvae_decode_tensor_count": len(real_decode_tensors),
        "executable_dacvae_decode_tensor_count": len(executable_decode_tensors),
        "missing_executable_dacvae_decode_tensor_count": len(required_executable_keys - executable_decode_keys),
        "mismatched_executable_dacvae_decode_tensor_count": len(executable_shape_mismatches),
        "executable_dacvae_encode_tensor_count": len(executable_encode_tensors),
        "missing_executable_dacvae_encode_tensor_count": len(required_executable_encode_keys - executable_encode_keys),
        "mismatched_executable_dacvae_encode_tensor_count": len(executable_encode_shape_mismatches),
        "metadata": metadata,
    }


def describe_codec_capabilities(
    codec: DACVAEBridgeConfig,
    *,
    model_config: ModelConfig | None = None,
) -> dict[str, object]:
    """Return user-facing codec availability and fallback policy for a runtime config."""

    mode = codec.runtime_mode
    family = model_config.checkpoint_family if model_config is not None else None
    uses_speaker = bool(model_config.use_speaker_condition) if model_config is not None else None
    report: dict[str, object] = {
        "runtime_mode": mode,
        "checkpoint_family": family,
        "codec_path": codec.codec_path,
        "mlx_decode_available": False,
        "mlx_encode_available": False,
        "requires_codec_artifact": mode == "mlx",
        "requires_pytorch_decode": False,
        "requires_pytorch_encode": False,
        "reference_encode_policy": "not-required"
        if uses_speaker is False
        else "mlx-artifact",
        "decode_policy": "mlx-artifact",
        "messages": [],
    }
    messages: list[str] = []
    if mode == "mlx":
        messages.append("MLX codec artifact is used for both encode and decode; artifact must include encode tensors.")
    else:
        messages.append(f"Unsupported codec runtime mode: {mode!r}.")

    if report["requires_codec_artifact"]:
        if codec.codec_path:
            try:
                artifact = inspect_mlx_codec_artifact(codec.codec_path)
            except (OSError, ValueError) as exc:
                report["artifact_error"] = str(exc)
                messages.append(str(exc))
            else:
                report["artifact"] = artifact
                report["mlx_decode_available"] = bool(artifact["has_executable_mlx_decode"])
                report["mlx_encode_available"] = bool(artifact["has_executable_mlx_encode"])
                if artifact.get("has_executable_mlx_encode"):
                    messages.append(
                        "Codec artifact contains executable Semantic-DACVAE encoder tensors for MLX reference-audio encode; "
                        "encode parity remains gated by local validation."
                    )
                if artifact.get("has_executable_mlx_decode"):
                    messages.append(
                        "Codec artifact contains executable Semantic-DACVAE decoder tensors for MLX decode; "
                        "acoustic parity remains gated by local validation."
                    )
                elif artifact.get("has_real_dacvae_decode") and not artifact["has_mlx_decode"]:
                    messages.append(
                        "Codec artifact contains real Semantic-DACVAE decoder tensors, but this runtime "
                        "does not yet implement the MLX DACVAE convolutional decoder executor."
                    )
                elif not artifact["has_mlx_decode"]:
                    messages.append("Codec artifact is missing decode_basis/decode_bias, so MLX decode is unavailable.")
                if mode == "mlx" and not artifact["has_executable_mlx_decode"]:
                    messages.append("codec_runtime_mode='mlx' requires executable Semantic-DACVAE decoder tensors.")
                if mode == "mlx" and not artifact["has_executable_mlx_encode"]:
                    messages.append("codec_runtime_mode='mlx' requires executable Semantic-DACVAE encoder tensors.")
        else:
            messages.append(
                "MLX codec modes require --codec-path pointing to a local DACVAE codec .npz; "
                "or use --codec-artifact-repo / --codec-artifact-dir for a hosted/local codec layout."
            )
    report["messages"] = tuple(messages)
    return report


class MLXDACVAEBridge:
    """MLX-native DACVAE artifact contract for encode/decode experiments.

    The artifact is intentionally explicit: it must provide small projection
    tensors and metadata instead of importing the upstream PyTorch codec.
    Real Semantic-DACVAE parity depends on converted codec artifacts produced
    outside this lightweight test fixture contract.
    """

    artifact_version = 1

    def __init__(self, *, config: DACVAEBridgeConfig, require_encode: bool = True) -> None:
        if not config.codec_path:
            raise ValueError(
                f"codec_runtime_mode={config.runtime_mode!r} requires --codec-path pointing to a local MLX "
                "DACVAE codec artifact .npz, or use --codec-artifact-repo / --codec-artifact-dir for a "
                "hosted/local codec layout."
            )
        self.config = config
        self.codec_path = Path(config.codec_path).expanduser()
        self.semantic_decoder = None
        self.semantic_encoder = None
        self.last_decode_timings_ms: dict[str, float] = {}
        try:
            import numpy as np

            with np.load(self.codec_path, allow_pickle=False) as archive:
                metadata_json = _load_npz_scalar_string(archive, "metadata_json")
                metadata = json.loads(metadata_json) if metadata_json else {}
                self.sample_rate = int(metadata.get("sample_rate", _load_npz_scalar_int(archive, "sample_rate")))
                self.hop_length = int(metadata.get("hop_length", _load_npz_scalar_int(archive, "hop_length")))
                self.latent_dim = int(metadata.get("latent_dim", _load_npz_scalar_int(archive, "latent_dim")))
                has_linear_decode = {"decode_basis", "decode_bias"}.issubset(archive.files)
                has_linear_encode = {"encode_basis", "encode_bias"}.issubset(archive.files)
                executable_names = [name for name in archive.files if name.startswith(EXECUTABLE_DECODER_PREFIX)]
                executable_keys = {name[len(EXECUTABLE_DECODER_PREFIX) :] for name in executable_names}
                executable_encode_names = [name for name in archive.files if name.startswith(EXECUTABLE_ENCODER_PREFIX)]
                executable_encode_keys = {name[len(EXECUTABLE_ENCODER_PREFIX) :] for name in executable_encode_names}
                semantic_decoder_config = semantic_dacvae_decoder_config_from_metadata(metadata)
                expected_shapes = semantic_dacvae_decoder_expected_shapes(semantic_decoder_config)
                has_complete_executable_decode = (
                    bool(executable_names)
                    and set(expected_shapes).issubset(executable_keys)
                    and all(
                        _npz_array_shape(archive, EXECUTABLE_DECODER_PREFIX + key) == expected
                        for key, expected in expected_shapes.items()
                    )
                )
                if has_complete_executable_decode and int(self.latent_dim) != int(semantic_decoder_config.codebook_dim):
                    raise ValueError(
                        "Executable Semantic-DACVAE decoder artifact latent_dim must match "
                        "semantic_dacvae_decoder_config.codebook_dim: "
                        f"latent_dim={self.latent_dim}, codebook_dim={semantic_decoder_config.codebook_dim}."
                    )
                has_executable_decode = has_complete_executable_decode
                semantic_encoder_config = semantic_dacvae_encoder_config_from_metadata(metadata)
                expected_encode_shapes = semantic_dacvae_encoder_expected_shapes(semantic_encoder_config)
                has_complete_executable_encode = (
                    bool(executable_encode_names)
                    and set(expected_encode_shapes).issubset(executable_encode_keys)
                    and all(
                        _npz_array_shape(archive, EXECUTABLE_ENCODER_PREFIX + key) == expected
                        for key, expected in expected_encode_shapes.items()
                    )
                )
                if (
                    has_complete_executable_encode
                    and require_encode
                    and int(self.latent_dim) != int(semantic_encoder_config.codebook_dim)
                ):
                    raise ValueError(
                        "Executable Semantic-DACVAE encoder artifact latent_dim must match "
                        "semantic_dacvae_encoder_config.codebook_dim: "
                        f"latent_dim={self.latent_dim}, codebook_dim={semantic_encoder_config.codebook_dim}."
                    )
                if require_encode and not (has_executable_decode and has_complete_executable_encode):
                    missing = []
                    if not has_executable_decode:
                        missing.append("decoder")
                    if not has_complete_executable_encode:
                        missing.append("encoder")
                    raise ValueError(
                        "codec_runtime_mode='mlx' requires executable Semantic-DACVAE "
                        + " and ".join(missing)
                        + " tensors in the codec artifact."
                    )
                if has_executable_decode:
                    self.semantic_decoder = load_semantic_dacvae_decoder_artifact(self.codec_path)
                    self.decode_basis = None
                    self.decode_bias = None
                else:
                    if metadata.get("artifact_kind") == "real_semantic_dacvae_decoder" and not has_linear_decode:
                        raise NotImplementedError(
                            "This artifact contains converted real Semantic-DACVAE decoder tensors, but no "
                            "executable MLX decoder tensor layout. Re-run scripts/convert_dacvae_decoder.py "
                            "from this version."
                        )
                    self.decode_basis = mx.array(archive["decode_basis"].astype("float32", copy=False))
                    self.decode_bias = mx.array(archive["decode_bias"].astype("float32", copy=False))
                if has_complete_executable_encode and require_encode:
                    self.semantic_encoder = load_semantic_dacvae_encoder_artifact(self.codec_path)
                    self.encode_basis = None
                    self.encode_bias = None
                else:
                    self.encode_basis = (
                        mx.array(archive["encode_basis"].astype("float32", copy=False))
                        if has_linear_encode
                        else None
                    )
                    self.encode_bias = (
                        mx.array(archive["encode_bias"].astype("float32", copy=False))
                        if has_linear_encode
                        else None
                    )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Converted MLX DACVAE codec artifact was not found: {self.codec_path}") from exc
        except KeyError as exc:
            raise ValueError(
                f"Converted MLX DACVAE codec artifact {self.codec_path} is missing required array {exc}."
            ) from exc

        if self.semantic_decoder is None:
            if tuple(self.decode_basis.shape) != (self.latent_dim, self.hop_length):
                raise ValueError(
                    f"decode_basis must have shape ({self.latent_dim}, {self.hop_length}), got {self.decode_basis.shape}"
                )
            if tuple(self.decode_bias.shape) != (self.hop_length,):
                raise ValueError(f"decode_bias must have shape ({self.hop_length},), got {self.decode_bias.shape}")
        if require_encode and self.semantic_encoder is None and (self.encode_basis is None or self.encode_bias is None):
            raise ValueError(
                f"Converted MLX DACVAE codec artifact {self.codec_path} is missing executable encoder tensors or encode_basis/encode_bias; "
                "codec_runtime_mode='mlx' requires an encode-capable codec artifact."
            )
        if self.encode_basis is not None and tuple(self.encode_basis.shape) != (self.hop_length, self.latent_dim):
            raise ValueError(
                f"encode_basis must have shape ({self.hop_length}, {self.latent_dim}), got {self.encode_basis.shape}"
            )
        if self.encode_bias is not None and tuple(self.encode_bias.shape) != (self.latent_dim,):
            raise ValueError(f"encode_bias must have shape ({self.latent_dim},), got {self.encode_bias.shape}")

    def encode_reference(
        self,
        path: str | Path,
        *,
        max_seconds: float | None,
        normalize_db: float | None,
        ensure_max: bool,
    ) -> mx.array:
        if self.semantic_encoder is None and (self.encode_basis is None or self.encode_bias is None):
            raise RuntimeError("This MLX DACVAE artifact is decode-only and cannot encode reference audio.")
        samples, sample_rate = _load_audio_numpy(path)
        if max_seconds is not None and float(max_seconds) > 0:
            samples = samples[: max(1, int(float(max_seconds) * float(sample_rate)))]
        samples = _resample_audio_linear(samples, source_rate=int(sample_rate), target_rate=int(self.sample_rate))
        if normalize_db is not None:
            samples = _normalize_audio_db(samples, target_db=float(normalize_db))
        elif ensure_max:
            samples = _ensure_audio_peak(samples)
        if samples.size == 0:
            raise ValueError("reference audio is empty")
        if self.semantic_encoder is not None:
            padded = _pad_audio_to_hop(samples, self.hop_length)
            latents = self.semantic_encoder(mx.array(padded[None, :, None], dtype=mx.float32))
            mx.eval(latents)
            return latents
        frames = _frame_audio(samples, self.hop_length)
        latents = mx.array(frames, dtype=mx.float32) @ self.encode_basis + self.encode_bias
        mx.eval(latents)
        return latents[None, :, :]

    def decode_to_wav_timed(
        self, latents: mx.array, output_path: str | Path, *, max_samples: int | None = None
    ) -> tuple[Path, dict[str, float]]:
        self.last_decode_timings_ms = {}
        if len(latents.shape) != 3:
            raise ValueError(f"Expected MLX latents with shape (B,T,D), got {latents.shape}")
        if int(latents.shape[0]) != 1:
            raise ValueError(f"MLX DACVAE decode currently supports batch size 1, got {latents.shape[0]}")
        if int(latents.shape[2]) != int(self.latent_dim):
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latents.shape[2]}")
        waveform = None
        frames = None
        samples = None
        output: Path | None = None
        timings_ms: dict[str, float] = {}
        try:
            started = time.perf_counter()
            if self.semantic_decoder is not None:
                waveform = self.semantic_decoder(latents.astype(mx.float32))
                timings_ms["decode_dacvae_model_compute"] = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                mx.eval(waveform)
                timings_ms["decode_dacvae_materialization"] = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                samples = _as_numpy(waveform[0, :, 0]).astype("float32", copy=False)
                timings_ms["decode_dacvae_host_transfer"] = (time.perf_counter() - started) * 1000.0
            else:
                frames = latents[0].astype(mx.float32) @ self.decode_basis + self.decode_bias
                timings_ms["decode_dacvae_model_compute"] = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                mx.eval(frames)
                timings_ms["decode_dacvae_materialization"] = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                samples = _as_numpy(frames.reshape((-1,))).astype("float32", copy=False)
                timings_ms["decode_dacvae_host_transfer"] = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            if max_samples is not None:
                samples = samples[: int(max_samples)]
            timings_ms["decode_dacvae_postprocess"] = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            output = save_wav_numpy(output_path, samples, self.sample_rate)
            timings_ms["decode_dacvae_wav_serialization"] = (time.perf_counter() - started) * 1000.0
            timings_ms["decode_dacvae_model"] = (
                timings_ms["decode_dacvae_model_compute"]
                + timings_ms["decode_dacvae_materialization"]
                + timings_ms["decode_dacvae_host_transfer"]
                + timings_ms["decode_dacvae_postprocess"]
            )
            timings_ms["audio_write"] = timings_ms["decode_dacvae_wav_serialization"]
            return output, dict(timings_ms)
        finally:
            del waveform
            del frames
            del samples
            started = time.perf_counter()
            release_mlx_runtime_memory()
            timings_ms["decode_dacvae_cleanup"] = (time.perf_counter() - started) * 1000.0
            timings_ms["audio_write_wav"] = timings_ms.get("decode_dacvae_wav_serialization", 0.0)
            timings_ms["decode_dacvae_model"] = sum(
                float(timings_ms.get(key, 0.0))
                for key in (
                    "decode_dacvae_model_compute",
                    "decode_dacvae_materialization",
                    "decode_dacvae_host_transfer",
                    "decode_dacvae_postprocess",
                )
            )
            self.last_decode_timings_ms = timings_ms

    def decode_to_wav(self, latents: mx.array, output_path: str | Path, *, max_samples: int | None = None) -> Path:
        output, _ = self.decode_to_wav_timed(latents, output_path, max_samples=max_samples)
        return output


def _load_audio_numpy(path: str | Path) -> tuple["object", int]:
    import numpy as np

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32")
    except Exception:
        with wave.open(str(path), "rb") as fh:
            sample_rate = int(fh.getframerate())
            channels = int(fh.getnchannels())
            width = int(fh.getsampwidth())
            frames = fh.readframes(fh.getnframes())
        if width != 2:
            raise RuntimeError("stdlib WAV fallback only supports PCM16 reference audio")
        data = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
        if channels > 1:
            data = data.reshape((-1, channels)).mean(axis=1)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype="float32"), int(sample_rate)


def _frame_audio(samples, hop_length: int):
    import numpy as np

    hop = int(hop_length)
    pad = (-int(samples.shape[0])) % hop
    if pad:
        mode = "reflect" if int(samples.shape[0]) > 1 else "edge"
        samples = np.pad(samples, (0, pad), mode=mode)
    return samples.reshape((-1, hop)).astype("float32", copy=False)


def _pad_audio_to_hop(samples, hop_length: int):
    import numpy as np

    samples = np.asarray(samples, dtype="float32")
    hop = int(hop_length)
    pad = (-int(samples.shape[0])) % hop
    if not pad:
        return samples
    mode = "reflect" if int(samples.shape[0]) > 1 and pad < int(samples.shape[0]) else "edge"
    return np.pad(samples, (0, pad), mode=mode).astype("float32", copy=False)


def _resample_audio_linear(samples, *, source_rate: int, target_rate: int):
    import numpy as np

    if int(source_rate) <= 0:
        raise ValueError(f"reference audio sample_rate must be positive, got {source_rate}")
    if int(target_rate) <= 0:
        raise ValueError(f"codec sample_rate must be positive, got {target_rate}")
    samples = np.asarray(samples, dtype="float32")
    if int(source_rate) == int(target_rate) or samples.size == 0:
        return samples
    target_len = max(1, int(round(samples.shape[0] * float(target_rate) / float(source_rate))))
    if samples.shape[0] == 1:
        return np.full((target_len,), float(samples[0]), dtype="float32")
    source_positions = np.linspace(0.0, float(samples.shape[0] - 1), num=int(target_len), dtype=np.float64)
    source_index = np.arange(samples.shape[0], dtype=np.float64)
    return np.interp(source_positions, source_index, samples.astype("float64")).astype("float32", copy=False)


def _ensure_audio_peak(samples):
    import numpy as np

    samples = np.asarray(samples, dtype="float32")
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples.astype("float32", copy=False)


def _normalize_audio_db(samples, *, target_db: float):
    import numpy as np

    samples = np.asarray(samples, dtype="float32")
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(np.square(samples.astype("float64")))))
    if rms <= 0.0:
        return samples
    target_rms = 10.0 ** (float(target_db) / 20.0)
    return _ensure_audio_peak(samples * (target_rms / rms))


def save_wav_numpy(path: str | Path, samples, sample_rate: int) -> Path:
    """Save mono float32 audio without importing PyTorch."""

    import numpy as np

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(out), np.asarray(samples, dtype="float32"), int(sample_rate))
        return out
    except Exception:
        pass
    clipped = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return out



class InferenceRuntime:
    """End-to-end prototype: MLX RF-DiT latent generation + selectable DACVAE decode."""

    def __init__(
        self,
        *,
        config: MLXRuntimeConfig,
        model: TextToLatentRFDiT | None = None,
        bridge: MLXDACVAEBridge | None = None,
        tokenizer: PretrainedTextTokenizer | None = None,
        caption_tokenizer: PretrainedTextTokenizer | None = None,
    ) -> None:
        self.config = config
        self.model = model or load_mlx_model(config.model_config, config.weights_path)
        if bridge is None:
            if config.codec.runtime_mode == "mlx":
                bridge = MLXDACVAEBridge(config=config.codec)
            else:
                raise ValueError(f"Unsupported codec runtime_mode={config.codec.runtime_mode!r}")
        self.bridge = bridge
        if self.bridge.latent_dim != int(config.model_config.latent_dim):
            raise ValueError(
                f"DACVAE latent_dim={self.bridge.latent_dim} does not match model latent_dim={config.model_config.latent_dim}."
            )
        text_repo = config.text_tokenizer_repo or config.model_config.text_tokenizer_repo
        try:
            self.tokenizer = tokenizer or PretrainedTextTokenizer.from_pretrained(
                text_repo,
                add_bos=bool(config.model_config.text_add_bos),
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to initialize text tokenizer for repo {text_repo!r}: {exc}") from exc
        self.caption_tokenizer = caption_tokenizer
        if config.model_config.use_caption_condition and self.caption_tokenizer is None:
            caption_repo = config.caption_tokenizer_repo or config.model_config.caption_tokenizer_repo_resolved
            try:
                self.caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                    caption_repo,
                    add_bos=bool(config.model_config.caption_add_bos_resolved),
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to initialize caption tokenizer for repo {caption_repo!r}: {exc}"
                ) from exc

    def generate(self, request: SamplingRequest) -> SamplingResult:
        messages: list[str] = []
        timings_ms: dict[str, float] = {}
        total_started = time.perf_counter()
        manual_seconds = None if request.seconds is None else float(request.seconds)
        if manual_seconds is not None and manual_seconds <= 0:
            raise ValueError(f"seconds must be positive when provided, got {request.seconds!r}")
        if float(request.duration_scale) <= 0:
            raise ValueError(f"duration_scale must be positive, got {request.duration_scale!r}")
        if request.max_auto_seconds is not None and float(request.max_auto_seconds) <= 0:
            raise ValueError(f"max_auto_seconds must be positive when provided, got {request.max_auto_seconds!r}")
        if request.max_auto_estimate_seconds is not None and float(request.max_auto_estimate_seconds) <= 0:
            raise ValueError(
                "max_auto_estimate_seconds must be positive when provided, "
                f"got {request.max_auto_estimate_seconds!r}"
            )
        rescale_k = None if request.rescale_k is None else float(request.rescale_k)
        rescale_sigma = None if request.rescale_sigma is None else float(request.rescale_sigma)
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and (not math.isfinite(rescale_k) or rescale_k <= 0):
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and (not math.isfinite(rescale_sigma) or rescale_sigma <= 0):
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")
        speaker_kv_scale = None if request.speaker_kv_scale is None else float(request.speaker_kv_scale)
        speaker_kv_min_t = None
        speaker_kv_max_layers = None if request.speaker_kv_max_layers is None else int(request.speaker_kv_max_layers)
        if speaker_kv_scale is not None:
            if not self.config.model_config.use_speaker_condition:
                raise ValueError("speaker_kv_scale requires a speaker-conditioned checkpoint.")
            if not math.isfinite(speaker_kv_scale) or speaker_kv_scale <= 0:
                raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
            speaker_kv_min_t = 0.9 if request.speaker_kv_min_t is None else float(request.speaker_kv_min_t)
            if not math.isfinite(speaker_kv_min_t) or not (0.0 <= speaker_kv_min_t <= 1.0):
                raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
            if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                raise ValueError(
                    f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                )
        ref_latent_path = request.ref_latent or None
        if request.ref_embed and (request.ref_wav is not None or ref_latent_path is not None or request.no_ref):
            raise ValueError("Specify only one of ref_embed, ref_latent, ref_wav, or no_ref.")
        if request.ref_embed and not self.config.model_config.use_speaker_condition:
            raise ValueError("ref_embed requires a speaker-conditioned checkpoint.")
        if ref_latent_path and not self.config.model_config.use_speaker_condition:
            raise ValueError("ref_latent requires a speaker-conditioned checkpoint.")
        if ref_latent_path and (request.ref_wav is not None or request.no_ref):
            raise ValueError("Specify only one of ref_latent, ref_wav, or no_ref.")
        if (
            request.ref_wav is None
            and ref_latent_path is None
            and request.ref_embed is None
            and not request.no_ref
            and self.config.model_config.use_speaker_condition
        ):
            raise ValueError(
                "Specify ref_wav, ref_latent, ref_embed, or set no_ref=True for an unconditional speaker path."
            )

        normalized_text = normalize_text(request.text).strip()
        if normalized_text == "":
            raise ValueError("text became empty after normalization.")

        started = time.perf_counter()
        text_ids, text_mask = self.tokenizer.encode(
            normalized_text,
            max_length=int(self.config.max_text_len),
        )
        caption_ids = caption_mask = None
        caption_text: str | None = None
        if self.config.model_config.use_caption_condition:
            caption_text = "" if request.caption is None else str(request.caption)
            caption_max = self.config.max_caption_len or self.config.max_text_len
            assert self.caption_tokenizer is not None
            caption_ids, caption_mask = self.caption_tokenizer.encode(caption_text, max_length=int(caption_max))
            if caption_text.strip() == "":
                caption_mask = mx.zeros_like(caption_mask)
        timings_ms["prepare_text_condition"] = (time.perf_counter() - started) * 1000.0

        ref_latent = ref_mask = None
        speaker_state = speaker_mask = None
        speaker_embed_metadata: dict[str, object] | None = None
        reference_latent_metadata: dict[str, object] | None = None
        speaker_condition_source = "none"
        started = time.perf_counter()
        if self.config.model_config.use_speaker_condition:
            if request.no_ref:
                ref_len = max(1, int(self.config.model_config.speaker_patch_size))
                ref_latent = mx.zeros(
                    (1, ref_len, int(self.config.model_config.patched_latent_dim)),
                    dtype=mx.float32,
                )
                ref_mask = mx.zeros((1, ref_len), dtype=mx.bool_)
                messages.append("speaker reference disabled; using unconditional speaker mask")
                speaker_condition_source = "none"
            elif request.ref_embed is not None:
                embed_started = time.perf_counter()
                speaker_state, speaker_mask, speaker_embed_metadata = load_speaker_embedding_safetensors(
                    request.ref_embed,
                    speaker_dim=int(self.config.model_config.speaker_dim),
                )
                timings_ms["load_speaker_embedding"] = (time.perf_counter() - embed_started) * 1000.0
                ref_len = max(1, int(self.config.model_config.speaker_patch_size))
                ref_latent = mx.zeros(
                    (1, ref_len, int(self.config.model_config.patched_latent_dim)),
                    dtype=mx.float32,
                )
                ref_mask = mx.zeros((1, ref_len), dtype=mx.bool_)
                speaker_condition_source = "embedding"
                messages.append(
                    "speaker embedding loaded: "
                    f"tensor={speaker_embed_metadata['tensor_key']} shape={speaker_embed_metadata['shape']}"
                )
            elif ref_latent_path is not None:
                latent_started = time.perf_counter()
                raw_ref, ref_mask, reference_latent_metadata = load_reference_latent_npz(
                    ref_latent_path,
                    latent_dim=int(self.config.model_config.latent_dim),
                )
                timings_ms["load_reference_latent"] = (time.perf_counter() - latent_started) * 1000.0
                ref_latent = patch_latents_drop_tail(raw_ref, int(self.config.model_config.latent_patch_size))
                ref_mask = mx.ones((1, ref_latent.shape[1]), dtype=mx.bool_)
                speaker_condition_source = "reference_latent"
                messages.append(
                    "reference latent loaded: "
                    f"tensor={reference_latent_metadata['tensor_key']} shape={reference_latent_metadata['shape']}"
                )
            else:
                assert request.ref_wav is not None
                encode_started = time.perf_counter()
                raw_ref = self.bridge.encode_reference(
                    request.ref_wav,
                    max_seconds=request.max_ref_seconds,
                    normalize_db=self.config.codec.normalize_db,
                    ensure_max=True,
                )
                timings_ms["encode_dacvae"] = (time.perf_counter() - encode_started) * 1000.0
                ref_latent = patch_latents_drop_tail(raw_ref, int(self.config.model_config.latent_patch_size))
                ref_mask = mx.ones((1, ref_latent.shape[1]), dtype=mx.bool_)
                speaker_condition_source = "ref_wav"
        timings_ms["prepare_reference_condition"] = (time.perf_counter() - started) * 1000.0

        duration_mode = "fallback"
        resolved_seconds: float | None = None
        if manual_seconds is not None:
            duration_mode = "manual"
            target_samples = int(manual_seconds * float(self.bridge.sample_rate))
            latent_steps = max(1, (target_samples + self.bridge.hop_length - 1) // self.bridge.hop_length)
            resolved_seconds = float(target_samples) / float(self.bridge.sample_rate)
            messages.append(f"manual duration override active: {resolved_seconds:.3f}s")
        elif self.config.model_config.use_duration_predictor:
            duration_mode = "predicted"
            started = time.perf_counter()
            has_speaker = mx.zeros((1,), dtype=mx.bool_)
            if speaker_mask is not None:
                has_speaker = speaker_mask.any(axis=1)
            elif ref_mask is not None:
                has_speaker = ref_mask.any(axis=1)
            duration_features = build_duration_features(
                [normalized_text],
                token_counts=text_mask.sum(axis=1),
                max_text_len=int(self.config.max_text_len),
                has_speaker=has_speaker,
            )
            encoded = self.model.encode_conditions(
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
            )
            if speaker_state is not None and speaker_mask is not None:
                from dataclasses import replace

                encoded = replace(
                    encoded,
                    speaker_state=speaker_state,
                    speaker_mask=speaker_mask,
                )
            pred_log_frames = self.model.predict_duration_log_frames(
                text_state=encoded.text_state,
                text_mask=encoded.text_mask,
                speaker_state=encoded.speaker_state,
                speaker_mask=encoded.speaker_mask,
                duration_features=duration_features,
                has_speaker=has_speaker,
            )
            pred_frames = float(_as_numpy(mx.expm1(pred_log_frames).astype(mx.float32)).mean())
            scaled_frames = max(1.0, pred_frames * float(request.duration_scale))
            max_auto_seconds = None if request.max_auto_seconds is None else float(request.max_auto_seconds)
            max_auto_estimate_seconds = (
                max_auto_seconds
                if request.max_auto_estimate_seconds is None
                else float(request.max_auto_estimate_seconds)
            )
            auto_duration_cap_active = False
            if max_auto_seconds is not None:
                baseline_seconds = estimate_fallback_duration_seconds(normalized_text)
                if baseline_seconds <= max_auto_estimate_seconds:
                    max_auto_frames = max(1.0, max_auto_seconds * float(self.bridge.sample_rate) / float(self.bridge.hop_length))
                    if scaled_frames > max_auto_frames:
                        messages.append(
                            "auto duration cap active for short prompt: "
                            f"predicted_frames={pred_frames:.1f}, scale={float(request.duration_scale):.3f}, "
                            f"estimate={baseline_seconds:.3f}s, max_estimate={max_auto_estimate_seconds:.3f}s, "
                            f"max_seconds={max_auto_seconds:.3f}"
                        )
                        scaled_frames = max_auto_frames
                        auto_duration_cap_active = True
            latent_steps = max(1, int(round(scaled_frames)))
            target_samples = int(latent_steps * self.bridge.hop_length)
            if auto_duration_cap_active and max_auto_seconds is not None:
                target_samples = min(target_samples, max(1, int(max_auto_seconds * float(self.bridge.sample_rate))))
            resolved_seconds = float(target_samples) / float(self.bridge.sample_rate)
            timings_ms["predict_duration"] = (time.perf_counter() - started) * 1000.0
            messages.append(
                "predicted duration active: "
                f"frames={pred_frames:.1f}, scale={float(request.duration_scale):.3f}, seconds={resolved_seconds:.3f}"
            )
            warning = predicted_duration_overallocation_warning(
                normalized_text,
                predicted_seconds=resolved_seconds,
            )
            if warning:
                messages.append(warning)
        elif self.config.model_config.use_caption_condition:
            duration_mode = "estimated"
            fallback_seconds = estimate_voicedesign_duration_seconds(normalized_text, caption=caption_text)
            fallback_seconds *= float(request.duration_scale)
            target_samples = max(int(self.bridge.hop_length), int(fallback_seconds * float(self.bridge.sample_rate)))
            latent_steps = max(1, (target_samples + self.bridge.hop_length - 1) // self.bridge.hop_length)
            resolved_seconds = float(target_samples) / float(self.bridge.sample_rate)
            messages.append(
                f"{self.config.model_config.checkpoint_family_label} has no learned duration predictor; "
                "estimated duration from text length and caption style hints: "
                f"scale={float(request.duration_scale):.3f}, seconds={resolved_seconds:.3f}. "
                "Pass --seconds for an explicit duration."
            )
        else:
            fallback_seconds = estimate_fallback_duration_seconds(normalized_text)
            target_samples = int(fallback_seconds * float(self.bridge.sample_rate))
            latent_steps = max(1, (target_samples + self.bridge.hop_length - 1) // self.bridge.hop_length)
            resolved_seconds = float(target_samples) / float(self.bridge.sample_rate)
            messages.append(
                f"{self.config.model_config.checkpoint_family_label} has no duration predictor; "
                "estimated fallback duration from text length: "
                f"{resolved_seconds:.3f}s. Pass --seconds for an explicit duration."
            )
        patched_steps = (latent_steps + int(self.config.model_config.latent_patch_size) - 1) // int(
            self.config.model_config.latent_patch_size
        )
        started = time.perf_counter()
        z_patched = sample_euler_rf_cfg(
            self.model,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            sequence_length=patched_steps,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            num_steps=int(request.num_steps),
            cfg_scale_text=float(request.cfg_scale_text),
            cfg_scale_caption=float(request.cfg_scale_caption),
            cfg_scale_speaker=float(request.cfg_scale_speaker),
            cfg_guidance_mode=request.cfg_guidance_mode,
            cfg_min_t=float(request.cfg_min_t),
            cfg_max_t=float(request.cfg_max_t),
            t_schedule_mode=request.t_schedule_mode,
            sway_coeff=float(request.sway_coeff),
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            seed=int(request.seed),
            use_context_kv_cache=bool(request.context_kv_cache),
        )
        z = unpatch_latents(z_patched, int(self.config.model_config.latent_patch_size))[:, :latent_steps]
        mx.eval(z)
        timings_ms["sample_rf"] = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        if hasattr(self.bridge, "decode_to_wav_timed"):
            output, decode_timings = self.bridge.decode_to_wav_timed(z, request.output_wav, max_samples=target_samples)
            for key, value in decode_timings.items():
                timings_ms[key] = float(value)
        else:
            output = self.bridge.decode_to_wav(z, request.output_wav, max_samples=target_samples)
        timings_ms["decode_dacvae"] = (time.perf_counter() - started) * 1000.0
        for key, value in getattr(self.bridge, "last_decode_timings_ms", {}).items():
            timings_ms[str(key)] = float(value)
        timings_ms["total_to_decode"] = (time.perf_counter() - total_started) * 1000.0
        codec_boundaries = self.describe_boundaries()["codec"]
        codec_encode_backend = (
            "not-required"
            if request.no_ref
            or request.ref_embed is not None
            or ref_latent_path is not None
            or not self.config.model_config.use_speaker_condition
            else str(codec_boundaries["encode_backend"])
        )
        return SamplingResult(
            output_wav=str(output),
            sample_rate=int(self.bridge.sample_rate),
            samples=target_samples,
            latent_steps=int(latent_steps),
            patched_steps=int(patched_steps),
            seed=int(request.seed),
            duration_mode=duration_mode,
            checkpoint_family=self.config.model_config.checkpoint_family,
            checkpoint_capabilities=self.config.model_config.checkpoint_capabilities,
            speaker_condition_source=speaker_condition_source,
            requested_seconds=manual_seconds,
            resolved_seconds=resolved_seconds,
            timings_ms=timings_ms,
            messages=tuple(messages),
            codec_backend=str(codec_boundaries["decode_backend"]),
            codec_encode_backend=codec_encode_backend,
            codec_decode_backend=str(codec_boundaries["decode_backend"]),
            t_schedule_mode=request.t_schedule_mode,
            sway_coeff=float(request.sway_coeff),
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
        )

    def describe_boundaries(self) -> dict[str, object]:
        return {
            "mlx": {
                "model": "TextToLatentRFDiT",
                "weights_path": self.config.weights_path,
                "latent_layout": "(batch, time, latent_dim)",
            },
            "codec_artifact": {
                "codec_repo": self.config.codec.codec_repo,
                "codec_device": self.config.codec.codec_device,
                "codec_runtime_mode": self.config.codec.runtime_mode,
                "codec_path": self.config.codec.codec_path,
                "sample_rate": self.bridge.sample_rate,
                "hop_length": self.bridge.hop_length,
            },
            "codec": {
                "implementation": self.bridge.__class__.__name__,
                "imports_pytorch": False,
                "encode_imports_pytorch": False,
                "decode_imports_pytorch": False,
                "encode_backend": getattr(self.bridge, "encode_backend", self.config.codec.runtime_mode),
                "decode_backend": getattr(self.bridge, "decode_backend", self.config.codec.runtime_mode),
                "capabilities": describe_codec_capabilities(self.config.codec, model_config=self.config.model_config),
            },
            "conversion": (
                "MLX codec artifact encode/decode; WAV IO crosses through NumPy"
            ),
            "config": asdict(self.config),
            "checkpoint_family": self.config.model_config.checkpoint_family,
            "checkpoint_family_label": self.config.model_config.checkpoint_family_label,
            "checkpoint_capabilities": self.config.model_config.checkpoint_capabilities,
        }


def iter_messages(result: SamplingResult) -> Iterable[str]:
    yield f"wrote: {result.output_wav}"
    yield f"sample_rate: {result.sample_rate}"
    yield f"samples: {result.samples}"
    yield f"latent_steps: {result.latent_steps}"
    yield f"patched_steps: {result.patched_steps}"
    yield f"seed: {result.seed}"
    yield f"checkpoint_family: {result.checkpoint_family}"
    yield "checkpoint_capabilities: " + ", ".join(result.checkpoint_capabilities)
    yield f"duration_mode: {result.duration_mode}"
    yield f"t_schedule_mode: {result.t_schedule_mode}"
    yield f"sway_coeff: {result.sway_coeff}"
    if result.rescale_k is not None or result.rescale_sigma is not None:
        yield f"rescale_k: {result.rescale_k}"
        yield f"rescale_sigma: {result.rescale_sigma}"
    if result.speaker_kv_scale is not None:
        yield f"speaker_kv_scale: {result.speaker_kv_scale}"
        yield f"speaker_kv_min_t: {result.speaker_kv_min_t}"
        yield f"speaker_kv_max_layers: {result.speaker_kv_max_layers}"
    if getattr(result, "codec_backend", None):
        yield f"codec_backend: {result.codec_backend}"
    if getattr(result, "codec_encode_backend", None):
        yield f"codec_encode_backend: {result.codec_encode_backend}"
    if getattr(result, "codec_decode_backend", None):
        yield f"codec_decode_backend: {result.codec_decode_backend}"
    if result.requested_seconds is not None:
        yield f"requested_seconds: {result.requested_seconds}"
    if result.resolved_seconds is not None:
        yield f"resolved_seconds: {result.resolved_seconds:.3f}"
    for name, value in sorted((result.timings_ms or {}).items()):
        yield f"[timing] {name}: {value:.3f} ms"
    for message in result.messages:
        yield message
