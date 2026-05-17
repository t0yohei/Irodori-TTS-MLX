"""Core MLX utilities for the irodori-tts-mlx prototype.

The package keeps top-level exports lazy so pure helpers such as
``irodori_mlx.hosted_weights`` can be imported in metadata-only smoke tests
without initializing MLX/Metal.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "ConditionEncoders": "encoders",
    "DACVAEBridgeConfig": "runtime",
    "DACVAEDecoderBlock": "dacvae",
    "DACVAEElu": "dacvae",
    "DACVAEQuantizerOutProj": "dacvae",
    "DACVAEResidualUnit": "dacvae",
    "DACVAESnake1d": "dacvae",
    "DACVAEWNConv1d": "dacvae",
    "DACVAEWNConvTranspose1d": "dacvae",
    "DiffusionBlock": "model",
    "EncodedConditions": "encoders",
    "EXECUTABLE_DECODER_PREFIX": "dacvae",
    "GenerationRequest": "runtime",
    "GenerationResult": "runtime",
    "JointAttention": "model",
    "LowRankAdaLN": "layers",
    "MLXDACVAERuntime": "runtime",
    "MLXRuntimeConfig": "runtime",
    "ModelConfig": "config",
    "PretrainedTextTokenizer": "runtime",
    "PyTorchDACVAEBridge": "runtime",
    "RMSNorm": "layers",
    "ReferenceLatentEncoder": "encoders",
    "SelfAttention": "encoders",
    "SemanticDACVAEDecoder": "dacvae",
    "SemanticDACVAEDecoderConfig": "dacvae",
    "SwiGLU": "layers",
    "TextBlock": "encoders",
    "TextEncoder": "encoders",
    "TextToLatentRFDiT": "model",
    "WeightLoadReport": "weights",
    "apply_rotary_emb": "layers",
    "assign_named_weights": "weights",
    "build_duration_features": "duration",
    "encoder_required_keys": "weights",
    "euler_timestep_schedule": "sampling",
    "get_timestep_embedding": "layers",
    "iter_messages": "runtime",
    "load_mlx_model": "runtime",
    "load_model_config_json": "runtime",
    "load_npz_weights": "weights",
    "load_semantic_dacvae_decoder_artifact": "dacvae",
    "masked_mean_token": "encoders",
    "mlx_to_torch_latents": "runtime",
    "patch_latents": "layers",
    "patch_sequence_with_mask": "layers",
    "precompute_freqs_cis": "layers",
    "rf_dit_required_keys": "weights",
    "sample_euler_rf_cfg": "sampling",
    "save_wav": "runtime",
    "semantic_dacvae_decoder_config_from_metadata": "dacvae",
    "semantic_dacvae_decoder_expected_shapes": "dacvae",
    "semantic_dacvae_decoder_required_keys": "dacvae",
    "torch_to_mlx_latents": "runtime",
    "unpatch_latents": "layers",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
