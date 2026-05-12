"""Core MLX utilities for the irodori-tts-mlx prototype."""

from .config import ModelConfig
from .encoders import (
    ConditionEncoders,
    EncodedConditions,
    ReferenceLatentEncoder,
    SelfAttention,
    TextBlock,
    TextEncoder,
    masked_mean_token,
)
from .model import DiffusionBlock, JointAttention, TextToLatentRFDiT
from .sampling import euler_timestep_schedule, sample_euler_rf_cfg
from .layers import (
    LowRankAdaLN,
    RMSNorm,
    SwiGLU,
    apply_rotary_emb,
    get_timestep_embedding,
    patch_latents,
    patch_sequence_with_mask,
    precompute_freqs_cis,
    unpatch_latents,
)
from .weights import WeightLoadReport, assign_named_weights, encoder_required_keys, load_npz_weights, rf_dit_required_keys

__all__ = [
    "ConditionEncoders",
    "DiffusionBlock",
    "EncodedConditions",
    "JointAttention",
    "LowRankAdaLN",
    "ModelConfig",
    "RMSNorm",
    "ReferenceLatentEncoder",
    "SelfAttention",
    "SwiGLU",
    "TextBlock",
    "TextEncoder",
    "TextToLatentRFDiT",
    "WeightLoadReport",
    "apply_rotary_emb",
    "assign_named_weights",
    "encoder_required_keys",
    "euler_timestep_schedule",
    "get_timestep_embedding",
    "load_npz_weights",
    "masked_mean_token",
    "patch_latents",
    "patch_sequence_with_mask",
    "precompute_freqs_cis",
    "rf_dit_required_keys",
    "sample_euler_rf_cfg",
    "unpatch_latents",
]
