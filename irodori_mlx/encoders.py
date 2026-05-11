from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, SwiGLU, apply_rotary_emb, patch_sequence_with_mask, precompute_freqs_cis


def _as_bool_mask(mask: mx.array, expected_shape: tuple[int, int], name: str) -> mx.array:
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {mask.shape}")
    return mask.astype(mx.bool_)


def _mask_float(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    return mask.astype(dtype)[..., None]


def _apply_batch_dropout(mask: mx.array, dropout: mx.array | None, name: str) -> mx.array:
    if dropout is None:
        return mask
    if tuple(dropout.shape) != (mask.shape[0],):
        raise ValueError(f"{name} must have shape ({mask.shape[0]},), got {dropout.shape}")
    return mx.where(dropout.astype(mx.bool_)[:, None], mx.zeros_like(mask), mask)


def masked_mean_token(state: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Prepend an upstream-compatible masked mean summary token."""
    if len(state.shape) != 3 or len(mask.shape) != 2:
        raise ValueError(f"Expected state=(B,S,D), mask=(B,S), got state={state.shape} mask={mask.shape}")
    if state.shape[:2] != mask.shape:
        raise ValueError(f"state/mask shape mismatch: state={state.shape} mask={mask.shape}")
    mask = mask.astype(mx.bool_)
    mask_f = _mask_float(mask, state.dtype)
    denom = mx.maximum(mx.sum(mask_f, axis=1, keepdims=True), mx.array(1.0, dtype=state.dtype))
    mean = mx.sum(state * mask_f, axis=1, keepdims=True) / denom
    has_any = mx.any(mask, axis=1, keepdims=True)
    return mx.concatenate([mean, state], axis=1), mx.concatenate([has_any, mask], axis=1)


class SelfAttention(nn.Module):
    """Irodori-TTS encoder self-attention with RoPE and key masking."""

    def __init__(self, dim: int, heads: int, norm_eps: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if (dim // heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)

    def __call__(self, x: mx.array, key_mask: mx.array | None, freqs_cis: mx.array) -> mx.array:
        if len(x.shape) != 3:
            raise ValueError(f"Expected x=(B,S,D), got {x.shape}")
        bsz, seq_len, _ = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        q = apply_rotary_emb(self.q_norm(q), freqs_cis[:seq_len])
        k = apply_rotary_emb(self.k_norm(k), freqs_cis[:seq_len])

        qh = mx.transpose(q, (0, 2, 1, 3))
        kh = mx.transpose(k, (0, 2, 1, 3))
        vh = mx.transpose(v, (0, 2, 1, 3))
        scores = (qh @ mx.transpose(kh, (0, 1, 3, 2))) / math.sqrt(float(self.head_dim))
        if key_mask is not None:
            key_mask = _as_bool_mask(key_mask, (bsz, seq_len), "key_mask")
            scores = mx.where(key_mask[:, None, None, :], scores, mx.array(-1e9, dtype=scores.dtype))
        attn = mx.softmax(scores, axis=-1)
        y = mx.transpose(attn @ vh, (0, 2, 1, 3)).reshape(bsz, seq_len, self.dim)
        y = y * mx.sigmoid(self.gate(x))
        return self.wo(y)


class TextBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, norm_eps: float, dropout: float = 0.0):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = SelfAttention(dim, heads, norm_eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array, freqs_cis: mx.array) -> mx.array:
        x = x + self.dropout(self.attention(self.attention_norm(x), key_mask=mask, freqs_cis=freqs_cis))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class TextEncoder(nn.Module):
    """Shared token encoder used for text and VoiceDesign caption conditioning."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        norm_eps: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [
            TextBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, norm_eps=norm_eps, dropout=dropout)
            for _ in range(layers)
        ]
        self.head_dim = dim // heads
        self._freqs_cis_cache: mx.array | None = None

    def _rope_freqs(self, seq_len: int) -> mx.array:
        if self._freqs_cis_cache is None or self._freqs_cis_cache.shape[0] < seq_len:
            self._freqs_cis_cache = precompute_freqs_cis(self.head_dim, seq_len)
        return self._freqs_cis_cache[:seq_len]

    def __call__(self, input_ids: mx.array, mask: mx.array) -> mx.array:
        if len(input_ids.shape) != 2:
            raise ValueError(f"input_ids must have shape (B,S), got {input_ids.shape}")
        mask = _as_bool_mask(mask, tuple(input_ids.shape), "mask")
        x = self.text_embedding(input_ids.astype(mx.int32))
        mask_f = _mask_float(mask, x.dtype)
        x = x * mask_f
        freqs = self._rope_freqs(input_ids.shape[1])
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs)
            x = x * mask_f
        return x * mask_f


class ReferenceLatentEncoder(nn.Module):
    """Reference-latent encoder used as speaker/style conditioning."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.speaker_patched_latent_dim, cfg.speaker_dim, bias=True)
        self.blocks = [
            TextBlock(
                dim=cfg.speaker_dim,
                heads=cfg.speaker_heads,
                mlp_ratio=cfg.speaker_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.speaker_layers)
        ]
        self.head_dim = cfg.speaker_dim // cfg.speaker_heads
        self._freqs_cis_cache: mx.array | None = None

    def _rope_freqs(self, seq_len: int) -> mx.array:
        if self._freqs_cis_cache is None or self._freqs_cis_cache.shape[0] < seq_len:
            self._freqs_cis_cache = precompute_freqs_cis(self.head_dim, seq_len)
        return self._freqs_cis_cache[:seq_len]

    def __call__(self, latent: mx.array, mask: mx.array) -> mx.array:
        if len(latent.shape) != 3:
            raise ValueError(f"latent must have shape (B,S,D), got {latent.shape}")
        mask = _as_bool_mask(mask, tuple(latent.shape[:2]), "mask")
        x = self.in_proj(latent) / 6.0
        mask_f = _mask_float(mask, x.dtype)
        x = x * mask_f
        freqs = self._rope_freqs(x.shape[1])
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs)
            x = x * mask_f
        return x * mask_f


@dataclass(frozen=True)
class EncodedConditions:
    text_state: mx.array
    text_mask: mx.array
    speaker_state: mx.array | None
    speaker_mask: mx.array | None
    caption_state: mx.array | None
    caption_mask: mx.array | None


class ConditionEncoders(nn.Module):
    """MLX text, speaker, and optional caption conditioning stack."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(
            vocab_size=cfg.text_vocab_size,
            dim=cfg.text_dim,
            layers=cfg.text_layers,
            heads=cfg.text_heads,
            mlp_ratio=cfg.text_mlp_ratio_resolved,
            norm_eps=cfg.norm_eps,
            dropout=cfg.dropout,
        )
        self.text_norm = RMSNorm(cfg.text_dim, eps=cfg.norm_eps)
        self.speaker_encoder = ReferenceLatentEncoder(cfg) if cfg.use_speaker_condition else None
        self.speaker_norm = RMSNorm(cfg.speaker_dim, eps=cfg.norm_eps) if cfg.use_speaker_condition else None
        self.caption_encoder = None
        self.caption_norm = None
        if cfg.use_caption_condition:
            self.caption_encoder = TextEncoder(
                vocab_size=cfg.caption_vocab_size_resolved,
                dim=cfg.caption_dim_resolved,
                layers=cfg.caption_layers_resolved,
                heads=cfg.caption_heads_resolved,
                mlp_ratio=cfg.caption_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
            )
            self.caption_norm = RMSNorm(cfg.caption_dim_resolved, eps=cfg.norm_eps)

    def __call__(
        self,
        *,
        text_input_ids: mx.array,
        text_mask: mx.array,
        ref_latent: mx.array | None,
        ref_mask: mx.array | None,
        caption_input_ids: mx.array | None = None,
        caption_mask: mx.array | None = None,
        text_condition_dropout: mx.array | None = None,
        speaker_condition_dropout: mx.array | None = None,
        caption_condition_dropout: mx.array | None = None,
    ) -> EncodedConditions:
        text_mask = _as_bool_mask(text_mask, tuple(text_input_ids.shape), "text_mask")
        text_mask = _apply_batch_dropout(text_mask, text_condition_dropout, "text_condition_dropout")
        text_state = self.text_norm(self.text_encoder(text_input_ids, text_mask))
        text_state = text_state * _mask_float(text_mask, text_state.dtype)

        speaker_state = None
        speaker_mask = None
        if self.cfg.use_speaker_condition:
            if self.speaker_encoder is None or self.speaker_norm is None:
                raise RuntimeError("speaker conditioning is enabled but modules are missing")
            if ref_latent is None or ref_mask is None:
                raise ValueError("ref_latent and ref_mask are required when speaker conditioning is enabled")
            ref_mask = _as_bool_mask(ref_mask, tuple(ref_latent.shape[:2]), "ref_mask")
            ref_mask = _apply_batch_dropout(ref_mask, speaker_condition_dropout, "speaker_condition_dropout")
            ref_latent, ref_mask = patch_sequence_with_mask(ref_latent, ref_mask, self.cfg.speaker_patch_size)
            speaker_state = self.speaker_norm(self.speaker_encoder(ref_latent, ref_mask))
            speaker_state = speaker_state * _mask_float(ref_mask, speaker_state.dtype)
            speaker_state, speaker_mask = masked_mean_token(speaker_state, ref_mask)

        caption_state = None
        if self.cfg.use_caption_condition:
            if self.caption_encoder is None or self.caption_norm is None:
                raise RuntimeError("caption conditioning is enabled but modules are missing")
            if caption_input_ids is None or caption_mask is None:
                raise ValueError("caption_input_ids and caption_mask are required when caption conditioning is enabled")
            caption_mask = _as_bool_mask(caption_mask, tuple(caption_input_ids.shape), "caption_mask")
            caption_mask = _apply_batch_dropout(caption_mask, caption_condition_dropout, "caption_condition_dropout")
            caption_state = self.caption_norm(self.caption_encoder(caption_input_ids, caption_mask))
            caption_state = caption_state * _mask_float(caption_mask, caption_state.dtype)

        return EncodedConditions(
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
        )
