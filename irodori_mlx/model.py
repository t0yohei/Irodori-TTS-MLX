from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .encoders import EncodedConditions, ConditionEncoders
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


def _as_bool_mask(mask: mx.array, expected_shape: tuple[int, int], name: str) -> mx.array:
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {mask.shape}")
    return mask.astype(mx.bool_)


def _attention_mask_floor(dtype: mx.Dtype) -> mx.array:
    """Return a finite masking floor for the active attention dtype."""
    return mx.array(mx.finfo(dtype).min, dtype=dtype)


class JointAttention(nn.Module):
    """Joint attention over latent tokens plus text/speaker/caption contexts.

    This mirrors upstream Irodori-TTS RF-DiT attention: latent self q/k use
    half-RoPE, while conditioning K/V tensors are projected separately and
    concatenated into the attention key/value sequence.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        text_ctx_dim: int,
        speaker_ctx_dim: int | None,
        caption_ctx_dim: int | None,
        norm_eps: float,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if (dim // heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if heads % 2 != 0:
            raise ValueError("heads must be even for upstream half-RoPE")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wk_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.wv_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.has_speaker_condition = speaker_ctx_dim is not None
        if self.has_speaker_condition:
            self.wk_speaker = nn.Linear(int(speaker_ctx_dim), dim, bias=False)
            self.wv_speaker = nn.Linear(int(speaker_ctx_dim), dim, bias=False)
        self.has_caption_condition = caption_ctx_dim is not None
        if self.has_caption_condition:
            self.wk_caption = nn.Linear(int(caption_ctx_dim), dim, bias=False)
            self.wv_caption = nn.Linear(int(caption_ctx_dim), dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)

    def _apply_rotary_half(self, x: mx.array, freqs_cis: mx.array) -> mx.array:
        rot, passthrough = mx.split(x, 2, axis=-2)
        return mx.concatenate([apply_rotary_emb(rot, freqs_cis), passthrough], axis=-2)

    def project_context_kv(
        self,
        *,
        text_context: mx.array,
        speaker_context: mx.array | None,
        caption_context: mx.array | None = None,
    ) -> tuple[mx.array, ...]:
        if len(text_context.shape) != 3:
            raise ValueError(f"text_context must have shape (B,S,D), got {text_context.shape}")
        bsz = text_context.shape[0]
        k_text = self.wk_text(text_context).reshape(bsz, text_context.shape[1], self.heads, self.head_dim)
        v_text = self.wv_text(text_context).reshape(bsz, text_context.shape[1], self.heads, self.head_dim)
        projected: list[mx.array] = [self.k_norm(k_text), v_text]
        if self.has_speaker_condition:
            if speaker_context is None:
                raise ValueError("speaker_context is required when speaker conditioning is enabled")
            if speaker_context.shape[0] != bsz:
                raise ValueError(
                    "Batch mismatch for context projection: "
                    f"text={text_context.shape} speaker={speaker_context.shape}"
                )
            k_speaker = self.wk_speaker(speaker_context).reshape(
                bsz, speaker_context.shape[1], self.heads, self.head_dim
            )
            v_speaker = self.wv_speaker(speaker_context).reshape(
                bsz, speaker_context.shape[1], self.heads, self.head_dim
            )
            projected.extend([self.k_norm(k_speaker), v_speaker])
        elif speaker_context is not None and speaker_context.shape[0] != bsz:
            raise ValueError(
                "Batch mismatch for ignored speaker context: "
                f"text={text_context.shape} speaker={speaker_context.shape}"
            )
        if self.has_caption_condition:
            if caption_context is None:
                raise ValueError("caption_context is required when caption conditioning is enabled")
            if caption_context.shape[0] != bsz:
                raise ValueError(
                    "Batch mismatch for caption context: "
                    f"text={text_context.shape} caption={caption_context.shape}"
                )
            k_caption = self.wk_caption(caption_context).reshape(
                bsz, caption_context.shape[1], self.heads, self.head_dim
            )
            v_caption = self.wv_caption(caption_context).reshape(
                bsz, caption_context.shape[1], self.heads, self.head_dim
            )
            projected.extend([self.k_norm(k_caption), v_caption])
        elif caption_context is not None and caption_context.shape[0] != bsz:
            raise ValueError(
                "Batch mismatch for ignored caption context: "
                f"text={text_context.shape} caption={caption_context.shape}"
            )
        return tuple(projected)

    def __call__(
        self,
        *,
        x: mx.array,
        text_context: mx.array,
        text_mask: mx.array | None,
        speaker_context: mx.array | None,
        speaker_mask: mx.array | None,
        caption_context: mx.array | None,
        caption_mask: mx.array | None,
        freqs_cis: mx.array,
        self_mask: mx.array | None = None,
        context_kv: tuple[mx.array, ...] | None = None,
    ) -> mx.array:
        if len(x.shape) != 3:
            raise ValueError(f"x must have shape (B,S,D), got {x.shape}")
        bsz, seq_len, _ = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_self = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v_self = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        projected = context_kv or self.project_context_kv(
            text_context=text_context,
            speaker_context=speaker_context,
            caption_context=caption_context,
        )

        offset = 0
        k_text, v_text = projected[offset], projected[offset + 1]
        offset += 2
        k_speaker = v_speaker = None
        if self.has_speaker_condition:
            k_speaker, v_speaker = projected[offset], projected[offset + 1]
            offset += 2
        k_caption = v_caption = None
        if self.has_caption_condition:
            k_caption, v_caption = projected[offset], projected[offset + 1]

        q = self._apply_rotary_half(self.q_norm(q), freqs_cis[:seq_len])
        k_self = self._apply_rotary_half(self.k_norm(k_self), freqs_cis[:seq_len])

        if self_mask is None:
            self_mask = mx.ones((bsz, seq_len), dtype=mx.bool_)
        else:
            self_mask = _as_bool_mask(self_mask, (bsz, seq_len), "self_mask")
        if text_mask is None:
            text_mask = mx.ones((bsz, text_context.shape[1]), dtype=mx.bool_)
        else:
            text_mask = _as_bool_mask(text_mask, (bsz, text_context.shape[1]), "text_mask")

        context_k = [k_self, k_text]
        context_v = [v_self, v_text]
        context_masks = [self_mask, text_mask]
        if self.has_speaker_condition:
            if speaker_context is None or k_speaker is None or v_speaker is None:
                raise ValueError("speaker_context is required when speaker conditioning is enabled")
            if speaker_mask is None:
                speaker_mask = mx.ones((bsz, speaker_context.shape[1]), dtype=mx.bool_)
            else:
                speaker_mask = _as_bool_mask(speaker_mask, (bsz, speaker_context.shape[1]), "speaker_mask")
            context_k.append(k_speaker)
            context_v.append(v_speaker)
            context_masks.append(speaker_mask)
        if self.has_caption_condition:
            if caption_context is None or k_caption is None or v_caption is None:
                raise ValueError("caption_context is required when caption conditioning is enabled")
            if caption_mask is None:
                caption_mask = mx.ones((bsz, caption_context.shape[1]), dtype=mx.bool_)
            else:
                caption_mask = _as_bool_mask(caption_mask, (bsz, caption_context.shape[1]), "caption_mask")
            context_k.append(k_caption)
            context_v.append(v_caption)
            context_masks.append(caption_mask)

        k = mx.concatenate(context_k, axis=1)
        v = mx.concatenate(context_v, axis=1)
        attn_mask = mx.concatenate(context_masks, axis=1)
        qh = mx.transpose(q, (0, 2, 1, 3))
        kh = mx.transpose(k, (0, 2, 1, 3))
        vh = mx.transpose(v, (0, 2, 1, 3))
        scores = (qh @ mx.transpose(kh, (0, 1, 3, 2))) / math.sqrt(float(self.head_dim))
        scores = mx.where(attn_mask[:, None, None, :], scores, _attention_mask_floor(scores.dtype))
        has_key = mx.any(attn_mask, axis=1)[:, None, None, None]
        attn = mx.softmax(scores, axis=-1)
        y = attn @ vh
        y = y * has_key.astype(y.dtype)
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(bsz, seq_len, self.dim)
        y = y * mx.sigmoid(self.gate(x))
        return self.wo(y)


class DiffusionBlock(nn.Module):
    """One RF-DiT block with timestep-conditioned AdaLN attention and MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = JointAttention(
            cfg.model_dim,
            cfg.num_heads,
            cfg.text_dim,
            cfg.speaker_dim if cfg.use_speaker_condition else None,
            cfg.caption_dim_resolved if cfg.use_caption_condition else None,
            norm_eps=cfg.norm_eps,
        )
        self.mlp = SwiGLU(cfg.model_dim, int(cfg.model_dim * cfg.mlp_ratio))
        self.attention_adaln = LowRankAdaLN(cfg.model_dim, cfg.adaln_rank, cfg.norm_eps)
        self.mlp_adaln = LowRankAdaLN(cfg.model_dim, cfg.adaln_rank, cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(
        self,
        *,
        x: mx.array,
        cond_embed: mx.array,
        text_state: mx.array,
        text_mask: mx.array,
        speaker_state: mx.array | None,
        speaker_mask: mx.array | None,
        caption_state: mx.array | None,
        caption_mask: mx.array | None,
        freqs_cis: mx.array,
        self_mask: mx.array | None = None,
        context_kv: tuple[mx.array, ...] | None = None,
    ) -> mx.array:
        h, attention_gate = self.attention_adaln(x, cond_embed)
        x = x + self.dropout(
            attention_gate
            * self.attention(
                x=h,
                text_context=text_state,
                text_mask=text_mask,
                speaker_context=speaker_state,
                speaker_mask=speaker_mask,
                caption_context=caption_state,
                caption_mask=caption_mask,
                freqs_cis=freqs_cis,
                self_mask=self_mask,
                context_kv=context_kv,
            )
        )
        h, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + self.dropout(mlp_gate * self.mlp(h))
        return x


class TextToLatentRFDiT(nn.Module):
    """Text/reference-latent conditioned RF-DiT over patched DACVAE latents."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.condition_encoders = ConditionEncoders(cfg)
        # Root aliases intentionally match upstream checkpoint names so
        # assign_named_weights can load converted RF-DiT archives directly.
        self.text_encoder = self.condition_encoders.text_encoder
        self.text_norm = self.condition_encoders.text_norm
        self.speaker_encoder = self.condition_encoders.speaker_encoder
        self.speaker_norm = self.condition_encoders.speaker_norm
        self.caption_encoder = self.condition_encoders.caption_encoder
        self.caption_norm = self.condition_encoders.caption_norm
        self.cond_module = [
            nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False),
            nn.Linear(cfg.model_dim, cfg.model_dim, bias=False),
            nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False),
        ]
        self.in_proj = nn.Linear(cfg.patched_latent_dim, cfg.model_dim, bias=True)
        self.blocks = [DiffusionBlock(cfg) for _ in range(cfg.num_layers)]
        self.out_norm = RMSNorm(cfg.model_dim, eps=cfg.norm_eps)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.patched_latent_dim, bias=True)
        self.out_proj.weight = mx.zeros_like(self.out_proj.weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias = mx.zeros_like(self.out_proj.bias)
        self.head_dim = cfg.model_dim // cfg.num_heads
        self._freqs_cis_cache: mx.array | None = None

    def _rope_freqs(self, seq_len: int) -> mx.array:
        if self._freqs_cis_cache is None or self._freqs_cis_cache.shape[0] < seq_len:
            self._freqs_cis_cache = precompute_freqs_cis(self.head_dim, seq_len)
        return self._freqs_cis_cache[:seq_len]

    def _condition_embedding(self, t: mx.array, dtype: mx.Dtype) -> mx.array:
        t_embed = get_timestep_embedding(t, self.cfg.timestep_embed_dim).astype(dtype)
        h = nn.silu(self.cond_module[0](t_embed))
        h = nn.silu(self.cond_module[1](h))
        return self.cond_module[2](h)[:, None, :]

    def encode_conditions(self, **kwargs) -> EncodedConditions:
        return self.condition_encoders(**kwargs)

    def forward_with_encoded_conditions(
        self,
        *,
        x_t: mx.array,
        t: mx.array,
        text_state: mx.array,
        text_mask: mx.array,
        speaker_state: mx.array | None,
        speaker_mask: mx.array | None,
        caption_state: mx.array | None = None,
        caption_mask: mx.array | None = None,
        latent_mask: mx.array | None = None,
        context_kv_cache: list[tuple[mx.array, ...]] | None = None,
    ) -> mx.array:
        if len(x_t.shape) != 3:
            raise ValueError(f"x_t must have shape (B,S,D), got {x_t.shape}")
        x_dtype = x_t.dtype
        cond_embed = self._condition_embedding(t, x_dtype)
        x = self.in_proj(x_t)
        freqs = self._rope_freqs(x.shape[1])
        if context_kv_cache is not None and len(context_kv_cache) != len(self.blocks):
            raise ValueError(
                f"context_kv_cache must have one entry per block ({len(self.blocks)}), "
                f"got {len(context_kv_cache)}"
            )
        for i, block in enumerate(self.blocks):
            x = block(
                x=x,
                cond_embed=cond_embed,
                text_state=text_state,
                text_mask=text_mask,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                caption_state=caption_state,
                caption_mask=caption_mask,
                freqs_cis=freqs,
                self_mask=latent_mask,
                context_kv=context_kv_cache[i] if context_kv_cache is not None else None,
            )
        x = self.out_norm(x)
        return self.out_proj(x).astype(x_dtype)

    def __call__(
        self,
        *,
        x_t: mx.array,
        t: mx.array,
        text_input_ids: mx.array,
        text_mask: mx.array,
        ref_latent: mx.array | None,
        ref_mask: mx.array | None,
        caption_input_ids: mx.array | None = None,
        caption_mask: mx.array | None = None,
        latent_mask: mx.array | None = None,
        text_condition_dropout: mx.array | None = None,
        speaker_condition_dropout: mx.array | None = None,
        caption_condition_dropout: mx.array | None = None,
    ) -> mx.array:
        patched = patch_latents(x_t, self.cfg.latent_patch_size)
        if latent_mask is not None and self.cfg.latent_patch_size > 1:
            _, latent_mask = patch_sequence_with_mask(
                mx.zeros_like(x_t[..., :1]),
                latent_mask,
                self.cfg.latent_patch_size,
            )
        encoded = self.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            text_condition_dropout=text_condition_dropout,
            speaker_condition_dropout=speaker_condition_dropout,
            caption_condition_dropout=caption_condition_dropout,
        )
        velocity = self.forward_with_encoded_conditions(
            x_t=patched,
            t=t,
            text_state=encoded.text_state,
            text_mask=encoded.text_mask,
            speaker_state=encoded.speaker_state,
            speaker_mask=encoded.speaker_mask,
            caption_state=encoded.caption_state,
            caption_mask=encoded.caption_mask,
            latent_mask=latent_mask,
        )
        return unpatch_latents(velocity, self.cfg.latent_patch_size)

    def build_context_kv_cache(
        self,
        *,
        text_state: mx.array,
        speaker_state: mx.array | None,
        caption_state: mx.array | None = None,
    ) -> list[tuple[mx.array, ...]]:
        return [
            block.attention.project_context_kv(
                text_context=text_state,
                speaker_context=speaker_state,
                caption_context=caption_state,
            )
            for block in self.blocks
        ]
