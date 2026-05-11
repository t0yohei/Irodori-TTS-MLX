from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mx.array:
    """Precompute RoPE frequencies as complex cos/sin values.

    Matches upstream Irodori-TTS/PyTorch semantics:
    `1 / theta ** (arange(0, dim, 2) / dim)` over positions `0..end-1`.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be a positive even integer, got {dim}")
    if end < 0:
        raise ValueError(f"end must be non-negative, got {end}")

    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)))
    positions = mx.arange(end, dtype=mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    return mx.cos(freqs) + (1j * mx.sin(freqs))


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply RoPE to `(batch, sequence, heads, head_dim)` activations."""
    if len(x.shape) != 4:
        raise ValueError(f"Expected x with shape (B, S, H, Dh), got {x.shape}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {x.shape[-1]}")
    seq_len = x.shape[1]
    if freqs_cis.shape[0] < seq_len or freqs_cis.shape[1] != x.shape[-1] // 2:
        raise ValueError(
            "freqs_cis shape mismatch: "
            f"need at least ({seq_len}, {x.shape[-1] // 2}), got {freqs_cis.shape}"
        )

    x_dtype = x.dtype
    x_float = x.astype(mx.float32)
    x_pairs = x_float.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    freqs = freqs_cis[:seq_len]
    cos = mx.real(freqs)[None, :, None, :]
    sin = mx.imag(freqs)[None, :, None, :]
    rotated = mx.stack(
        [x_real * cos - x_imag * sin, x_real * sin + x_imag * cos],
        axis=-1,
    ).reshape(x.shape)
    return rotated.astype(x_dtype)


def get_timestep_embedding(timestep: mx.array, dim: int) -> mx.array:
    """Return upstream-compatible sinusoidal timestep embeddings.

    The frequency computation is fp32. The returned embedding is cast back to
    the input timestep dtype when that dtype is floating point.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be a positive even integer, got {dim}")
    half = dim // 2
    input_dtype = timestep.dtype
    freqs = 1000.0 * mx.exp(
        -mx.log(mx.array(10000.0, dtype=mx.float32))
        * mx.arange(half, dtype=mx.float32)
        / float(half)
    )
    args = timestep.astype(mx.float32)[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if input_dtype in {mx.float16, mx.bfloat16, mx.float32, mx.float64}:
        return emb.astype(input_dtype)
    return emb


class RMSNorm(nn.Module):
    """Root-mean-square normalization with fp32 internal statistics."""

    def __init__(self, dim: int | Tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        shape = (dim,) if isinstance(dim, int) else tuple(dim)
        if not shape or any(int(d) <= 0 for d in shape):
            raise ValueError(f"dim must describe a positive shape, got {dim}")
        self.weight = mx.ones(shape)
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        x_float = x.astype(mx.float32)
        normalized = x_float * mx.rsqrt(mx.mean(x_float * x_float, axis=-1, keepdims=True) + self.eps)
        return (normalized * self.weight.astype(mx.float32)).astype(x_dtype)


class SwiGLU(nn.Module):
    """SwiGLU MLP: `w2(silu(w1(x)) * w3(x))`."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class LowRankAdaLN(nn.Module):
    """Low-rank AdaLN used by Irodori-TTS diffusion blocks.

    `cond_embed` is split into shift, scale, and gate chunks. Each chunk passes
    through a low-rank residual adapter, then the normalized activation is
    modulated as `x * (1 + scale) + shift`; the residual gate is `tanh(gate)`.
    """

    def __init__(self, model_dim: int, rank: int, eps: float):
        super().__init__()
        rank = max(1, min(int(rank), int(model_dim)))
        self.eps = float(eps)
        self.shift_down = nn.Linear(model_dim, rank, bias=False)
        self.scale_down = nn.Linear(model_dim, rank, bias=False)
        self.gate_down = nn.Linear(model_dim, rank, bias=False)
        self.shift_up = nn.Linear(rank, model_dim, bias=True)
        self.scale_up = nn.Linear(rank, model_dim, bias=True)
        self.gate_up = nn.Linear(rank, model_dim, bias=True)
        self._zero_output_projections()

    def _zero_output_projections(self) -> None:
        for layer in (self.shift_up, self.scale_up, self.gate_up):
            layer.weight = mx.zeros_like(layer.weight)
            if "bias" in layer:
                layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, x: mx.array, cond_embed: mx.array) -> tuple[mx.array, mx.array]:
        if cond_embed.shape[-1] != x.shape[-1] * 3:
            raise ValueError(
                "cond_embed last dimension must be 3 * model_dim, "
                f"got cond={cond_embed.shape[-1]} model_dim={x.shape[-1]}"
            )
        shift, scale, gate = mx.split(cond_embed, 3, axis=-1)
        shift = self.shift_up(self.shift_down(nn.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(nn.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(nn.silu(gate))) + gate

        x_dtype = x.dtype
        x_float = x.astype(mx.float32)
        x_norm = x_float * mx.rsqrt(mx.mean(x_float * x_float, axis=-1, keepdims=True) + self.eps)
        x_modulated = x_norm * (1.0 + scale.astype(mx.float32)) + shift.astype(mx.float32)
        return x_modulated.astype(x_dtype), mx.tanh(gate)


def patch_sequence_with_mask(
    seq: mx.array,
    mask: mx.array,
    patch_size: int,
) -> tuple[mx.array, mx.array]:
    """Patch a `(B,S,D)` sequence and reduce a `(B,S)` mask with all()."""
    if patch_size <= 1:
        return seq, mask
    if len(seq.shape) != 3 or len(mask.shape) != 2:
        raise ValueError(f"Expected seq=(B,S,D), mask=(B,S), got seq={seq.shape} mask={mask.shape}")
    if seq.shape[0] != mask.shape[0] or seq.shape[1] != mask.shape[1]:
        raise ValueError(f"Sequence/mask shape mismatch: seq={seq.shape}, mask={mask.shape}")
    bsz, seq_len, dim = seq.shape
    usable = (seq_len // patch_size) * patch_size
    if usable <= 0:
        raise ValueError(f"Sequence too short for patch_size={patch_size}: seq_len={seq_len}")
    patched = seq[:, :usable].reshape(bsz, usable // patch_size, dim * patch_size)
    patched_mask = mx.all(mask[:, :usable].reshape(bsz, usable // patch_size, patch_size), axis=-1)
    return patched, patched_mask


def patch_latents(latents: mx.array, patch_size: int) -> mx.array:
    """Patch latent sequences from `(B,S,D)` to `(B,S//P,D*P)`."""
    if patch_size <= 1:
        return latents
    if len(latents.shape) != 3:
        raise ValueError(f"Expected latents with shape (B,S,D), got {latents.shape}")
    bsz, seq_len, dim = latents.shape
    usable = (seq_len // patch_size) * patch_size
    if usable != seq_len:
        raise ValueError(f"seq_len={seq_len} must be divisible by patch_size={patch_size}")
    return latents.reshape(bsz, seq_len // patch_size, dim * patch_size)


def unpatch_latents(patched: mx.array, patch_size: int) -> mx.array:
    """Invert `patch_latents`, converting `(B,S,D*P)` back to `(B,S*P,D)`."""
    if patch_size <= 1:
        return patched
    if len(patched.shape) != 3:
        raise ValueError(f"Expected patched latents with shape (B,S,D), got {patched.shape}")
    bsz, seq_len, patched_dim = patched.shape
    if patched_dim % patch_size != 0:
        raise ValueError(f"last dim={patched_dim} must be divisible by patch_size={patch_size}")
    return patched.reshape(bsz, seq_len * patch_size, patched_dim // patch_size)
