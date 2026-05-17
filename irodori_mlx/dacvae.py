from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


def _normalize_weight(weight: mx.array, *, except_dim: int = 0, eps: float = 1e-12) -> mx.array:
    if len(weight.shape) != 3:
        raise ValueError(f"DACVAE weight normalization expects a rank-3 tensor, got {weight.shape}")
    axes = tuple(axis for axis in range(len(weight.shape)) if axis != int(except_dim))
    return mx.sqrt(mx.sum(weight * weight, axis=axes, keepdims=True) + eps)


def dacvae_snake(x: mx.array, alpha: mx.array) -> mx.array:
    return x + mx.square(mx.sin(alpha * x)) / (alpha + 1e-9)


def _elu(x: mx.array, *, alpha: float = 1.0) -> mx.array:
    return mx.where(x > 0, x, float(alpha) * (mx.exp(x) - 1.0))


def _pad1d(x: mx.array, left: int, right: int) -> mx.array:
    if left <= 0 and right <= 0:
        return x
    return mx.pad(x, [(0, 0), (max(0, int(left)), max(0, int(right))), (0, 0)])


def _trim_time(x: mx.array, left: int, right: int) -> mx.array:
    start = max(0, int(left))
    end = int(x.shape[1]) - max(0, int(right))
    if end <= start:
        return x[:, 0:0, :]
    return x[:, start:end, :]


class DACVAESnake1d(nn.Module):
    """Snake activation for channel-last DACVAE sequences shaped (B, T, C)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, int(channels)), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return dacvae_snake(x, self.alpha)


class DACVAEWNConv1d(nn.Module):
    """DACVAE Conv1d with upstream padding semantics and optional weight norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
    ):
        super().__init__()
        if norm not in {"none", "weight_norm"}:
            raise ValueError(f"Unsupported DACVAE conv norm: {norm!r}")
        if pad_mode not in {"none", "auto"}:
            raise ValueError(f"Unsupported DACVAE conv pad_mode: {pad_mode!r}")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.pad_mode = str(pad_mode)
        self.use_weight_norm = norm == "weight_norm"
        self.padding = (
            int(padding)
            if padding is not None
            else ((self.kernel_size - self.stride) * self.dilation // 2 if self.pad_mode == "none" else 0)
        )
        scale = math.sqrt(1.0 / float(int(in_channels) * self.kernel_size))
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(int(out_channels), self.kernel_size, int(in_channels)),
        )
        if self.use_weight_norm:
            self.weight_g = _normalize_weight(weight)
            self.weight_v = weight / (self.weight_g + 1e-12)
        else:
            self.weight = weight
        if bias:
            self.bias = mx.zeros((int(out_channels),), dtype=mx.float32)

    def effective_weight(self) -> mx.array:
        if self.use_weight_norm:
            return self.weight_g * self.weight_v / _normalize_weight(self.weight_v)
        return self.weight

    def _auto_pad(self, x: mx.array) -> mx.array:
        if self.pad_mode == "none":
            return x
        length = int(x.shape[1])
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel - self.stride
        n_frames = (length - effective_kernel + padding_total) / float(self.stride) + 1.0
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - padding_total)
        extra_padding = max(0, ideal_length - length)
        if self.causal:
            return _pad1d(x, padding_total, extra_padding)
        padding_right = extra_padding // 2
        padding_left = padding_total - padding_right
        return _pad1d(x, padding_left, padding_right + extra_padding)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            self._auto_pad(x),
            self.effective_weight(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class DACVAEWNConvTranspose1d(nn.Module):
    """DACVAE ConvTranspose1d with upstream unpadding semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        output_padding: int | None = None,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
    ):
        super().__init__()
        if norm not in {"none", "weight_norm"}:
            raise ValueError(f"Unsupported DACVAE transposed conv norm: {norm!r}")
        if pad_mode not in {"none", "auto"}:
            raise ValueError(f"Unsupported DACVAE transposed conv pad_mode: {pad_mode!r}")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.pad_mode = str(pad_mode)
        self.use_weight_norm = norm == "weight_norm"
        if self.pad_mode == "none":
            default_padding = (self.stride + 1) // 2
            default_output_padding = 1 if self.stride % 2 else 0
        else:
            default_padding = 0
            default_output_padding = 0
        self.padding = int(default_padding if padding is None else padding)
        self.output_padding = int(default_output_padding if output_padding is None else output_padding)
        scale = math.sqrt(1.0 / float(int(in_channels) * self.kernel_size))
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(int(out_channels), self.kernel_size, int(in_channels)),
        )
        if self.use_weight_norm:
            self.weight_g = _normalize_weight(weight, except_dim=2)
            self.weight_v = weight / (self.weight_g + 1e-12)
        else:
            self.weight = weight
        if bias:
            self.bias = mx.zeros((int(out_channels),), dtype=mx.float32)

    def effective_weight(self) -> mx.array:
        if self.use_weight_norm:
            return self.weight_g * self.weight_v / _normalize_weight(self.weight_v, except_dim=2)
        return self.weight

    def _unpad(self, x: mx.array) -> mx.array:
        if self.pad_mode == "none":
            return x
        padding_total = self.kernel_size - self.stride
        if self.causal:
            return _trim_time(x, 0, padding_total)
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        return _trim_time(x, padding_left, padding_right)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(
            x,
            self.effective_weight(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return self._unpad(y)


class DACVAEElu(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def __call__(self, x: mx.array) -> mx.array:
        return _elu(x, alpha=self.alpha)


class DACVAEResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        kernel: int = 7,
        dilation: int = 1,
        act: str = "Snake",
        compress: int = 1,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
        true_skip: bool = False,
    ):
        super().__init__()
        if act not in {"Snake", "ELU"}:
            raise ValueError(f"Unsupported DACVAE residual activation: {act!r}")
        hidden = int(dim) // int(compress)
        self.true_skip = bool(true_skip)
        self.act1 = DACVAESnake1d(dim) if act == "Snake" else DACVAEElu()
        self.conv1 = DACVAEWNConv1d(
            dim,
            hidden,
            int(kernel),
            dilation=int(dilation),
            causal=causal,
            pad_mode=pad_mode,
            norm=norm,
        )
        self.act2 = DACVAESnake1d(hidden) if act == "Snake" else DACVAEElu()
        self.conv2 = DACVAEWNConv1d(
            hidden,
            dim,
            1,
            causal=causal,
            pad_mode=pad_mode,
            norm=norm,
        )

    @staticmethod
    def _shortcut(x: mx.array, y: mx.array, *, true_skip: bool) -> mx.array:
        if true_skip:
            return x
        pad = (int(x.shape[1]) - int(y.shape[1])) // 2
        if pad > 0:
            return x[:, pad : int(x.shape[1]) - pad, :]
        return x

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(self.act1(x))
        y = self.conv2(self.act2(y))
        return y + self._shortcut(x, y, true_skip=self.true_skip)


class DACVAEDecoderBlock(nn.Module):
    """Main Semantic-DACVAE decoder block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        *,
        stride_wm: int | None = None,
        downsampling_factor: int = 3,
        last_kernel_size: int | None = None,
    ):
        super().__init__()
        stride_wm = int(stride if stride_wm is None else stride_wm)
        self.main_upsample = [
            DACVAESnake1d(input_dim),
            DACVAEWNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * int(stride),
                stride=int(stride),
                causal=False,
                pad_mode="none",
                norm="weight_norm",
            ),
        ]
        wm_input = int(input_dim) // int(downsampling_factor)
        wm_output = int(output_dim) // int(downsampling_factor)
        self.watermark_upsample = [
            DACVAEElu(),
            DACVAEWNConvTranspose1d(
                wm_input,
                wm_output,
                kernel_size=2 * stride_wm,
                stride=stride_wm,
                causal=True,
                pad_mode="auto",
                norm="none",
            ),
            DACVAEResidualUnit(
                wm_output,
                kernel=3,
                act="ELU",
                compress=2,
                causal=True,
                pad_mode="auto",
                norm="none",
                true_skip=True,
            ),
            DACVAEResidualUnit(
                wm_output,
                kernel=3,
                act="ELU",
                compress=2,
                causal=True,
                pad_mode="auto",
                norm="none",
                true_skip=True,
            ),
        ]
        self.residuals = [
            DACVAEResidualUnit(output_dim, dilation=1, act="Snake"),
            DACVAEResidualUnit(output_dim, dilation=3, act="Snake"),
            DACVAEResidualUnit(output_dim, dilation=9, act="Snake"),
        ]
        self.last_residual = (
            DACVAEResidualUnit(
                output_dim,
                kernel=int(last_kernel_size),
                act="Snake",
                pad_mode="none",
                norm="weight_norm",
                causal=False,
                true_skip=True,
            )
            if last_kernel_size is not None
            else nn.Identity()
        )
        self.watermark_downsample = [
            DACVAEElu(),
            DACVAEWNConv1d(
                wm_output,
                wm_input,
                kernel_size=2 * stride_wm,
                stride=stride_wm,
                causal=True,
                pad_mode="auto",
                norm="none",
            ),
        ]

    @staticmethod
    def _run(layers: list[nn.Module], x: mx.array) -> mx.array:
        for layer in layers:
            x = layer(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        x = self._run(self.main_upsample, x)
        x = self._run(self.residuals, x)
        return self.last_residual(x)

    def upsample_group(self, x: mx.array) -> mx.array:
        return self._run(self.watermark_upsample, x)

    def downsample_group(self, x: mx.array) -> mx.array:
        return self._run(self.watermark_downsample, x)


class DACVAEQuantizerOutProj(DACVAEWNConv1d):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, kernel_size=1, norm="weight_norm")


@dataclass(frozen=True)
class SemanticDACVAEDecoderConfig:
    latent_dim: int = 32
    decoder_dim: int = 1536
    decoder_rates: tuple[int, ...] = (8, 8, 4, 2)
    wm_rates: tuple[int, ...] = (8, 5, 4, 2)
    codebook_dim: int = 32
    output_channels: int = 1


class SemanticDACVAEDecoder(nn.Module):
    """Decode Irodori runtime latents through the Semantic-DACVAE decoder path.

    Input latents use Irodori runtime layout (B, T, codebook_dim). The MLX
    implementation stays channel-last internally and returns waveform samples as
    (B, samples, output_channels).
    """

    def __init__(self, config: SemanticDACVAEDecoderConfig = SemanticDACVAEDecoderConfig()):
        super().__init__()
        if len(config.decoder_rates) != len(config.wm_rates):
            raise ValueError("decoder_rates and wm_rates must have the same length")
        self.config = config
        self.quantizer_out_proj = DACVAEQuantizerOutProj(config.codebook_dim, config.latent_dim)
        self.conv_in = DACVAEWNConv1d(config.latent_dim, config.decoder_dim, kernel_size=7, stride=1)
        blocks = []
        for idx, (stride, wm_stride) in enumerate(zip(config.decoder_rates, config.wm_rates)):
            input_dim = config.decoder_dim // (2**idx)
            output_dim = config.decoder_dim // (2 ** (idx + 1))
            blocks.append(DACVAEDecoderBlock(input_dim, output_dim, int(stride), stride_wm=int(wm_stride)))
        self.blocks = blocks
        final_dim = config.decoder_dim // (2 ** len(config.decoder_rates))
        self.snake_out = DACVAESnake1d(final_dim)
        self.conv_out = DACVAEWNConv1d(final_dim, config.output_channels, kernel_size=7, stride=1)

    def __call__(self, latents: mx.array) -> mx.array:
        if len(latents.shape) != 3:
            raise ValueError(f"SemanticDACVAEDecoder expects latents shaped (B,T,C), got {latents.shape}")
        if int(latents.shape[2]) != int(self.config.codebook_dim):
            raise ValueError(f"Expected codebook_dim={self.config.codebook_dim}, got {latents.shape[2]}")
        x = self.quantizer_out_proj(latents)
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return mx.tanh(self.conv_out(self.snake_out(x)))


__all__ = [
    "DACVAEElu",
    "DACVAEDecoderBlock",
    "DACVAEQuantizerOutProj",
    "DACVAEResidualUnit",
    "DACVAESnake1d",
    "DACVAEWNConv1d",
    "DACVAEWNConvTranspose1d",
    "SemanticDACVAEDecoder",
    "SemanticDACVAEDecoderConfig",
    "dacvae_snake",
]
