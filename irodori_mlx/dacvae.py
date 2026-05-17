from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


EXECUTABLE_DECODER_PREFIX = "dacvae_decoder_exec/"
EXECUTABLE_ENCODER_PREFIX = "dacvae_encoder_exec/"


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


class DACVAEQuantizerInProj(DACVAEWNConv1d):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, kernel_size=1, norm="weight_norm")


@dataclass(frozen=True)
class SemanticDACVAEEncoderConfig:
    input_channels: int = 1
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 8, 10, 12)
    latent_dim: int = 1024
    codebook_dim: int = 32


class DACVAEEncoderBlock(nn.Module):
    """Main Semantic-DACVAE encoder block."""

    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.residuals = [
            DACVAEResidualUnit(input_dim, dilation=1, act="Snake"),
            DACVAEResidualUnit(input_dim, dilation=3, act="Snake"),
            DACVAEResidualUnit(input_dim, dilation=9, act="Snake"),
        ]
        self.downsample_act = DACVAESnake1d(input_dim)
        self.downsample = DACVAEWNConv1d(
            input_dim,
            output_dim,
            kernel_size=2 * int(stride),
            stride=int(stride),
            causal=False,
            pad_mode="none",
            norm="weight_norm",
        )

    def __call__(self, x: mx.array) -> mx.array:
        for residual in self.residuals:
            x = residual(x)
        return self.downsample(self.downsample_act(x))


class SemanticDACVAEEncoder(nn.Module):
    """Encode mono waveforms into Semantic-DACVAE 32-dim latent means."""

    def __init__(self, config: SemanticDACVAEEncoderConfig = SemanticDACVAEEncoderConfig()):
        super().__init__()
        self.config = config
        self.conv_in = DACVAEWNConv1d(config.input_channels, config.encoder_dim, kernel_size=7, stride=1)
        blocks = []
        for idx, stride in enumerate(config.encoder_rates):
            input_dim = config.encoder_dim * (2**idx)
            output_dim = config.encoder_dim * (2 ** (idx + 1))
            blocks.append(DACVAEEncoderBlock(input_dim, output_dim, int(stride)))
        self.blocks = blocks
        final_dim = config.encoder_dim * (2 ** len(config.encoder_rates))
        if final_dim != int(config.latent_dim):
            raise ValueError(f"encoder final dim {final_dim} must match latent_dim={config.latent_dim}")
        self.snake_out = DACVAESnake1d(config.latent_dim)
        self.conv_out = DACVAEWNConv1d(config.latent_dim, config.latent_dim, kernel_size=3, stride=1)
        self.quantizer_in_proj = DACVAEQuantizerInProj(config.latent_dim, 2 * config.codebook_dim)

    def __call__(self, samples: mx.array) -> mx.array:
        if len(samples.shape) != 3:
            raise ValueError(f"SemanticDACVAEEncoder expects samples shaped (B,T,C), got {samples.shape}")
        if int(samples.shape[2]) != int(self.config.input_channels):
            raise ValueError(f"Expected input_channels={self.config.input_channels}, got {samples.shape[2]}")
        x = self.conv_in(samples)
        for block in self.blocks:
            x = block(x)
        projected = self.quantizer_in_proj(self.conv_out(self.snake_out(x)))
        return projected[:, :, : int(self.config.codebook_dim)]


@dataclass(frozen=True)
class SemanticDACVAEDecoderConfig:
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: tuple[int, ...] = (12, 10, 8, 2)
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


def semantic_dacvae_decoder_config_from_metadata(metadata: dict[str, object] | None) -> SemanticDACVAEDecoderConfig:
    payload = (metadata or {}).get("semantic_dacvae_decoder_config")
    if payload is None:
        return SemanticDACVAEDecoderConfig()
    if not isinstance(payload, dict):
        raise ValueError("semantic_dacvae_decoder_config metadata must be an object")
    allowed = {
        "latent_dim",
        "decoder_dim",
        "decoder_rates",
        "wm_rates",
        "codebook_dim",
        "output_channels",
    }
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    if "decoder_rates" in kwargs:
        kwargs["decoder_rates"] = tuple(int(value) for value in kwargs["decoder_rates"])
    if "wm_rates" in kwargs:
        kwargs["wm_rates"] = tuple(int(value) for value in kwargs["wm_rates"])
    return SemanticDACVAEDecoderConfig(**kwargs)


def semantic_dacvae_encoder_config_from_metadata(metadata: dict[str, object] | None) -> SemanticDACVAEEncoderConfig:
    payload = (metadata or {}).get("semantic_dacvae_encoder_config")
    if payload is None:
        return SemanticDACVAEEncoderConfig()
    if not isinstance(payload, dict):
        raise ValueError("semantic_dacvae_encoder_config metadata must be an object")
    allowed = {
        "input_channels",
        "encoder_dim",
        "encoder_rates",
        "latent_dim",
        "codebook_dim",
    }
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    if "encoder_rates" in kwargs:
        kwargs["encoder_rates"] = tuple(int(value) for value in kwargs["encoder_rates"])
    return SemanticDACVAEEncoderConfig(**kwargs)


def semantic_dacvae_encoder_required_keys(
    config: SemanticDACVAEEncoderConfig = SemanticDACVAEEncoderConfig(),
) -> tuple[str, ...]:
    return tuple(semantic_dacvae_encoder_expected_shapes(config))


def semantic_dacvae_encoder_expected_shapes(
    config: SemanticDACVAEEncoderConfig = SemanticDACVAEEncoderConfig(),
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    def conv(prefix: str, *, in_channels: int, out_channels: int, kernel_size: int) -> None:
        shapes[f"{prefix}.weight_g"] = (int(out_channels), 1, 1)
        shapes[f"{prefix}.weight_v"] = (int(out_channels), int(kernel_size), int(in_channels))
        shapes[f"{prefix}.bias"] = (int(out_channels),)

    conv("conv_in", in_channels=config.input_channels, out_channels=config.encoder_dim, kernel_size=7)
    for index, stride in enumerate(config.encoder_rates):
        block = f"blocks.{index}"
        input_dim = config.encoder_dim * (2**index)
        output_dim = config.encoder_dim * (2 ** (index + 1))
        for residual_index in range(3):
            residual = f"{block}.residuals.{residual_index}"
            shapes[f"{residual}.act1.alpha"] = (1, 1, int(input_dim))
            conv(f"{residual}.conv1", in_channels=input_dim, out_channels=input_dim, kernel_size=7)
            shapes[f"{residual}.act2.alpha"] = (1, 1, int(input_dim))
            conv(f"{residual}.conv2", in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        shapes[f"{block}.downsample_act.alpha"] = (1, 1, int(input_dim))
        conv(f"{block}.downsample", in_channels=input_dim, out_channels=output_dim, kernel_size=2 * int(stride))
    final_dim = config.encoder_dim * (2 ** len(config.encoder_rates))
    shapes["snake_out.alpha"] = (1, 1, int(final_dim))
    conv("conv_out", in_channels=final_dim, out_channels=config.latent_dim, kernel_size=3)
    conv("quantizer_in_proj", in_channels=config.latent_dim, out_channels=2 * config.codebook_dim, kernel_size=1)
    return shapes


def semantic_dacvae_decoder_required_keys(
    config: SemanticDACVAEDecoderConfig = SemanticDACVAEDecoderConfig(),
) -> tuple[str, ...]:
    return tuple(semantic_dacvae_decoder_expected_shapes(config))


def semantic_dacvae_decoder_expected_shapes(
    config: SemanticDACVAEDecoderConfig = SemanticDACVAEDecoderConfig(),
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    def conv(prefix: str, *, in_channels: int, out_channels: int, kernel_size: int, transposed: bool = False) -> None:
        weight_shape = (int(out_channels), int(kernel_size), int(in_channels))
        shapes[f"{prefix}.weight_g"] = (1, 1, int(in_channels)) if transposed else (int(out_channels), 1, 1)
        shapes[f"{prefix}.weight_v"] = weight_shape
        shapes[f"{prefix}.bias"] = (int(out_channels),)

    conv("quantizer_out_proj", in_channels=config.codebook_dim, out_channels=config.latent_dim, kernel_size=1)
    conv("conv_in", in_channels=config.latent_dim, out_channels=config.decoder_dim, kernel_size=7)
    for index, _stride in enumerate(config.decoder_rates):
        block = f"blocks.{index}"
        input_dim = config.decoder_dim // (2**index)
        output_dim = config.decoder_dim // (2 ** (index + 1))
        shapes[f"{block}.main_upsample.0.alpha"] = (1, 1, int(input_dim))
        conv(
            f"{block}.main_upsample.1",
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2 * int(_stride),
            transposed=True,
        )
        for residual_index in range(3):
            residual = f"{block}.residuals.{residual_index}"
            shapes[f"{residual}.act1.alpha"] = (1, 1, int(output_dim))
            conv(f"{residual}.conv1", in_channels=output_dim, out_channels=output_dim, kernel_size=7)
            shapes[f"{residual}.act2.alpha"] = (1, 1, int(output_dim))
            conv(f"{residual}.conv2", in_channels=output_dim, out_channels=output_dim, kernel_size=1)
    final_dim = config.decoder_dim // (2 ** len(config.decoder_rates))
    shapes["snake_out.alpha"] = (1, 1, int(final_dim))
    conv("conv_out", in_channels=final_dim, out_channels=config.output_channels, kernel_size=7)
    return shapes


def load_semantic_dacvae_decoder_artifact(
    path: str | Path,
    *,
    strict: bool = True,
) -> SemanticDACVAEDecoder:
    """Load an executable Semantic-DACVAE decoder artifact into MLX modules."""

    import numpy as np

    from .weights import assign_named_weights

    artifact_path = Path(path).expanduser()
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            metadata_json = archive["metadata_json"]
            metadata_value = metadata_json.item() if getattr(metadata_json, "shape", ()) == () else metadata_json[0]
            metadata = json.loads(str(metadata_value))
            config = semantic_dacvae_decoder_config_from_metadata(metadata)
            required = set(semantic_dacvae_decoder_required_keys(config))
            weights = {
                key: mx.array(archive[name].astype("float32", copy=False))
                for name in archive.files
                if name.startswith(EXECUTABLE_DECODER_PREFIX)
                for key in (name[len(EXECUTABLE_DECODER_PREFIX) :],)
                if key in required
            }
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Semantic-DACVAE decoder artifact was not found: {artifact_path}") from exc
    except KeyError as exc:
        raise ValueError(f"Semantic-DACVAE decoder artifact {artifact_path} is missing {exc}.") from exc
    if not weights:
        raise ValueError(f"Semantic-DACVAE decoder artifact {artifact_path} has no executable decoder tensors.")

    decoder = SemanticDACVAEDecoder(config)
    assign_named_weights(
        decoder,
        weights,
        required=semantic_dacvae_decoder_required_keys(config),
        strict=strict,
    )
    mx.eval(decoder.parameters())
    return decoder


def load_semantic_dacvae_encoder_artifact(
    path: str | Path,
    *,
    strict: bool = True,
) -> SemanticDACVAEEncoder:
    """Load an executable Semantic-DACVAE encoder artifact into MLX modules."""

    import numpy as np

    from .weights import assign_named_weights

    artifact_path = Path(path).expanduser()
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            metadata_json = archive["metadata_json"]
            metadata_value = metadata_json.item() if getattr(metadata_json, "shape", ()) == () else metadata_json[0]
            metadata = json.loads(str(metadata_value))
            config = semantic_dacvae_encoder_config_from_metadata(metadata)
            required = set(semantic_dacvae_encoder_required_keys(config))
            weights = {
                key: mx.array(archive[name].astype("float32", copy=False))
                for name in archive.files
                if name.startswith(EXECUTABLE_ENCODER_PREFIX)
                for key in (name[len(EXECUTABLE_ENCODER_PREFIX) :],)
                if key in required
            }
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Semantic-DACVAE encoder artifact was not found: {artifact_path}") from exc
    except KeyError as exc:
        raise ValueError(f"Semantic-DACVAE encoder artifact {artifact_path} is missing {exc}.") from exc
    if not weights:
        raise ValueError(f"Semantic-DACVAE encoder artifact {artifact_path} has no executable encoder tensors.")

    encoder = SemanticDACVAEEncoder(config)
    assign_named_weights(
        encoder,
        weights,
        required=semantic_dacvae_encoder_required_keys(config),
        strict=strict,
    )
    mx.eval(encoder.parameters())
    return encoder


__all__ = [
    "DACVAEElu",
    "DACVAEDecoderBlock",
    "DACVAEEncoderBlock",
    "DACVAEQuantizerInProj",
    "DACVAEQuantizerOutProj",
    "DACVAEResidualUnit",
    "DACVAESnake1d",
    "DACVAEWNConv1d",
    "DACVAEWNConvTranspose1d",
    "SemanticDACVAEDecoder",
    "SemanticDACVAEDecoderConfig",
    "SemanticDACVAEEncoder",
    "SemanticDACVAEEncoderConfig",
    "EXECUTABLE_DECODER_PREFIX",
    "EXECUTABLE_ENCODER_PREFIX",
    "dacvae_snake",
    "semantic_dacvae_decoder_config_from_metadata",
    "semantic_dacvae_decoder_expected_shapes",
    "semantic_dacvae_encoder_config_from_metadata",
    "semantic_dacvae_encoder_expected_shapes",
    "load_semantic_dacvae_encoder_artifact",
    "load_semantic_dacvae_decoder_artifact",
    "semantic_dacvae_decoder_required_keys",
    "semantic_dacvae_encoder_required_keys",
]
