from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import mlx.core as mx

from .config import ModelConfig
from .duration import build_duration_features
from .layers import unpatch_latents
from .model import TextToLatentRFDiT
from .sampling import sample_euler_rf_cfg
from .weights import assign_named_weights, load_npz_weights, rf_dit_required_keys


@dataclass(frozen=True)
class DACVAEBridgeConfig:
    """Configuration for the v0 PyTorch DACVAE bridge."""

    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    codec_device: str = "cpu"
    runtime_mode: str = "persistent"
    deterministic_encode: bool = True
    deterministic_decode: bool = True
    enable_watermark: bool = False
    normalize_db: float | None = -16.0


@dataclass(frozen=True)
class MLXRuntimeConfig:
    """Configuration for the first end-to-end MLX + PyTorch bridge runtime."""

    model_config: ModelConfig
    weights_path: str
    text_tokenizer_repo: str | None = None
    caption_tokenizer_repo: str | None = None
    text_max_length: int = 256
    caption_max_length: int | None = None
    codec: DACVAEBridgeConfig = DACVAEBridgeConfig()


@dataclass(frozen=True)
class GenerationRequest:
    text: str
    output_wav: str
    reference_wav: str | None = None
    no_reference: bool = False
    caption: str | None = None
    seconds: float | None = None
    duration_scale: float = 1.0
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    seed: int = 0
    max_reference_seconds: float | None = 30.0
    use_context_kv_cache: bool = True


@dataclass(frozen=True)
class GenerationResult:
    output_wav: str
    sample_rate: int
    samples: int
    latent_steps: int
    patched_steps: int
    seed: int
    duration_mode: str
    requested_seconds: float | None = None
    resolved_seconds: float | None = None
    timings_ms: dict[str, float] | None = None
    messages: tuple[str, ...] = ()


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps.
        raise RuntimeError(
            "PyTorch is required for the DACVAE bridge. Install torch and the DACVAE runtime dependencies."
        ) from exc
    return torch


def _as_numpy(value: mx.array):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is normally present with MLX.
        raise RuntimeError("numpy is required for MLX/PyTorch tensor conversion.") from exc
    return np.array(value)


def _release_torch_runtime_memory(torch_module, device: str) -> None:
    gc.collect()
    device_name = str(device).lower()
    try:
        if device_name.startswith("mps") and hasattr(torch_module, "mps") and hasattr(torch_module.mps, "empty_cache"):
            torch_module.mps.empty_cache()
        elif device_name.startswith("cuda") and hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
            torch_module.cuda.empty_cache()
    except Exception:
        pass


def torch_to_mlx_latents(tensor) -> mx.array:
    """Convert a PyTorch latent tensor to an MLX array through an explicit CPU boundary."""

    torch = _require_torch()
    if tensor.ndim != 3:
        raise ValueError(f"Expected latent tensor shape (B,T,D), got {tuple(tensor.shape)}")
    return mx.array(tensor.detach().to(device="cpu", dtype=torch.float32).numpy())


def mlx_to_torch_latents(latents: mx.array, *, device: str = "cpu"):
    """Convert an MLX latent array to a PyTorch tensor through an explicit CPU boundary."""

    torch = _require_torch()
    if len(latents.shape) != 3:
        raise ValueError(f"Expected MLX latents with shape (B,T,D), got {latents.shape}")
    return torch.from_numpy(_as_numpy(latents)).to(device=device, dtype=torch.float32)


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
    return ModelConfig(**payload)


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


def load_mlx_model(config: ModelConfig, weights_path: str | Path) -> TextToLatentRFDiT:
    """Load a converted MLX RF-DiT model from an `.npz` archive."""

    model = TextToLatentRFDiT(config)
    weights = load_npz_weights(weights_path)
    assign_named_weights(
        model,
        weights,
        required=rf_dit_required_keys(config),
        strict=True,
    )
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
                "transformers is required for text tokenization. Install transformers and sentencepiece."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=False)
        return cls(tokenizer, add_bos=add_bos)

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


class PyTorchDACVAEBridge:
    """PyTorch DACVAE encode/decode boundary used by the v0 MLX prototype."""

    def __init__(self, *, config: DACVAEBridgeConfig) -> None:
        self.config = config
        self.torch = _require_torch()
        try:
            from irodori_tts.codec import DACVAECodec
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError(
                "The PyTorch DACVAE bridge currently reuses upstream irodori_tts.codec.DACVAECodec. "
                "Install the upstream Irodori-TTS package or add its checkout to PYTHONPATH."
            ) from exc
        self.codec = DACVAECodec.load(
            repo_id=config.codec_repo,
            device=config.codec_device,
            deterministic_encode=config.deterministic_encode,
            deterministic_decode=config.deterministic_decode,
            enable_watermark=config.enable_watermark,
            normalize_db=config.normalize_db,
        )
        self.sample_rate = int(self.codec.sample_rate)
        self.latent_dim = int(self.codec.latent_dim)
        self.hop_length = int(getattr(self.codec.model, "hop_length"))

    def encode_reference(
        self,
        path: str | Path,
        *,
        max_seconds: float | None,
        normalize_db: float | None,
        ensure_max: bool,
    ) -> mx.array:
        wav, sample_rate = _load_audio_torch(path)
        latent = None
        if max_seconds is not None and float(max_seconds) > 0:
            max_samples = max(1, int(float(max_seconds) * float(sample_rate)))
            if wav.shape[-1] > max_samples:
                wav = wav[..., :max_samples]
        try:
            latent = self.codec.encode_waveform(
                wav.unsqueeze(0),
                sample_rate=int(sample_rate),
                normalize_db=normalize_db,
                ensure_max=ensure_max,
            ).cpu()
            return torch_to_mlx_latents(latent)
        finally:
            del wav
            if latent is not None:
                del latent
            _release_torch_runtime_memory(self.torch, str(self.codec.device))

    def decode_to_wav(self, latents: mx.array, output_path: str | Path, *, max_samples: int | None = None) -> Path:
        z = mlx_to_torch_latents(latents, device=str(self.codec.device))
        audio = None
        try:
            audio = self.codec.decode_latent(z).detach().cpu()[0]
            if max_samples is not None:
                audio = audio[:, : int(max_samples)]
            return save_wav(output_path, audio, self.sample_rate)
        finally:
            del z
            if audio is not None:
                del audio
            _release_torch_runtime_memory(self.torch, str(self.codec.device))


def _run_codec_worker(*, action: str, config: DACVAEBridgeConfig, extra_args: list[str]) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    argv = [
        sys.executable,
        "-m",
        "irodori_mlx.codec_worker",
        action,
        "--config-json",
        json.dumps(asdict(config), sort_keys=True),
        *extra_args,
    ]
    completed = subprocess.run(
        argv,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"codec worker failed: {action}")
    return completed.stdout


class SubprocessDACVAEBridge:
    """PyTorch DACVAE boundary that isolates encode/decode work in short-lived subprocesses."""

    def __init__(self, *, config: DACVAEBridgeConfig) -> None:
        self.config = config
        raw = _run_codec_worker(action="describe", config=self._worker_config(config), extra_args=[])
        json_line = next((line for line in reversed(raw.splitlines()) if line.strip().startswith("{")), "")
        if not json_line:
            raise RuntimeError("codec worker describe did not return JSON metadata")
        payload = json.loads(json_line)
        self.sample_rate = int(payload["sample_rate"])
        self.latent_dim = int(payload["latent_dim"])
        self.hop_length = int(payload["hop_length"])

    @staticmethod
    def _caller_absolute_path(path: str | Path) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    @classmethod
    def _caller_resolved_codec_repo(cls, codec_repo: str) -> str:
        candidate = Path(codec_repo).expanduser()
        if candidate.is_absolute() or codec_repo.startswith((".", "~")):
            return str(candidate.resolve(strict=False))
        caller_candidate = Path.cwd() / candidate
        if caller_candidate.exists():
            return str(caller_candidate.resolve(strict=False))
        return codec_repo

    @classmethod
    def _worker_config(cls, config: DACVAEBridgeConfig) -> DACVAEBridgeConfig:
        return replace(config, codec_repo=cls._caller_resolved_codec_repo(config.codec_repo))

    def encode_reference(
        self,
        path: str | Path,
        *,
        max_seconds: float | None,
        normalize_db: float | None,
        ensure_max: bool,
    ) -> mx.array:
        with tempfile.NamedTemporaryFile(prefix="irodori-ref-", suffix=".npy", delete=False) as fh:
            latents_path = Path(fh.name)
        try:
            extra_args = [
                "--reference-wav",
                str(self._caller_absolute_path(path)),
                "--output-latents",
                str(latents_path),
            ]
            if max_seconds is not None:
                extra_args.extend(["--max-seconds", str(max_seconds)])
            if normalize_db is not None:
                extra_args.extend(["--normalize-db", str(normalize_db)])
            if ensure_max:
                extra_args.append("--ensure-max")
            _run_codec_worker(action="encode", config=self._worker_config(self.config), extra_args=extra_args)
            import numpy as np

            return mx.array(np.load(latents_path).astype("float32", copy=False))
        finally:
            latents_path.unlink(missing_ok=True)

    def decode_to_wav(self, latents: mx.array, output_path: str | Path, *, max_samples: int | None = None) -> Path:
        import numpy as np

        resolved_output_path = self._caller_absolute_path(output_path)
        with tempfile.NamedTemporaryFile(prefix="irodori-latents-", suffix=".npy", delete=False) as fh:
            latents_path = Path(fh.name)
        try:
            np.save(latents_path, _as_numpy(latents).astype("float32", copy=False))
            extra_args = [
                "--input-latents",
                str(latents_path),
                "--output-wav",
                str(resolved_output_path),
            ]
            if max_samples is not None:
                extra_args.extend(["--max-samples", str(max_samples)])
            _run_codec_worker(action="decode", config=self._worker_config(self.config), extra_args=extra_args)
            return resolved_output_path
        finally:
            latents_path.unlink(missing_ok=True)


def _load_audio_torch(path: str | Path):
    torch = _require_torch()
    try:
        import torchaudio

        return torchaudio.load(str(path))
    except Exception:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError("Loading audio requires torchaudio or soundfile.") from exc
        data, sample_rate = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, int(sample_rate)


def save_wav(path: str | Path, audio, sample_rate: int) -> Path:
    """Save mono audio with torchaudio/soundfile, falling back to stdlib PCM16."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio

        torchaudio.save(str(out), audio, int(sample_rate))
        return out
    except Exception:
        pass
    try:
        import soundfile as sf

        sf.write(str(out), audio.squeeze(0).numpy(), int(sample_rate))
        return out
    except Exception:
        pass

    import numpy as np

    samples = audio.squeeze(0).numpy().astype("float32")
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return out


class MLXDACVAERuntime:
    """End-to-end prototype: MLX RF-DiT latent generation + PyTorch DACVAE decode."""

    def __init__(
        self,
        *,
        config: MLXRuntimeConfig,
        model: TextToLatentRFDiT | None = None,
        bridge: PyTorchDACVAEBridge | None = None,
        tokenizer: PretrainedTextTokenizer | None = None,
        caption_tokenizer: PretrainedTextTokenizer | None = None,
    ) -> None:
        self.config = config
        self.model = model or load_mlx_model(config.model_config, config.weights_path)
        if bridge is None:
            if config.codec.runtime_mode == "persistent":
                bridge = PyTorchDACVAEBridge(config=config.codec)
            elif config.codec.runtime_mode == "subprocess":
                bridge = SubprocessDACVAEBridge(config=config.codec)
            else:
                raise ValueError(f"Unsupported codec runtime_mode={config.codec.runtime_mode!r}")
        self.bridge = bridge
        if self.bridge.latent_dim != int(config.model_config.latent_dim):
            raise ValueError(
                f"DACVAE latent_dim={self.bridge.latent_dim} does not match model latent_dim={config.model_config.latent_dim}."
            )
        text_repo = config.text_tokenizer_repo or config.model_config.text_tokenizer_repo
        self.tokenizer = tokenizer or PretrainedTextTokenizer.from_pretrained(
            text_repo,
            add_bos=bool(config.model_config.text_add_bos),
        )
        self.caption_tokenizer = caption_tokenizer
        if config.model_config.use_caption_condition and self.caption_tokenizer is None:
            caption_repo = config.caption_tokenizer_repo or config.model_config.caption_tokenizer_repo_resolved
            self.caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                caption_repo,
                add_bos=bool(config.model_config.caption_add_bos_resolved),
            )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        messages: list[str] = []
        timings_ms: dict[str, float] = {}
        total_started = time.perf_counter()
        manual_seconds = None if request.seconds is None else float(request.seconds)
        if manual_seconds is not None and manual_seconds <= 0:
            raise ValueError(f"seconds must be positive when provided, got {request.seconds!r}")
        if float(request.duration_scale) <= 0:
            raise ValueError(f"duration_scale must be positive, got {request.duration_scale!r}")
        if request.reference_wav is None and not request.no_reference and self.config.model_config.use_speaker_condition:
            raise ValueError("Specify reference_wav, or set no_reference=True for an unconditional speaker path.")

        started = time.perf_counter()
        text_ids, text_mask = self.tokenizer.encode(
            request.text,
            max_length=int(self.config.text_max_length),
        )
        caption_ids = caption_mask = None
        if self.config.model_config.use_caption_condition:
            caption_text = "" if request.caption is None else str(request.caption)
            caption_max = self.config.caption_max_length or self.config.text_max_length
            assert self.caption_tokenizer is not None
            caption_ids, caption_mask = self.caption_tokenizer.encode(caption_text, max_length=int(caption_max))
            if caption_text.strip() == "":
                caption_mask = mx.zeros_like(caption_mask)
        timings_ms["prepare_text_condition"] = (time.perf_counter() - started) * 1000.0

        ref_latent = ref_mask = None
        started = time.perf_counter()
        if self.config.model_config.use_speaker_condition:
            if request.no_reference:
                ref_len = max(1, int(self.config.model_config.speaker_patch_size))
                ref_latent = mx.zeros(
                    (1, ref_len, int(self.config.model_config.patched_latent_dim)),
                    dtype=mx.float32,
                )
                ref_mask = mx.zeros((1, ref_len), dtype=mx.bool_)
                messages.append("speaker reference disabled; using unconditional speaker mask")
            else:
                assert request.reference_wav is not None
                raw_ref = self.bridge.encode_reference(
                    request.reference_wav,
                    max_seconds=request.max_reference_seconds,
                    normalize_db=self.config.codec.normalize_db,
                    ensure_max=True,
                )
                ref_latent = patch_latents_drop_tail(raw_ref, int(self.config.model_config.latent_patch_size))
                ref_mask = mx.ones((1, ref_latent.shape[1]), dtype=mx.bool_)
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
            if ref_mask is not None:
                has_speaker = ref_mask.any(axis=1)
            duration_features = build_duration_features(
                [request.text],
                token_counts=text_mask.sum(axis=1),
                max_text_len=int(self.config.text_max_length),
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
            latent_steps = max(1, int(round(scaled_frames)))
            target_samples = int(latent_steps * self.bridge.hop_length)
            resolved_seconds = float(target_samples) / float(self.bridge.sample_rate)
            timings_ms["predict_duration"] = (time.perf_counter() - started) * 1000.0
            messages.append(
                "predicted duration active: "
                f"frames={pred_frames:.1f}, scale={float(request.duration_scale):.3f}, seconds={resolved_seconds:.3f}"
            )
        else:
            fallback_seconds = 5.0
            target_samples = int(fallback_seconds * float(self.bridge.sample_rate))
            latent_steps = max(1, (target_samples + self.bridge.hop_length - 1) // self.bridge.hop_length)
            resolved_seconds = float(target_samples) / float(self.bridge.sample_rate)
            messages.append(
                "duration predictor unavailable; falling back to fixed duration "
                f"{resolved_seconds:.3f}s"
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
            num_steps=int(request.num_steps),
            cfg_scale_text=float(request.cfg_scale_text),
            cfg_scale_caption=float(request.cfg_scale_caption),
            cfg_scale_speaker=float(request.cfg_scale_speaker),
            cfg_guidance_mode=request.cfg_guidance_mode,
            cfg_min_t=float(request.cfg_min_t),
            cfg_max_t=float(request.cfg_max_t),
            seed=int(request.seed),
            use_context_kv_cache=bool(request.use_context_kv_cache),
        )
        z = unpatch_latents(z_patched, int(self.config.model_config.latent_patch_size))[:, :latent_steps]
        mx.eval(z)
        timings_ms["sample_rf"] = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        output = self.bridge.decode_to_wav(z, request.output_wav, max_samples=target_samples)
        timings_ms["decode_dacvae"] = (time.perf_counter() - started) * 1000.0
        timings_ms["total_to_decode"] = (time.perf_counter() - total_started) * 1000.0
        return GenerationResult(
            output_wav=str(output),
            sample_rate=int(self.bridge.sample_rate),
            samples=target_samples,
            latent_steps=int(latent_steps),
            patched_steps=int(patched_steps),
            seed=int(request.seed),
            duration_mode=duration_mode,
            requested_seconds=manual_seconds,
            resolved_seconds=resolved_seconds,
            timings_ms=timings_ms,
            messages=tuple(messages),
        )

    def describe_boundaries(self) -> dict[str, object]:
        return {
            "mlx": {
                "model": "TextToLatentRFDiT",
                "weights_path": self.config.weights_path,
                "latent_layout": "(batch, time, latent_dim)",
            },
            "pytorch": {
                "codec_repo": self.config.codec.codec_repo,
                "codec_device": self.config.codec.codec_device,
                "codec_runtime_mode": self.config.codec.runtime_mode,
                "sample_rate": self.bridge.sample_rate,
                "hop_length": self.bridge.hop_length,
            },
            "conversion": "PyTorch tensor -> CPU NumPy -> MLX array, and reverse for DACVAE decode",
            "config": asdict(self.config),
        }


def iter_messages(result: GenerationResult) -> Iterable[str]:
    yield f"wrote: {result.output_wav}"
    yield f"sample_rate: {result.sample_rate}"
    yield f"samples: {result.samples}"
    yield f"latent_steps: {result.latent_steps}"
    yield f"patched_steps: {result.patched_steps}"
    yield f"seed: {result.seed}"
    yield f"duration_mode: {result.duration_mode}"
    if result.requested_seconds is not None:
        yield f"requested_seconds: {result.requested_seconds}"
    if result.resolved_seconds is not None:
        yield f"resolved_seconds: {result.resolved_seconds:.3f}"
    for name, value in sorted((result.timings_ms or {}).items()):
        yield f"[timing] {name}: {value:.3f} ms"
    for message in result.messages:
        yield message
