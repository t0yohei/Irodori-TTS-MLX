from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from .encoders import EncodedConditions
from .model import TextToLatentRFDiT

CFGGUIDANCE_MODE = Literal["independent", "joint", "reduced"]


@dataclass(frozen=True)
class _ConditionBundle:
    text_state: mx.array
    text_mask: mx.array
    speaker_state: mx.array | None
    speaker_mask: mx.array | None
    caption_state: mx.array | None
    caption_mask: mx.array | None


def euler_timestep_schedule(num_steps: int, *, init_scale: float = 0.999, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Return the upstream RF Euler schedule from near-one to zero."""
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps!r}")
    return mx.linspace(1.0, 0.0, int(num_steps) + 1, dtype=dtype) * float(init_scale)


def _as_mode(cfg_guidance_mode: str) -> str:
    mode = str(cfg_guidance_mode).strip().lower()
    if mode not in {"independent", "joint", "reduced"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, reduced."
        )
    return mode


def _cat_optional(values: list[mx.array | None]) -> mx.array | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
    return mx.concatenate(present, axis=0)


def _bundle(encoded: EncodedConditions) -> _ConditionBundle:
    return _ConditionBundle(
        text_state=encoded.text_state,
        text_mask=encoded.text_mask,
        speaker_state=encoded.speaker_state,
        speaker_mask=encoded.speaker_mask,
        caption_state=encoded.caption_state,
        caption_mask=encoded.caption_mask,
    )


def _uncond_bundle(cond: _ConditionBundle, *, text: bool, speaker: bool, caption: bool) -> _ConditionBundle:
    return _ConditionBundle(
        text_state=mx.zeros_like(cond.text_state) if text else cond.text_state,
        text_mask=mx.zeros_like(cond.text_mask) if text else cond.text_mask,
        speaker_state=mx.zeros_like(cond.speaker_state) if speaker and cond.speaker_state is not None else cond.speaker_state,
        speaker_mask=mx.zeros_like(cond.speaker_mask) if speaker and cond.speaker_mask is not None else cond.speaker_mask,
        caption_state=mx.zeros_like(cond.caption_state) if caption and cond.caption_state is not None else cond.caption_state,
        caption_mask=mx.zeros_like(cond.caption_mask) if caption and cond.caption_mask is not None else cond.caption_mask,
    )


def _concat_bundles(bundles: list[_ConditionBundle]) -> _ConditionBundle:
    return _ConditionBundle(
        text_state=mx.concatenate([bundle.text_state for bundle in bundles], axis=0),
        text_mask=mx.concatenate([bundle.text_mask for bundle in bundles], axis=0),
        speaker_state=_cat_optional([bundle.speaker_state for bundle in bundles]),
        speaker_mask=_cat_optional([bundle.speaker_mask for bundle in bundles]),
        caption_state=_cat_optional([bundle.caption_state for bundle in bundles]),
        caption_mask=_cat_optional([bundle.caption_mask for bundle in bundles]),
    )


def _forward(
    model: TextToLatentRFDiT,
    *,
    x_t: mx.array,
    t: mx.array,
    bundle: _ConditionBundle,
    context_kv_cache: list[tuple[mx.array, ...]] | None = None,
) -> mx.array:
    return model.forward_with_encoded_conditions(
        x_t=x_t,
        t=t,
        text_state=bundle.text_state,
        text_mask=bundle.text_mask,
        speaker_state=bundle.speaker_state,
        speaker_mask=bundle.speaker_mask,
        caption_state=bundle.caption_state,
        caption_mask=bundle.caption_mask,
        context_kv_cache=context_kv_cache,
    )


def _build_cache(
    model: TextToLatentRFDiT,
    bundle: _ConditionBundle,
    *,
    use_context_kv_cache: bool,
) -> list[tuple[mx.array, ...]] | None:
    if not use_context_kv_cache:
        return None
    return model.build_context_kv_cache(
        text_state=bundle.text_state,
        speaker_state=bundle.speaker_state,
        caption_state=bundle.caption_state,
    )


def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    *,
    text_input_ids: mx.array,
    text_mask: mx.array,
    ref_latent: mx.array | None,
    ref_mask: mx.array | None,
    sequence_length: int,
    caption_input_ids: mx.array | None = None,
    caption_mask: mx.array | None = None,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    truncation_factor: float | None = None,
    use_context_kv_cache: bool = True,
) -> mx.array:
    """Sample patched RF latents with Euler steps and classifier-free guidance.

    The public argument names intentionally mirror upstream Irodori-TTS where
    practical. The returned tensor is in patched latent space with shape
    ``(batch, sequence_length, model.cfg.patched_latent_dim)``.
    """
    if int(sequence_length) <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length!r}")
    mode = _as_mode(cfg_guidance_mode)
    if not model.cfg.use_speaker_condition:
        cfg_scale_speaker = 0.0

    batch_size = int(text_input_ids.shape[0])
    latent_dim = int(model.cfg.patched_latent_dim)
    x_t = mx.random.normal(
        (batch_size, int(sequence_length), latent_dim),
        dtype=mx.float32,
        key=mx.random.key(int(seed)),
    )
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    encoded = model.encode_conditions(
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
    )
    cond = _bundle(encoded)

    has_text_cfg = float(cfg_scale_text) > 0.0
    has_speaker_cfg = float(cfg_scale_speaker) > 0.0 and cond.speaker_state is not None
    has_caption_cfg = (
        bool(model.cfg.use_caption_condition)
        and float(cfg_scale_caption) > 0.0
        and cond.caption_mask is not None
        and bool(mx.array(mx.any(cond.caption_mask)).item())
    )
    enabled: list[tuple[str, float]] = []
    if has_text_cfg:
        enabled.append(("text", float(cfg_scale_text)))
    if has_speaker_cfg:
        enabled.append(("speaker", float(cfg_scale_speaker)))
    if has_caption_cfg:
        enabled.append(("caption", float(cfg_scale_caption)))

    cond_cache = _build_cache(model, cond, use_context_kv_cache=use_context_kv_cache)
    independent_names = ["cond"]
    independent_bundle = cond
    independent_cache = None
    joint_uncond = None
    joint_cache = None

    if mode == "independent" and enabled:
        bundles = [cond]
        for name, _scale in enabled:
            independent_names.append(name)
            bundles.append(
                _uncond_bundle(
                    cond,
                    text=name == "text",
                    speaker=name == "speaker",
                    caption=name == "caption",
                )
            )
        independent_bundle = _concat_bundles(bundles)
        independent_cache = _build_cache(model, independent_bundle, use_context_kv_cache=use_context_kv_cache)
    elif mode in {"joint", "reduced"} and enabled:
        if mode == "joint" and len(enabled) > 1:
            scales = [scale for _name, scale in enabled]
            if max(scales) - min(scales) > 1e-6:
                raise ValueError(
                    "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                    "set matching text/speaker/caption scales."
                )
        joint_uncond = _uncond_bundle(cond, text=has_text_cfg, speaker=has_speaker_cfg, caption=has_caption_cfg)
        joint_cache = _build_cache(model, joint_uncond, use_context_kv_cache=use_context_kv_cache)

    schedule = euler_timestep_schedule(int(num_steps), dtype=x_t.dtype)
    for i in range(int(num_steps)):
        t = schedule[i]
        t_next = schedule[i + 1]
        tt = mx.full((batch_size,), t, dtype=x_t.dtype)
        use_cfg = bool(enabled) and (float(cfg_min_t) <= float(t.item()) <= float(cfg_max_t))

        if use_cfg and mode == "independent":
            cfg_mult = len(independent_names)
            x_t_cfg = mx.concatenate([x_t] * cfg_mult, axis=0)
            tt_cfg = mx.concatenate([tt] * cfg_mult, axis=0)
            v_out = _forward(
                model,
                x_t=x_t_cfg,
                t=tt_cfg,
                bundle=independent_bundle,
                context_kv_cache=independent_cache,
            )
            chunks = mx.split(v_out, cfg_mult, axis=0)
            v = chunks[0]
            scales = {name: scale for name, scale in enabled}
            if len(independent_names) - 1 != len(chunks) - 1:
                raise RuntimeError("CFG chunk/name mismatch")
            for name, chunk in zip(independent_names[1:], chunks[1:]):
                v = v + scales[name] * (chunks[0] - chunk)
        elif use_cfg and mode in {"joint", "reduced"}:
            v_cond = _forward(model, x_t=x_t, t=tt, bundle=cond, context_kv_cache=cond_cache)
            if joint_uncond is None:
                v = v_cond
            else:
                joint_scale = enabled[0][1] if mode == "joint" else max(scale for _name, scale in enabled)
                v_uncond = _forward(model, x_t=x_t, t=tt, bundle=joint_uncond, context_kv_cache=joint_cache)
                v = v_cond + joint_scale * (v_cond - v_uncond)
        else:
            v = _forward(model, x_t=x_t, t=tt, bundle=cond, context_kv_cache=cond_cache)

        x_t = x_t + v * (t_next - t)

    return x_t
