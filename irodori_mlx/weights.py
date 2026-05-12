from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from .config import ModelConfig


@dataclass(frozen=True)
class WeightLoadReport:
    assigned: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def load_npz_weights(path: str | Path) -> dict[str, mx.array]:
    """Load a converted `.npz` archive into MLX arrays keyed by upstream names."""
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: mx.array(archive[name]) for name in archive.files}


def _resolve_parent(root: object, path: str) -> tuple[object, str] | None:
    parts = path.split(".")
    if len(parts) < 2:
        return None
    obj: object = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(obj, (list, tuple)):
            index = int(part)
            if index >= len(obj):
                return None
            obj = obj[index]
            continue
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj, parts[-1]


def assign_named_weights(
    root: object,
    weights: Mapping[str, mx.array],
    *,
    required: tuple[str, ...] = (),
    allowed_prefixes: tuple[str, ...] | None = None,
    strict: bool = True,
) -> WeightLoadReport:
    """Assign upstream-compatible named arrays to an MLX module tree.

    Numeric path components traverse Python lists, so keys like
    `text_encoder.blocks.0.attention.wq.weight` map onto `blocks[0]`.
    """
    assigned: list[str] = []
    unexpected: list[str] = []
    for name, value in weights.items():
        if allowed_prefixes is not None and not name.startswith(allowed_prefixes):
            unexpected.append(name)
            continue
        resolved = _resolve_parent(root, name)
        if resolved is None:
            unexpected.append(name)
            continue
        parent, attr = resolved
        if not hasattr(parent, attr):
            unexpected.append(name)
            continue
        current = getattr(parent, attr)
        if hasattr(current, "shape") and tuple(current.shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch for {name}: expected {current.shape}, got {value.shape}")
        setattr(parent, attr, value)
        assigned.append(name)

    missing = tuple(name for name in required if name not in assigned)
    if strict and (missing or unexpected):
        problems: list[str] = []
        if missing:
            problems.append("missing=" + ", ".join(missing[:10]))
        if unexpected:
            problems.append("unexpected=" + ", ".join(unexpected[:10]))
        raise ValueError("weight assignment failed: " + "; ".join(problems))
    return WeightLoadReport(
        assigned=tuple(sorted(assigned)),
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
    )


def encoder_required_keys(
    *,
    prefix: str,
    layers: int,
    dim: int,
    heads: int,
    mlp_ratio: float,
    has_embedding: bool,
    has_input_projection: bool,
) -> tuple[str, ...]:
    """Build the required upstream key set for a text-like encoder."""
    keys: list[str] = []
    if has_embedding:
        keys.append(f"{prefix}.text_embedding.weight")
    if has_input_projection:
        keys.extend([f"{prefix}.in_proj.weight", f"{prefix}.in_proj.bias"])
    hidden = int(dim * mlp_ratio)
    head_dim = dim // heads
    for i in range(layers):
        block = f"{prefix}.blocks.{i}"
        keys.extend(
            [
                f"{block}.attention_norm.weight",
                f"{block}.attention.gate.weight",
                f"{block}.attention.k_norm.weight",
                f"{block}.attention.q_norm.weight",
                f"{block}.attention.wk.weight",
                f"{block}.attention.wo.weight",
                f"{block}.attention.wq.weight",
                f"{block}.attention.wv.weight",
                f"{block}.mlp_norm.weight",
                f"{block}.mlp.w1.weight",
                f"{block}.mlp.w2.weight",
                f"{block}.mlp.w3.weight",
            ]
        )
        # Keep the observed dimensions close to the caller-visible config.
        if dim <= 0 or heads <= 0 or head_dim <= 0 or hidden <= 0:
            raise ValueError("invalid encoder dimensions")
    return tuple(keys)


def rf_dit_required_keys(cfg: ModelConfig) -> tuple[str, ...]:
    """Build required upstream key names for the full RF-DiT forward model."""
    keys: list[str] = []
    keys.extend(
        encoder_required_keys(
            prefix="text_encoder",
            layers=cfg.text_layers,
            dim=cfg.text_dim,
            heads=cfg.text_heads,
            mlp_ratio=cfg.text_mlp_ratio_resolved,
            has_embedding=True,
            has_input_projection=False,
        )
    )
    keys.append("text_norm.weight")
    if cfg.use_speaker_condition:
        keys.extend(
            encoder_required_keys(
                prefix="speaker_encoder",
                layers=cfg.speaker_layers,
                dim=cfg.speaker_dim,
                heads=cfg.speaker_heads,
                mlp_ratio=cfg.speaker_mlp_ratio_resolved,
                has_embedding=False,
                has_input_projection=True,
            )
        )
        keys.append("speaker_norm.weight")
    if cfg.use_caption_condition:
        keys.extend(
            encoder_required_keys(
                prefix="caption_encoder",
                layers=cfg.caption_layers_resolved,
                dim=cfg.caption_dim_resolved,
                heads=cfg.caption_heads_resolved,
                mlp_ratio=cfg.caption_mlp_ratio_resolved,
                has_embedding=True,
                has_input_projection=False,
            )
        )
        keys.append("caption_norm.weight")

    keys.extend(
        [
            "cond_module.0.weight",
            "cond_module.2.weight",
            "cond_module.4.weight",
            "in_proj.weight",
            "in_proj.bias",
            "out_norm.weight",
            "out_proj.weight",
            "out_proj.bias",
        ]
    )
    for i in range(cfg.num_layers):
        block = f"blocks.{i}"
        keys.extend(
            [
                f"{block}.attention.gate.weight",
                f"{block}.attention.k_norm.weight",
                f"{block}.attention.q_norm.weight",
                f"{block}.attention.wk.weight",
                f"{block}.attention.wo.weight",
                f"{block}.attention.wq.weight",
                f"{block}.attention.wv.weight",
                f"{block}.attention.wk_text.weight",
                f"{block}.attention.wv_text.weight",
                f"{block}.mlp.w1.weight",
                f"{block}.mlp.w2.weight",
                f"{block}.mlp.w3.weight",
            ]
        )
        if cfg.use_speaker_condition:
            keys.extend(
                [
                    f"{block}.attention.wk_speaker.weight",
                    f"{block}.attention.wv_speaker.weight",
                ]
            )
        if cfg.use_caption_condition:
            keys.extend(
                [
                    f"{block}.attention.wk_caption.weight",
                    f"{block}.attention.wv_caption.weight",
                ]
            )
        for adaln in ("attention_adaln", "mlp_adaln"):
            for branch in ("gate", "scale", "shift"):
                keys.extend(
                    [
                        f"{block}.{adaln}.{branch}_down.weight",
                        f"{block}.{adaln}.{branch}_up.weight",
                        f"{block}.{adaln}.{branch}_up.bias",
                    ]
                )
    return tuple(keys)
