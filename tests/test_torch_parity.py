from __future__ import annotations

from dataclasses import asdict
import os
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import torch

    from irodori_mlx.config import ModelConfig as MLXModelConfig
    from irodori_mlx.model import TextToLatentRFDiT
    from irodori_mlx.weights import assign_named_weights, rf_dit_required_keys

    HAS_DEPS = True
except Exception as exc:  # pragma: no cover - dependency-specific skip path.
    HAS_DEPS = False
    IMPORT_ERROR = exc


def default_upstream_path() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if parent.name == "repos":
            return parent / "_scratch" / "Irodori-TTS-upstream"
    return resolved.parents[3] / "_scratch" / "Irodori-TTS-upstream"


DEFAULT_UPSTREAM = default_upstream_path()
UPSTREAM_PATH = Path(os.environ.get("IRODORI_TTS_UPSTREAM_PATH", DEFAULT_UPSTREAM))


def require_deps(test_func):
    return unittest.skipUnless(HAS_DEPS, f"PyTorch/MLX parity dependencies unavailable: {globals().get('IMPORT_ERROR')}")(test_func)


def load_upstream_modules():
    model_py = UPSTREAM_PATH / "irodori_tts" / "model.py"
    if not model_py.exists():
        raise unittest.SkipTest(
            "upstream Irodori-TTS checkout not found; set IRODORI_TTS_UPSTREAM_PATH to enable parity tests"
        )
    sys.path.insert(0, str(UPSTREAM_PATH))
    try:
        import irodori_tts.config as upstream_config
        import irodori_tts.model as upstream_model
        return upstream_config, upstream_model
    finally:
        try:
            sys.path.remove(str(UPSTREAM_PATH))
        except ValueError:
            pass


def tiny_mlx_config() -> MLXModelConfig:
    return MLXModelConfig(
        latent_dim=4,
        latent_patch_size=1,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=1.5,
        text_mlp_ratio=1.5,
        speaker_mlp_ratio=1.5,
        text_vocab_size=32,
        text_dim=8,
        text_layers=1,
        text_heads=2,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        speaker_patch_size=1,
        timestep_embed_dim=8,
        adaln_rank=2,
        norm_eps=1e-5,
        dropout=0.0,
    )


def copy_deterministic_weights(torch_model, mlx_model, cfg: MLXModelConfig) -> None:
    rng = np.random.default_rng(1234)
    mlx_weights = {}
    state = torch_model.state_dict()
    with torch.no_grad():
        for name in rf_dit_required_keys(cfg):
            tensor = state[name]
            if tensor.dtype == torch.bool:
                values = np.zeros(tuple(tensor.shape), dtype=np.bool_)
            elif tensor.ndim == 0:
                values = np.array(0.01, dtype=np.float32)
            elif ".text_embedding.weight" in name:
                values = rng.normal(0.0, 0.03, tuple(tensor.shape)).astype(np.float32)
            elif name.endswith(".bias"):
                values = rng.normal(0.0, 0.01, tuple(tensor.shape)).astype(np.float32)
            else:
                values = rng.normal(0.0, 0.02, tuple(tensor.shape)).astype(np.float32)
            tensor.copy_(torch.from_numpy(values).to(dtype=tensor.dtype))
            mlx_weights[name] = mx.array(values)
    assign_named_weights(mlx_model, mlx_weights, required=rf_dit_required_keys(cfg), strict=True)


class TorchParityTests(unittest.TestCase):
    @require_deps
    def test_tiny_rf_dit_forward_matches_upstream_pytorch(self):
        upstream_config, upstream_model = load_upstream_modules()
        cfg = tiny_mlx_config()
        torch_cfg = upstream_config.ModelConfig(**asdict(cfg))
        torch_model = upstream_model.TextToLatentRFDiT(torch_cfg).eval()
        mlx_model = TextToLatentRFDiT(cfg)
        copy_deterministic_weights(torch_model, mlx_model, cfg)

        x = np.linspace(-0.2, 0.3, 1 * 2 * 4, dtype=np.float32).reshape(1, 2, 4)
        ref = np.linspace(0.1, 0.4, 1 * 2 * 4, dtype=np.float32).reshape(1, 2, 4)
        text_ids = np.array([[1, 2, 0]], dtype=np.int64)
        text_mask = np.array([[True, True, False]])
        ref_mask = np.array([[True, True]])
        t = np.array([0.375], dtype=np.float32)

        with torch.no_grad():
            torch_out = torch_model(
                x_t=torch.from_numpy(x),
                t=torch.from_numpy(t),
                text_input_ids=torch.from_numpy(text_ids),
                text_mask=torch.from_numpy(text_mask),
                ref_latent=torch.from_numpy(ref),
                ref_mask=torch.from_numpy(ref_mask),
            ).detach().cpu().numpy()
        mlx_out = np.array(
            mlx_model(
                x_t=mx.array(x),
                t=mx.array(t),
                text_input_ids=mx.array(text_ids.astype(np.int32)),
                text_mask=mx.array(text_mask),
                ref_latent=mx.array(ref),
                ref_mask=mx.array(ref_mask),
            )
        )
        np.testing.assert_allclose(mlx_out, torch_out, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
