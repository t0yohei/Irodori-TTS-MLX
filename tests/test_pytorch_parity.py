from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx import (
        ConditionEncoders as MlxConditionEncoders,
        LowRankAdaLN as MlxLowRankAdaLN,
        ModelConfig as MlxModelConfig,
        RMSNorm as MlxRMSNorm,
        ReferenceLatentEncoder as MlxReferenceLatentEncoder,
        SelfAttention as MlxSelfAttention,
        SwiGLU as MlxSwiGLU,
        TextEncoder as MlxTextEncoder,
        assign_named_weights,
        apply_rotary_emb as mlx_apply_rotary_emb,
        get_timestep_embedding as mlx_get_timestep_embedding,
        patch_sequence_with_mask as mlx_patch_sequence_with_mask,
        precompute_freqs_cis as mlx_precompute_freqs_cis,
    )

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only on machines without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc

try:
    import torch

    HAS_TORCH = True
except Exception as exc:  # pragma: no cover - exercised only on machines without torch.
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = exc


def default_upstream_path() -> Path:
    resolved = Path(__file__).resolve()
    repo_root = next(
        (
            parent
            for parent in resolved.parents
            if (parent / "irodori_mlx").is_dir() and (parent / "tests").is_dir()
        ),
        resolved.parent,
    )
    for anchor in (repo_root, *repo_root.parents):
        candidate = anchor / "_scratch" / "Irodori-TTS-upstream"
        if candidate.exists():
            return candidate
    return repo_root.parent / "_scratch" / "Irodori-TTS-upstream"


DEFAULT_UPSTREAM_PATH = default_upstream_path()
UPSTREAM_PATH = Path(os.environ.get("IRODORI_TTS_UPSTREAM_PATH", DEFAULT_UPSTREAM_PATH))


def require_parity_deps(test_func):
    reason = None
    if not HAS_MLX:
        reason = f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}"
    elif not HAS_TORCH:
        reason = f"PyTorch is not available: {globals().get('TORCH_IMPORT_ERROR')}"
    elif not (UPSTREAM_PATH / "irodori_tts" / "model.py").exists():
        reason = (
            "Upstream Irodori-TTS checkout not found. Set "
            "IRODORI_TTS_UPSTREAM_PATH=/path/to/Irodori-TTS."
        )
    return unittest.skipIf(reason is not None, reason)(test_func)


def import_upstream_model():
    upstream = str(UPSTREAM_PATH)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from irodori_tts import model as upstream_model  # type: ignore[import-not-found]

    return upstream_model


def to_np(value):
    return np.array(value)


def torch_to_np(value: "torch.Tensor") -> np.ndarray:
    return value.detach().cpu().numpy()


def deterministic_array(name: str, shape: tuple[int, ...], scale: float = 0.05) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64))
    if size == 0:
        return np.zeros(shape, dtype=np.float32)
    offset = (sum(ord(ch) for ch in name) % 17) - 8
    values = (np.arange(size, dtype=np.float32) + offset) * scale
    return values.reshape(shape).astype(np.float32)


def fill_torch_module(module: "torch.nn.Module", scale: float = 0.05) -> dict[str, np.ndarray]:
    state = module.state_dict()
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        if name.startswith("_freqs_cis_cache"):
            continue
        if tensor.dtype == torch.bool:
            array = np.ones(tuple(tensor.shape), dtype=np.bool_)
        else:
            array = deterministic_array(name, tuple(tensor.shape), scale=scale)
        state[name] = torch.as_tensor(array, dtype=tensor.dtype, device=tensor.device)
        arrays[name] = array
    module.load_state_dict(state, strict=False)
    module.eval()
    return arrays


def assign_mlx_arrays(module: object, arrays: dict[str, np.ndarray]) -> None:
    direct = {name: value for name, value in arrays.items() if "." not in name}
    nested = {name: value for name, value in arrays.items() if "." in name}
    for name, value in direct.items():
        current = getattr(module, name)
        if tuple(current.shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch for {name}: expected {current.shape}, got {value.shape}")
        setattr(module, name, mx.array(value))
    if nested:
        assign_named_weights(
            module,
            {name: mx.array(value) for name, value in nested.items()},
            required=tuple(nested.keys()),
        )


def assert_close_with_context(
    testcase: unittest.TestCase,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    diff = np.abs(actual - expected)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_index = tuple(int(i) for i in np.unravel_index(int(diff.argmax()), diff.shape)) if diff.size else ()
    testcase.assertTrue(
        np.allclose(actual, expected, rtol=rtol, atol=atol),
        msg=(
            f"{label} mismatch: shape={actual.shape} rtol={rtol} atol={atol} "
            f"max_abs={max_abs} max_index={max_index} "
            f"actual={actual[max_index] if diff.size else 'n/a'} "
            f"expected={expected[max_index] if diff.size else 'n/a'}"
        ),
    )


class PyTorchMlxParityTests(unittest.TestCase):
    RTOL = 2e-5
    ATOL = 2e-5

    @require_parity_deps
    def test_core_layer_outputs_match_upstream_pytorch(self):
        upstream = import_upstream_model()
        freqs_torch = upstream.precompute_freqs_cis(dim=4, end=3)
        freqs_mlx = mlx_precompute_freqs_cis(dim=4, end=3)
        assert_close_with_context(
            self,
            to_np(freqs_mlx),
            torch_to_np(freqs_torch),
            rtol=1e-6,
            atol=1e-6,
            label="precompute_freqs_cis",
        )

        x_np = (np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4) - 7.0) / 11.0
        rotary_torch = upstream.apply_rotary_emb(torch.tensor(x_np), freqs_torch)
        rotary_mlx = mlx_apply_rotary_emb(mx.array(x_np), freqs_mlx)
        assert_close_with_context(
            self,
            to_np(rotary_mlx),
            torch_to_np(rotary_torch),
            rtol=1e-6,
            atol=1e-6,
            label="apply_rotary_emb",
        )

        timestep_np = np.array([0.0, 0.25, 0.75], dtype=np.float32)
        timestep_torch = upstream.get_timestep_embedding(torch.tensor(timestep_np), dim=6)
        timestep_mlx = mlx_get_timestep_embedding(mx.array(timestep_np), dim=6)
        assert_close_with_context(
            self,
            to_np(timestep_mlx),
            torch_to_np(timestep_torch),
            rtol=5e-6,
            atol=5e-6,
            label="get_timestep_embedding",
        )

        rms_torch = upstream.RMSNorm(4, eps=1e-5)
        rms_arrays = fill_torch_module(rms_torch)
        rms_mlx = MlxRMSNorm(4, eps=1e-5)
        assign_mlx_arrays(rms_mlx, rms_arrays)
        x2_np = deterministic_array("rms_input", (2, 3, 4), scale=0.03)
        assert_close_with_context(
            self,
            to_np(rms_mlx(mx.array(x2_np))),
            torch_to_np(rms_torch(torch.tensor(x2_np))),
            rtol=self.RTOL,
            atol=self.ATOL,
            label="RMSNorm",
        )

        swiglu_torch = upstream.SwiGLU(dim=4, hidden_dim=6)
        swiglu_arrays = fill_torch_module(swiglu_torch, scale=0.025)
        swiglu_mlx = MlxSwiGLU(dim=4, hidden_dim=6)
        assign_mlx_arrays(swiglu_mlx, swiglu_arrays)
        assert_close_with_context(
            self,
            to_np(swiglu_mlx(mx.array(x2_np))),
            torch_to_np(swiglu_torch(torch.tensor(x2_np))),
            rtol=self.RTOL,
            atol=self.ATOL,
            label="SwiGLU",
        )

        adaln_torch = upstream.LowRankAdaLN(model_dim=4, rank=2, eps=1e-5)
        adaln_arrays = fill_torch_module(adaln_torch, scale=0.02)
        adaln_mlx = MlxLowRankAdaLN(model_dim=4, rank=2, eps=1e-5)
        assign_mlx_arrays(adaln_mlx, adaln_arrays)
        cond_np = deterministic_array("adaln_cond", (2, 3, 12), scale=0.02)
        mlx_x, mlx_gate = adaln_mlx(mx.array(x2_np), mx.array(cond_np))
        torch_x, torch_gate = adaln_torch(torch.tensor(x2_np), torch.tensor(cond_np))
        assert_close_with_context(
            self,
            to_np(mlx_x),
            torch_to_np(torch_x),
            rtol=self.RTOL,
            atol=self.ATOL,
            label="LowRankAdaLN activations",
        )
        assert_close_with_context(
            self,
            to_np(mlx_gate),
            torch_to_np(torch_gate),
            rtol=self.RTOL,
            atol=self.ATOL,
            label="LowRankAdaLN gate",
        )

        seq_np = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        mask_np = np.array([[True, True, False, True], [True, True, True, True]])
        mlx_seq, mlx_mask = mlx_patch_sequence_with_mask(mx.array(seq_np), mx.array(mask_np), patch_size=2)
        torch_seq, torch_mask = upstream.patch_sequence_with_mask(
            torch.tensor(seq_np), torch.tensor(mask_np), patch_size=2
        )
        np.testing.assert_array_equal(to_np(mlx_seq), torch_to_np(torch_seq))
        np.testing.assert_array_equal(to_np(mlx_mask), torch_to_np(torch_mask))

    @require_parity_deps
    def test_self_attention_output_matches_upstream_pytorch(self):
        upstream = import_upstream_model()
        torch_attention = upstream.SelfAttention(dim=8, heads=2, norm_eps=1e-5)
        arrays = fill_torch_module(torch_attention, scale=0.015)
        mlx_attention = MlxSelfAttention(dim=8, heads=2, norm_eps=1e-5)
        assign_mlx_arrays(mlx_attention, arrays)

        x_np = deterministic_array("self_attention_input", (2, 4, 8), scale=0.02)
        mask_np = np.array([[True, True, False, True], [True, False, True, True]])
        freqs_torch = upstream.precompute_freqs_cis(dim=4, end=4)
        freqs_mlx = mlx_precompute_freqs_cis(dim=4, end=4)
        assert_close_with_context(
            self,
            to_np(mlx_attention(mx.array(x_np), key_mask=mx.array(mask_np), freqs_cis=freqs_mlx)),
            torch_to_np(torch_attention(torch.tensor(x_np), key_mask=torch.tensor(mask_np), freqs_cis=freqs_torch)),
            rtol=3e-5,
            atol=3e-5,
            label="SelfAttention",
        )

    @require_parity_deps
    def test_text_encoder_output_matches_upstream_pytorch(self):
        upstream = import_upstream_model()
        kwargs = dict(vocab_size=17, dim=8, layers=1, heads=2, mlp_ratio=1.5, norm_eps=1e-5, dropout=0.0)
        torch_encoder = upstream.TextEncoder(**kwargs)
        arrays = fill_torch_module(torch_encoder, scale=0.012)
        mlx_encoder = MlxTextEncoder(**kwargs)
        assign_mlx_arrays(mlx_encoder, arrays)

        input_ids_np = np.array([[1, 2, 3, 4], [4, 0, 2, 1]], dtype=np.int64)
        mask_np = np.array([[True, True, False, True], [True, False, True, False]])
        assert_close_with_context(
            self,
            to_np(mlx_encoder(mx.array(input_ids_np), mx.array(mask_np))),
            torch_to_np(torch_encoder(torch.tensor(input_ids_np), torch.tensor(mask_np))),
            rtol=5e-5,
            atol=5e-5,
            label="TextEncoder",
        )

    @require_parity_deps
    def test_reference_encoder_and_condition_wrapper_match_upstream_pytorch(self):
        upstream = import_upstream_model()
        cfg_kwargs = dict(
            latent_dim=4,
            latent_patch_size=1,
            text_vocab_size=19,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            text_mlp_ratio=1.5,
            speaker_dim=8,
            speaker_layers=1,
            speaker_heads=2,
            speaker_mlp_ratio=1.5,
            speaker_patch_size=2,
            dropout=0.0,
            norm_eps=1e-5,
        )
        torch_cfg = upstream.ModelConfig(**cfg_kwargs)
        mlx_cfg = MlxModelConfig(**cfg_kwargs)

        torch_ref = upstream.ReferenceLatentEncoder(torch_cfg)
        ref_arrays = fill_torch_module(torch_ref, scale=0.01)
        mlx_ref = MlxReferenceLatentEncoder(mlx_cfg)
        assign_mlx_arrays(mlx_ref, ref_arrays)
        latent_np = deterministic_array("reference_latent", (2, 3, 8), scale=0.02)
        mask_np = np.array([[True, True, False], [True, True, True]])
        assert_close_with_context(
            self,
            to_np(mlx_ref(mx.array(latent_np), mx.array(mask_np))),
            torch_to_np(torch_ref(torch.tensor(latent_np), torch.tensor(mask_np))),
            rtol=5e-5,
            atol=5e-5,
            label="ReferenceLatentEncoder",
        )

        torch_rf_model = upstream.TextToLatentRFDiT(torch_cfg)
        full_model_arrays = fill_torch_module(torch_rf_model, scale=0.01)
        condition_arrays = {
            name: value
            for name, value in full_model_arrays.items()
            if name.startswith(("text_encoder.", "text_norm.", "speaker_encoder.", "speaker_norm."))
        }
        mlx_conditions = MlxConditionEncoders(mlx_cfg)
        assign_mlx_arrays(mlx_conditions, condition_arrays)

        text_ids_np = np.array([[1, 2, 3, 4], [5, 6, 0, 1]], dtype=np.int64)
        text_mask_np = np.array([[True, True, False, True], [True, False, True, True]])
        ref_latent_np = deterministic_array("condition_ref_latent", (2, 4, 4), scale=0.02)
        ref_mask_np = np.array([[True, True, True, False], [True, True, False, False]])
        mlx_encoded = mlx_conditions(
            text_input_ids=mx.array(text_ids_np),
            text_mask=mx.array(text_mask_np),
            ref_latent=mx.array(ref_latent_np),
            ref_mask=mx.array(ref_mask_np),
        )
        torch_encoded = torch_rf_model.encode_conditions(
            text_input_ids=torch.tensor(text_ids_np),
            text_mask=torch.tensor(text_mask_np),
            ref_latent=torch.tensor(ref_latent_np),
            ref_mask=torch.tensor(ref_mask_np),
        )
        torch_text_state, torch_text_mask, torch_speaker_state, torch_speaker_mask, _, _ = torch_encoded
        assert_close_with_context(
            self,
            to_np(mlx_encoded.text_state),
            torch_to_np(torch_text_state),
            rtol=5e-5,
            atol=5e-5,
            label="ConditionEncoders text_state",
        )
        np.testing.assert_array_equal(to_np(mlx_encoded.text_mask), torch_to_np(torch_text_mask))
        assert_close_with_context(
            self,
            to_np(mlx_encoded.speaker_state),
            torch_to_np(torch_speaker_state),
            rtol=5e-5,
            atol=5e-5,
            label="ConditionEncoders speaker_state",
        )
        np.testing.assert_array_equal(to_np(mlx_encoded.speaker_mask), torch_to_np(torch_speaker_mask))

    @require_parity_deps
    def test_voicedesign_caption_condition_wrapper_matches_upstream_pytorch(self):
        upstream = import_upstream_model()
        cfg_kwargs = dict(
            latent_dim=4,
            latent_patch_size=1,
            text_vocab_size=23,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            text_mlp_ratio=1.5,
            use_caption_condition=True,
            caption_vocab_size=29,
            caption_dim=8,
            caption_layers=1,
            caption_heads=2,
            caption_mlp_ratio=1.5,
            dropout=0.0,
            norm_eps=1e-5,
        )
        torch_cfg = upstream.ModelConfig(**cfg_kwargs)
        mlx_cfg = MlxModelConfig(**cfg_kwargs)

        torch_rf_model = upstream.TextToLatentRFDiT(torch_cfg)
        full_model_arrays = fill_torch_module(torch_rf_model, scale=0.01)
        condition_arrays = {
            name: value
            for name, value in full_model_arrays.items()
            if name.startswith(("text_encoder.", "text_norm.", "caption_encoder.", "caption_norm."))
        }
        mlx_conditions = MlxConditionEncoders(mlx_cfg)
        assign_mlx_arrays(mlx_conditions, condition_arrays)

        text_ids_np = np.array([[1, 2, 3, 4], [5, 6, 0, 1]], dtype=np.int64)
        text_mask_np = np.array([[True, True, False, True], [True, False, True, True]])
        caption_ids_np = np.array([[7, 8, 9], [10, 0, 12]], dtype=np.int64)
        caption_mask_np = np.array([[True, True, True], [True, False, True]])
        mlx_encoded = mlx_conditions(
            text_input_ids=mx.array(text_ids_np),
            text_mask=mx.array(text_mask_np),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=mx.array(caption_ids_np),
            caption_mask=mx.array(caption_mask_np),
        )
        torch_encoded = torch_rf_model.encode_conditions(
            text_input_ids=torch.tensor(text_ids_np),
            text_mask=torch.tensor(text_mask_np),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=torch.tensor(caption_ids_np),
            caption_mask=torch.tensor(caption_mask_np),
        )
        torch_text_state, torch_text_mask, _, _, torch_caption_state, torch_caption_mask = torch_encoded
        assert_close_with_context(
            self,
            to_np(mlx_encoded.text_state),
            torch_to_np(torch_text_state),
            rtol=5e-5,
            atol=5e-5,
            label="VoiceDesign ConditionEncoders text_state",
        )
        np.testing.assert_array_equal(to_np(mlx_encoded.text_mask), torch_to_np(torch_text_mask))
        assert_close_with_context(
            self,
            to_np(mlx_encoded.caption_state),
            torch_to_np(torch_caption_state),
            rtol=5e-5,
            atol=5e-5,
            label="VoiceDesign ConditionEncoders caption_state",
        )
        np.testing.assert_array_equal(to_np(mlx_encoded.caption_mask), torch_to_np(torch_caption_mask))


if __name__ == "__main__":
    unittest.main()
