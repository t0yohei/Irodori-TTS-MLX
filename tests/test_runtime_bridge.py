from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.config import ModelConfig
    from irodori_mlx.encoders import EncodedConditions
    from irodori_mlx.runtime import (
        DACVAEBridgeConfig,
        GenerationRequest,
        MLXDACVAERuntime,
        MLXRuntimeConfig,
        load_model_config_json,
        mlx_to_torch_latents,
        torch_to_mlx_latents,
    )

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


class FakeTokenizer:
    def encode(self, text: str, *, max_length: int):
        del text
        ids = [1, 2] + [0] * max(0, int(max_length) - 2)
        mask = [True, True] + [False] * max(0, int(max_length) - 2)
        return mx.array([ids[:max_length]], dtype=mx.int32), mx.array([mask[:max_length]], dtype=mx.bool_)


class FakeBridge:
    sample_rate = 16000
    hop_length = 320
    latent_dim = 4

    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode_reference(self, path, *, max_seconds, normalize_db, ensure_max):
        self.encoded.append((str(path), max_seconds, normalize_db, ensure_max))
        return mx.ones((1, 7, self.latent_dim), dtype=mx.float32)

    def decode_to_wav(self, latents, output_path, *, max_samples=None):
        self.decoded.append((latents, output_path, max_samples))
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake wav")
        return path


class FakeModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.forward_shapes = []

    def encode_conditions(self, **kwargs):
        batch = int(kwargs["text_input_ids"].shape[0])
        text_mask = kwargs["text_mask"].astype(mx.bool_)
        speaker_state = speaker_mask = None
        if self.cfg.use_speaker_condition:
            speaker_state = mx.ones((batch, int(kwargs["ref_mask"].shape[1]), self.cfg.speaker_dim), dtype=mx.float32)
            speaker_mask = kwargs["ref_mask"].astype(mx.bool_)
        return EncodedConditions(
            text_state=mx.ones((batch, int(text_mask.shape[1]), self.cfg.text_dim), dtype=mx.float32),
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=None,
            caption_mask=None,
        )

    def build_context_kv_cache(self, *, text_state, speaker_state, caption_state=None):
        return [(text_state, speaker_state, caption_state)]

    def forward_with_encoded_conditions(self, *, x_t, t, text_state, text_mask, speaker_state, speaker_mask, caption_state=None, caption_mask=None, context_kv_cache=None):
        del t, text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask, context_kv_cache
        self.forward_shapes.append(tuple(x_t.shape))
        return mx.zeros_like(x_t)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        latent_patch_size=2,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=1.5,
        text_vocab_size=32,
        text_tokenizer_repo="example/text-tokenizer",
        text_dim=8,
        text_layers=1,
        text_heads=2,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        timestep_embed_dim=8,
        adaln_rank=2,
        norm_eps=1e-5,
        dropout=0.0,
    )


class RuntimeBridgeTests(unittest.TestCase):
    @require_mlx
    def test_runtime_encodes_reference_samples_mlx_latents_and_decodes_wav(self):
        cfg = tiny_config()
        bridge = FakeBridge()
        runtime = MLXDACVAERuntime(
            config=MLXRuntimeConfig(
                model_config=cfg,
                weights_path="unused.npz",
                text_max_length=4,
                codec=DACVAEBridgeConfig(normalize_db=None),
            ),
            model=FakeModel(cfg),
            bridge=bridge,
            tokenizer=FakeTokenizer(),
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.wav"
            result = runtime.generate(
                GenerationRequest(
                    text="hello",
                    reference_wav="ref.wav",
                    output_wav=str(out),
                    seconds=0.04,
                    num_steps=1,
                    cfg_scale_text=0.0,
                    cfg_scale_speaker=0.0,
                )
            )
        self.assertEqual(result.output_wav, str(out))
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.samples, 640)
        self.assertEqual(result.latent_steps, 2)
        self.assertEqual(result.patched_steps, 1)
        self.assertEqual(len(bridge.encoded), 1)
        self.assertEqual(len(bridge.decoded), 1)
        decoded_latents, _output_path, max_samples = bridge.decoded[0]
        self.assertEqual(decoded_latents.shape, (1, 2, 4))
        self.assertEqual(max_samples, 640)

    @require_mlx
    def test_no_reference_builds_unconditional_speaker_mask(self):
        cfg = tiny_config()
        bridge = FakeBridge()
        runtime = MLXDACVAERuntime(
            config=MLXRuntimeConfig(model_config=cfg, weights_path="unused.npz", text_max_length=3),
            model=FakeModel(cfg),
            bridge=bridge,
            tokenizer=FakeTokenizer(),
        )
        with tempfile.TemporaryDirectory() as td:
            result = runtime.generate(
                GenerationRequest(
                    text="hello",
                    output_wav=str(Path(td) / "out.wav"),
                    no_reference=True,
                    seconds=0.02,
                    num_steps=1,
                    cfg_scale_text=0.0,
                    cfg_scale_speaker=0.0,
                )
            )
        self.assertEqual(bridge.encoded, [])
        self.assertIn("speaker reference disabled", "\n".join(result.messages))

    @require_mlx
    def test_model_config_exposes_tokenizer_defaults(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.text_tokenizer_repo, "sbintuitions/sarashina2.2-0.5b")
        self.assertEqual(cfg.caption_tokenizer_repo_resolved, cfg.text_tokenizer_repo)

    @require_mlx
    def test_load_model_config_json_accepts_inline_object_or_path(self):
        inline = load_model_config_json('{"use_caption_condition": true, "caption_vocab_size": 32}')
        self.assertTrue(inline.use_caption_condition)
        self.assertEqual(inline.caption_vocab_size_resolved, 32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text('{"latent_dim": 8, "text_vocab_size": 64}', encoding="utf-8")
            from_path = load_model_config_json(path)
        self.assertEqual(from_path.latent_dim, 8)
        self.assertEqual(from_path.text_vocab_size, 64)

    @require_mlx
    def test_torch_mlx_latent_conversion_roundtrip_when_torch_is_available(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"torch is not available: {exc}")
        torch_latents = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        mlx_latents = torch_to_mlx_latents(torch_latents)
        roundtrip = mlx_to_torch_latents(mlx_latents)
        np.testing.assert_allclose(roundtrip.numpy(), torch_latents.numpy())


if __name__ == "__main__":
    unittest.main()
