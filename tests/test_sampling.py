from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.config import ModelConfig
    from irodori_mlx.encoders import EncodedConditions
    from irodori_mlx.model import TextToLatentRFDiT
    from irodori_mlx.sampling import euler_timestep_schedule, sample_euler_rf_cfg

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only on machines without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


def to_np(value):
    return np.array(value)


class FakeSamplerModel:
    def __init__(self, *, caption: bool = True, speaker: bool = True):
        self.cfg = SimpleNamespace(
            patched_latent_dim=2,
            use_caption_condition=caption,
            use_speaker_condition=speaker,
        )
        self.calls = []
        self.cache_builds = 0

    def encode_conditions(self, **kwargs):
        batch = int(kwargs["text_input_ids"].shape[0])
        text_mask = kwargs["text_mask"].astype(mx.bool_)
        speaker_state = speaker_mask = None
        if self.cfg.use_speaker_condition:
            speaker_len = int(kwargs["ref_mask"].shape[1])
            speaker_state = mx.ones((batch, speaker_len, 3), dtype=mx.float32)
            speaker_mask = kwargs["ref_mask"].astype(mx.bool_)
        caption_state = caption_mask = None
        if self.cfg.use_caption_condition:
            caption_len = int(kwargs["caption_mask"].shape[1])
            caption_state = mx.ones((batch, caption_len, 3), dtype=mx.float32)
            caption_mask = kwargs["caption_mask"].astype(mx.bool_)
        return EncodedConditions(
            text_state=mx.ones((batch, int(text_mask.shape[1]), 3), dtype=mx.float32),
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
        )

    def build_context_kv_cache(self, *, text_state, speaker_state, caption_state=None):
        self.cache_builds += 1
        return [(text_state, speaker_state, caption_state)]

    def forward_with_encoded_conditions(
        self,
        *,
        x_t,
        t,
        text_state,
        text_mask,
        speaker_state,
        speaker_mask,
        caption_state=None,
        caption_mask=None,
        context_kv_cache=None,
    ):
        self.calls.append(
            {
                "batch": int(x_t.shape[0]),
                "t": to_np(t),
                "cached": context_kv_cache is not None,
            }
        )
        text_on = mx.any(text_mask, axis=1).astype(x_t.dtype)
        speaker_on = mx.zeros_like(text_on)
        if speaker_mask is not None:
            speaker_on = mx.any(speaker_mask, axis=1).astype(x_t.dtype)
        caption_on = mx.zeros_like(text_on)
        if caption_mask is not None:
            caption_on = mx.any(caption_mask, axis=1).astype(x_t.dtype)
        scalar = text_on + 10.0 * speaker_on + 100.0 * caption_on
        return mx.broadcast_to(scalar[:, None, None], x_t.shape)


class CaptionContentSamplerModel(FakeSamplerModel):
    def __init__(self):
        super().__init__(caption=True, speaker=False)

    def encode_conditions(self, **kwargs):
        encoded = super().encode_conditions(**kwargs)
        caption_ids = kwargs["caption_input_ids"].astype(mx.float32)
        caption_mask = kwargs["caption_mask"].astype(mx.bool_)
        caption_state = caption_ids[:, :, None] * mx.ones((1, 1, 3), dtype=mx.float32)
        caption_state = caption_state * caption_mask[:, :, None].astype(mx.float32)
        return EncodedConditions(
            text_state=encoded.text_state,
            text_mask=encoded.text_mask,
            speaker_state=None,
            speaker_mask=None,
            caption_state=caption_state,
            caption_mask=caption_mask,
        )

    def forward_with_encoded_conditions(
        self,
        *,
        x_t,
        t,
        text_state,
        text_mask,
        speaker_state,
        speaker_mask,
        caption_state=None,
        caption_mask=None,
        context_kv_cache=None,
    ):
        self.calls.append(
            {
                "batch": int(x_t.shape[0]),
                "t": to_np(t),
                "cached": context_kv_cache is not None,
            }
        )
        caption_score = mx.sum(caption_state, axis=(1, 2)) / 100.0
        return mx.broadcast_to(caption_score[:, None, None].astype(x_t.dtype), x_t.shape)


class SamplingTests(unittest.TestCase):
    def tiny_config(self) -> ModelConfig:
        return ModelConfig(
            latent_dim=4,
            latent_patch_size=1,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=1.5,
            text_vocab_size=32,
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

    @require_mlx
    def test_euler_schedule_matches_upstream_shape_and_scale(self):
        schedule = euler_timestep_schedule(4)
        np.testing.assert_allclose(to_np(schedule), np.array([0.999, 0.74925, 0.4995, 0.24975, 0.0], np.float32))

    @require_mlx
    def test_fixed_seed_noise_is_deterministic_without_cfg(self):
        model = FakeSamplerModel(caption=False)
        kwargs = dict(
            model=model,
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=mx.ones((1, 2, 2), dtype=mx.float32),
            ref_mask=mx.array([[True, True]]),
            sequence_length=3,
            num_steps=1,
            seed=123,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=True,
        )
        out1 = sample_euler_rf_cfg(**kwargs)
        out2 = sample_euler_rf_cfg(**kwargs)
        self.assertEqual(out1.shape, (1, 3, 2))
        np.testing.assert_allclose(to_np(out1), to_np(out2), rtol=0, atol=0)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call["cached"] for call in model.calls))

    @require_mlx
    def test_independent_cfg_batches_cond_and_each_uncond_path(self):
        model = FakeSamplerModel(caption=True, speaker=True)
        seed = 7
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=mx.ones((1, 2, 2), dtype=mx.float32),
            ref_mask=mx.array([[True, True]]),
            caption_input_ids=mx.array([[3, 4]], dtype=mx.int32),
            caption_mask=mx.array([[True, True]]),
            sequence_length=1,
            num_steps=1,
            cfg_scale_text=2.0,
            cfg_scale_speaker=3.0,
            cfg_scale_caption=4.0,
            cfg_guidance_mode="independent",
            seed=seed,
            use_context_kv_cache=False,
        )
        self.assertEqual([call["batch"] for call in model.calls], [4])
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(seed))
        # cond=111, text-uncond=110, speaker-uncond=101, caption-uncond=11
        guided_velocity = 111 + 2 * (111 - 110) + 3 * (111 - 101) + 4 * (111 - 11)
        expected = init + guided_velocity * (0.0 - 0.999)
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_joint_cfg_uses_two_forwards_and_requires_equal_scales(self):
        model = FakeSamplerModel(caption=True, speaker=True)
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=mx.ones((1, 2, 2), dtype=mx.float32),
            ref_mask=mx.array([[True, True]]),
            caption_input_ids=mx.array([[3]], dtype=mx.int32),
            caption_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=1,
            cfg_scale_text=2.0,
            cfg_scale_speaker=2.0,
            cfg_scale_caption=2.0,
            cfg_guidance_mode="joint",
            seed=5,
            use_context_kv_cache=False,
        )
        self.assertEqual([call["batch"] for call in model.calls], [1, 1])
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(5))
        expected = init + (111 + 2 * (111 - 0)) * (0.0 - 0.999)
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

        with self.assertRaisesRegex(ValueError, "equal enabled guidance scales"):
            sample_euler_rf_cfg(
                FakeSamplerModel(caption=True, speaker=True),
                text_input_ids=mx.array([[1]], dtype=mx.int32),
                text_mask=mx.array([[True]]),
                ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
                ref_mask=mx.array([[True]]),
                caption_input_ids=mx.array([[2]], dtype=mx.int32),
                caption_mask=mx.array([[True]]),
                sequence_length=1,
                num_steps=1,
                cfg_scale_text=1.0,
                cfg_scale_speaker=2.0,
                cfg_scale_caption=1.0,
                cfg_guidance_mode="joint",
            )

    @require_mlx
    def test_reduced_cfg_uses_max_enabled_scale_without_equal_scale_requirement(self):
        model = FakeSamplerModel(caption=True, speaker=True)
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            caption_input_ids=mx.array([[2]], dtype=mx.int32),
            caption_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=1,
            cfg_scale_text=1.0,
            cfg_scale_speaker=2.0,
            cfg_scale_caption=3.0,
            cfg_guidance_mode="reduced",
            seed=9,
            use_context_kv_cache=False,
        )
        self.assertEqual([call["batch"] for call in model.calls], [1, 1])
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(9))
        expected = init + (111 + 3 * (111 - 0)) * (0.0 - 0.999)
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_fixed_seed_caption_content_changes_mlx_sample(self):
        common = dict(
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_speaker=0.0,
            cfg_scale_caption=0.0,
            seed=17,
            use_context_kv_cache=True,
        )
        calm = sample_euler_rf_cfg(
            CaptionContentSamplerModel(),
            caption_input_ids=mx.array([[3, 4, 5]], dtype=mx.int32),
            caption_mask=mx.array([[True, True, True]]),
            **common,
        )
        energetic = sample_euler_rf_cfg(
            CaptionContentSamplerModel(),
            caption_input_ids=mx.array([[30, 40, 50]], dtype=mx.int32),
            caption_mask=mx.array([[True, True, True]]),
            **common,
        )

        self.assertFalse(np.allclose(to_np(calm), to_np(energetic), rtol=0, atol=0))

    @require_mlx
    def test_caption_cfg_cache_on_and_off_are_equivalent_for_same_caption(self):
        common = dict(
            text_input_ids=mx.array([[1, 2]], dtype=mx.int32),
            text_mask=mx.array([[True, True]]),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=mx.array([[9, 8, 7]], dtype=mx.int32),
            caption_mask=mx.array([[True, True, True]]),
            sequence_length=2,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_speaker=0.0,
            cfg_scale_caption=2.0,
            cfg_guidance_mode="independent",
            seed=23,
        )
        cached_model = CaptionContentSamplerModel()
        uncached_model = CaptionContentSamplerModel()
        cached = sample_euler_rf_cfg(cached_model, use_context_kv_cache=True, **common)
        uncached = sample_euler_rf_cfg(uncached_model, use_context_kv_cache=False, **common)

        np.testing.assert_allclose(to_np(cached), to_np(uncached), rtol=1e-6, atol=1e-6)
        self.assertGreater(cached_model.cache_builds, 0)
        self.assertEqual(uncached_model.cache_builds, 0)

    @require_mlx
    def test_sampler_runs_against_tiny_mlx_rf_dit_model(self):
        model = TextToLatentRFDiT(self.tiny_config())
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1, 2, 0]], dtype=mx.int32),
            text_mask=mx.array([[True, True, False]]),
            ref_latent=mx.ones((1, 2, 4), dtype=mx.float32),
            ref_mask=mx.array([[True, True]]),
            sequence_length=2,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            seed=11,
        )
        self.assertEqual(out.shape, (1, 2, 4))
        self.assertEqual(out.dtype, mx.float32)
        self.assertTrue(np.isfinite(to_np(out)).all())

    @require_mlx
    def test_speaker_state_override_replaces_encoded_reference_condition(self):
        model = FakeSamplerModel(caption=False, speaker=True)
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.zeros((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[False]]),
            sequence_length=1,
            speaker_state=mx.ones((1, 2, 3), dtype=mx.float32),
            speaker_mask=mx.array([[True, True]]),
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_speaker=0.0,
            seed=5,
            use_context_kv_cache=False,
        )
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(5))
        expected = init + 11.0 * (0.0 - 0.999)
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_speaker_state_override_can_omit_reference_latents(self):
        model = FakeSamplerModel(caption=False, speaker=True)
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=None,
            ref_mask=None,
            sequence_length=1,
            speaker_state=mx.ones((1, 2, 3), dtype=mx.float32),
            speaker_mask=mx.array([[True, True]]),
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_speaker=0.0,
            seed=5,
            use_context_kv_cache=False,
        )
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(5))
        expected = init + 11.0 * (0.0 - 0.999)
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_invalid_guidance_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported cfg_guidance_mode"):
            sample_euler_rf_cfg(
                FakeSamplerModel(caption=False),
                text_input_ids=mx.array([[1]], dtype=mx.int32),
                text_mask=mx.array([[True]]),
                ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
                ref_mask=mx.array([[True]]),
                sequence_length=1,
                cfg_guidance_mode="alternating",
            )


if __name__ == "__main__":
    unittest.main()
