from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import mlx.core as mx

    from irodori_mlx.config import ModelConfig
    from irodori_mlx.encoders import EncodedConditions
    from irodori_mlx.model import TextToLatentRFDiT
    from irodori_mlx.sampling import (
        euler_timestep_schedule,
        sample_euler_rf_cfg,
        scale_speaker_kv_cache,
        temporal_score_rescale,
    )

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
        if self.cfg.use_speaker_condition:
            return [
                (
                    text_state,
                    mx.ones_like(text_state),
                    speaker_state,
                    mx.ones_like(speaker_state),
                    caption_state,
                )
            ]
        return [(text_state, mx.ones_like(text_state), caption_state)]

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


class CacheSpeakerScaleSamplerModel(FakeSamplerModel):
    def __init__(self):
        super().__init__(caption=False, speaker=True)

    def build_context_kv_cache(self, *, text_state, speaker_state, caption_state=None):
        self.cache_builds += 1
        batch = int(text_state.shape[0])
        return [
            (
                mx.zeros((batch, 1, 1, 1), dtype=mx.float32),
                mx.zeros((batch, 1, 1, 1), dtype=mx.float32),
                mx.ones((batch, 1, 1, 1), dtype=mx.float32),
                mx.ones((batch, 1, 1, 1), dtype=mx.float32),
            )
        ]

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
        speaker_k = context_kv_cache[0][2].reshape((x_t.shape[0], -1)).mean(axis=1).astype(x_t.dtype)
        scalar = text_on + 10.0 * speaker_k
        return mx.broadcast_to(scalar[:, None, None], x_t.shape)


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
    def test_sway_schedule_matches_upstream_formula(self):
        schedule = euler_timestep_schedule(4, mode="sway", sway_coeff=-1.0)
        u = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        u = u + -1.0 * (np.cos(0.5 * np.pi * u) + u - 1.0)
        u = np.clip(u, 0.0, 1.0)
        expected = (1.0 - u) * 0.999

        np.testing.assert_allclose(to_np(schedule), expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_invalid_schedule_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported t_schedule_mode"):
            euler_timestep_schedule(4, mode="zigzag")

    @require_mlx
    def test_temporal_score_rescale_matches_upstream_formula(self):
        v = mx.array([[[2.0, -1.0]]], dtype=mx.float32)
        x_t = mx.array([[[0.5, 1.5]]], dtype=mx.float32)

        out = temporal_score_rescale(v_pred=v, x_t=x_t, t=0.5, rescale_k=2.0, rescale_sigma=0.5)

        one_minus_t = 0.5
        snr = (one_minus_t * one_minus_t) / (0.5 * 0.5)
        sigma_sq = 0.5 * 0.5
        ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / 2.0 + 1.0)
        expected = (ratio * (one_minus_t * to_np(v) + to_np(x_t)) - to_np(x_t)) / one_minus_t
        np.testing.assert_allclose(to_np(out), expected, rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_scale_speaker_kv_cache_scales_only_requested_layers(self):
        cache = [
            (
                mx.ones((1, 1, 1), dtype=mx.float32),
                mx.ones((1, 1, 1), dtype=mx.float32) * 2,
                mx.ones((1, 1, 1), dtype=mx.float32) * 3,
                mx.ones((1, 1, 1), dtype=mx.float32) * 4,
            ),
            (
                mx.ones((1, 1, 1), dtype=mx.float32) * 5,
                mx.ones((1, 1, 1), dtype=mx.float32) * 6,
                mx.ones((1, 1, 1), dtype=mx.float32) * 7,
                mx.ones((1, 1, 1), dtype=mx.float32) * 8,
            ),
        ]

        scaled = scale_speaker_kv_cache(cache, scale=10.0, max_layers=1)

        self.assertEqual(float(scaled[0][0].item()), 1.0)
        self.assertEqual(float(scaled[0][1].item()), 2.0)
        self.assertEqual(float(scaled[0][2].item()), 30.0)
        self.assertEqual(float(scaled[0][3].item()), 40.0)
        self.assertEqual(float(scaled[1][2].item()), 7.0)
        self.assertEqual(float(cache[0][2].item()), 3.0)

    @require_mlx
    def test_non_finite_sway_coeff_raises(self):
        with self.assertRaisesRegex(ValueError, "sway_coeff must be finite"):
            euler_timestep_schedule(4, mode="sway", sway_coeff=float("inf"))

    @require_mlx
    def test_non_decreasing_sway_schedule_raises(self):
        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            euler_timestep_schedule(4, mode="sway", sway_coeff=10.0)

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
    def test_alternating_cfg_rotates_enabled_condition_unconds(self):
        model = FakeSamplerModel(caption=True, speaker=True)
        seed = 13
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            caption_input_ids=mx.array([[2]], dtype=mx.int32),
            caption_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=3,
            cfg_scale_text=2.0,
            cfg_scale_speaker=3.0,
            cfg_scale_caption=4.0,
            cfg_guidance_mode="alternating",
            cfg_min_t=0.0,
            seed=seed,
            use_context_kv_cache=True,
        )

        self.assertEqual([call["batch"] for call in model.calls], [1, 1, 1, 1, 1, 1])
        self.assertTrue(all(call["cached"] for call in model.calls))
        self.assertEqual(model.cache_builds, 4)
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(seed))
        schedule = euler_timestep_schedule(3)
        velocities = [
            111 + 2 * (111 - 110),
            111 + 3 * (111 - 101),
            111 + 4 * (111 - 11),
        ]
        expected = init
        for i, velocity in enumerate(velocities):
            expected = expected + velocity * (schedule[i + 1] - schedule[i])
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_alternating_cfg_skips_disabled_conditions(self):
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
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_speaker=3.0,
            cfg_scale_caption=4.0,
            cfg_guidance_mode="alternating",
            cfg_min_t=0.0,
            seed=19,
            use_context_kv_cache=False,
        )

        self.assertEqual([call["batch"] for call in model.calls], [1, 1, 1, 1])
        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(19))
        schedule = euler_timestep_schedule(2)
        expected = init
        for i, velocity in enumerate([111 + 3 * (111 - 101), 111 + 4 * (111 - 11)]):
            expected = expected + velocity * (schedule[i + 1] - schedule[i])
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_alternating_cfg_rotates_by_absolute_diffusion_step(self):
        model = FakeSamplerModel(caption=True, speaker=True)
        seed = 29
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            caption_input_ids=mx.array([[2]], dtype=mx.int32),
            caption_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=3,
            cfg_scale_text=2.0,
            cfg_scale_speaker=3.0,
            cfg_scale_caption=4.0,
            cfg_guidance_mode="alternating",
            cfg_min_t=0.0,
            cfg_max_t=0.7,
            seed=seed,
            use_context_kv_cache=False,
        )

        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(seed))
        schedule = euler_timestep_schedule(3)
        velocities = [111, 111 + 3 * (111 - 101), 111 + 4 * (111 - 11)]
        expected = init
        for i, velocity in enumerate(velocities):
            expected = expected + velocity * (schedule[i + 1] - schedule[i])
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
                cfg_guidance_mode="unsupported",
            )

    @require_mlx
    def test_sampler_accepts_sway_schedule(self):
        model = FakeSamplerModel(caption=False)
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            seed=3,
            t_schedule_mode="sway",
            sway_coeff=-1.0,
            use_context_kv_cache=False,
        )

        self.assertEqual(out.shape, (1, 1, 2))
        self.assertEqual([call["batch"] for call in model.calls], [1, 1])

    @require_mlx
    def test_sampler_applies_temporal_score_rescale(self):
        model = FakeSamplerModel(caption=False, speaker=False)
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=None,
            ref_mask=None,
            sequence_length=1,
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            seed=7,
            use_context_kv_cache=False,
            rescale_k=2.0,
            rescale_sigma=0.5,
        )

        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(7))
        v = mx.ones_like(init)
        schedule = euler_timestep_schedule(1)
        t = schedule[0]
        v = temporal_score_rescale(v_pred=v, x_t=init, t=t, rescale_k=2.0, rescale_sigma=0.5)
        expected = init + v * (schedule[1] - schedule[0])
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_sampler_forces_cache_when_speaker_kv_scale_is_enabled(self):
        model = FakeSamplerModel(caption=False, speaker=True)
        sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            seed=7,
            use_context_kv_cache=False,
            speaker_kv_scale=1.5,
        )

        self.assertGreater(model.cache_builds, 0)
        self.assertTrue(all(call["cached"] for call in model.calls))

    @require_mlx
    def test_sampler_leaves_joint_uncond_speaker_cache_unscaled(self):
        model = CacheSpeakerScaleSamplerModel()
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=1,
            cfg_scale_text=2.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            cfg_guidance_mode="joint",
            seed=7,
            use_context_kv_cache=False,
            speaker_kv_scale=3.0,
        )

        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(7))
        v_cond = 31.0
        v_uncond = 10.0
        expected = init + (v_cond + 2.0 * (v_cond - v_uncond)) * (0.0 - 0.999)
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_sampler_unscales_speaker_cache_after_min_t(self):
        model = CacheSpeakerScaleSamplerModel()
        out = sample_euler_rf_cfg(
            model,
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=mx.ones((1, 1, 2), dtype=mx.float32),
            ref_mask=mx.array([[True]]),
            sequence_length=1,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            seed=7,
            use_context_kv_cache=False,
            speaker_kv_scale=3.0,
            speaker_kv_min_t=0.5,
        )

        init = mx.random.normal((1, 1, 2), dtype=mx.float32, key=mx.random.key(7))
        schedule = euler_timestep_schedule(2)
        expected = init + 31.0 * (schedule[1] - schedule[0]) + 11.0 * (schedule[2] - schedule[1])
        np.testing.assert_allclose(to_np(out), to_np(expected), rtol=1e-6, atol=1e-6)

    @require_mlx
    def test_sampler_rejects_invalid_quality_knobs(self):
        common = dict(
            model=FakeSamplerModel(caption=False, speaker=False),
            text_input_ids=mx.array([[1]], dtype=mx.int32),
            text_mask=mx.array([[True]]),
            ref_latent=None,
            ref_mask=None,
            sequence_length=1,
            num_steps=1,
        )
        with self.assertRaisesRegex(ValueError, "rescale_k and rescale_sigma"):
            sample_euler_rf_cfg(**common, rescale_k=2.0)
        with self.assertRaisesRegex(ValueError, "rescale_k must be > 0"):
            sample_euler_rf_cfg(**common, rescale_k=float("nan"), rescale_sigma=0.5)
        with self.assertRaisesRegex(ValueError, "speaker-conditioned"):
            sample_euler_rf_cfg(**common, speaker_kv_scale=1.5)


if __name__ == "__main__":
    unittest.main()
