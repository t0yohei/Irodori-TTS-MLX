from __future__ import annotations

import unittest

import numpy as np

try:
    from irodori_mlx.duration import (
        build_duration_features,
        estimate_fallback_duration_seconds,
        estimate_voicedesign_duration_seconds,
        predicted_duration_overallocation_warning,
    )

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


class DurationFeatureTests(unittest.TestCase):
    @require_mlx
    def test_estimate_fallback_duration_short_prompt_is_below_old_fixed_five_seconds(self):
        seconds = estimate_fallback_duration_seconds("こんにちは。")

        self.assertGreaterEqual(seconds, 1.6)
        self.assertLess(seconds, 5.0)

    @require_mlx
    def test_estimate_fallback_duration_smoke_text_is_longer_than_old_fixed_five_seconds(self):
        seconds = estimate_fallback_duration_seconds(
            "こんにちは。私はいろどりです。今日は音声生成のテストをしています。"
        )

        self.assertGreater(seconds, 5.0)
        self.assertLess(seconds, 8.0)

    @require_mlx
    def test_estimate_fallback_duration_clamps_very_long_prompts(self):
        seconds = estimate_fallback_duration_seconds("今日は音声生成のテストです。" * 20)

        self.assertEqual(seconds, 10.0)

    @require_mlx
    def test_estimate_fallback_duration_treats_long_vowels_as_reduced_weight(self):
        long_vowel_seconds = estimate_fallback_duration_seconds("コーヒー")
        explicit_vowel_seconds = estimate_fallback_duration_seconds("コオヒイ")

        self.assertLess(long_vowel_seconds, explicit_vowel_seconds)

    @require_mlx
    def test_estimate_voicedesign_duration_ignores_caption_as_spoken_text(self):
        text = "こんにちは。"
        long_caption = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" * 4

        text_only = estimate_fallback_duration_seconds(text)
        voicedesign = estimate_voicedesign_duration_seconds(text, caption=long_caption)

        self.assertLess(voicedesign, estimate_fallback_duration_seconds(text + long_caption))
        self.assertLess(voicedesign, text_only * 1.12)

    @require_mlx
    def test_estimate_voicedesign_duration_caption_speed_hints_nudge_estimate(self):
        text = "こんにちは。今日は良い天気です。"

        neutral = estimate_voicedesign_duration_seconds(text, caption="自然な声")
        slow = estimate_voicedesign_duration_seconds(text, caption="落ち着いてゆっくり読み上げてください")
        fast = estimate_voicedesign_duration_seconds(text, caption="明るく元気に速めに読み上げてください")

        self.assertGreater(slow, neutral)
        self.assertLess(fast, neutral)

    @require_mlx
    def test_predicted_duration_warning_flags_short_prompt_overallocation(self):
        warning = predicted_duration_overallocation_warning(
            "こんにちは。今日は良い天気です。",
            predicted_seconds=6.0,
        )

        self.assertIsNotNone(warning)
        self.assertIn("--duration-scale 0.75", str(warning))

    @require_mlx
    def test_predicted_duration_warning_ignores_close_prediction(self):
        self.assertIsNone(
            predicted_duration_overallocation_warning(
                "こんにちは。今日は良い天気です。",
                predicted_seconds=2.9,
            )
        )

    @require_mlx
    def test_predicted_duration_warning_ignores_long_prompt(self):
        self.assertIsNone(
            predicted_duration_overallocation_warning(
                "こんにちは。私はいろどりです。今日は音声生成のテストをしています。" * 2,
                predicted_seconds=8.0,
            )
        )

    @require_mlx
    def test_build_duration_features_matches_expected_aux_dim(self):
        features = build_duration_features(
            ["こんにちはー！", "abc..."],
            token_counts=[4, 3],
            max_text_len=16,
            has_speaker=[True, False],
        )

        values = np.array(features)
        self.assertEqual(values.shape, (2, 14))
        self.assertEqual(values[0, -1], 1.0)
        self.assertEqual(values[1, -1], 0.0)
        self.assertGreater(values[0, 5], 0.0)
        self.assertGreaterEqual(values[1, 12], 0.0)


if __name__ == "__main__":
    unittest.main()
