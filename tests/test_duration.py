from __future__ import annotations

import unittest

import numpy as np

try:
    from irodori_mlx.duration import build_duration_features, estimate_fallback_duration_seconds

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


class DurationFeatureTests(unittest.TestCase):
    def test_estimate_fallback_duration_short_prompt_is_below_old_fixed_five_seconds(self):
        seconds = estimate_fallback_duration_seconds("こんにちは。")

        self.assertGreaterEqual(seconds, 1.6)
        self.assertLess(seconds, 5.0)

    def test_estimate_fallback_duration_smoke_text_is_longer_than_old_fixed_five_seconds(self):
        seconds = estimate_fallback_duration_seconds(
            "こんにちは。私はいろどりです。今日は音声生成のテストをしています。"
        )

        self.assertGreater(seconds, 5.0)
        self.assertLess(seconds, 8.0)

    def test_estimate_fallback_duration_clamps_very_long_prompts(self):
        seconds = estimate_fallback_duration_seconds("今日は音声生成のテストです。" * 20)

        self.assertEqual(seconds, 10.0)

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
