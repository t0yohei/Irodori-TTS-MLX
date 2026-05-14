from __future__ import annotations

import unittest

import numpy as np

try:
    from irodori_mlx.duration import build_duration_features

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - exercised only without MLX.
    HAS_MLX = False
    MLX_IMPORT_ERROR = exc


def require_mlx(test_func):
    return unittest.skipUnless(HAS_MLX, f"MLX is not available: {globals().get('MLX_IMPORT_ERROR')}")(test_func)


class DurationFeatureTests(unittest.TestCase):
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
