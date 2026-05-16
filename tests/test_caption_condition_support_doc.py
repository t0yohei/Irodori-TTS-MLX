from __future__ import annotations

import unittest
from pathlib import Path


class CaptionConditionSupportDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "caption_condition_support.md").read_text(encoding="utf-8")

    def test_caption_cfg_parity_contract_is_documented(self):
        required_terms = [
            "test_fixed_seed_caption_content_changes_mlx_sample",
            "test_caption_cfg_cache_on_and_off_are_equivalent_for_same_caption",
            "test_voicedesign_caption_condition_wrapper_matches_upstream_pytorch",
            "caption_state",
            "caption_mask",
            "python3 -m unittest tests.test_sampling tests.test_runtime_bridge tests.test_generate_wav_script -v",
            "IRODORI_TTS_UPSTREAM_PATH=/path/to/Irodori-TTS",
            "python3 -m unittest tests.test_pytorch_parity -v",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_caption_prompting_and_cfg_limitations_are_documented(self):
        required_terms = [
            "--cfg-scale-caption 3.0",
            "--cfg-guidance-mode independent",
            "--cfg-guidance-mode joint",
            "--cfg-scale",
            "--no-context-kv-cache",
            "not bitwise final waveform parity",
            "different RNG streams and backend kernels",
            "alternating",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)


if __name__ == "__main__":
    unittest.main()
