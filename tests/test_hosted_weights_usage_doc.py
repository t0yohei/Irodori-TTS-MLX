from __future__ import annotations

import unittest
from pathlib import Path


class HostedWeightsUsageDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "hosted_weights_usage.md").read_text(encoding="utf-8")

    def test_doc_links_to_epic_and_layout_contract(self):
        self.assertIn("#85", self.doc)
        self.assertIn("#78", self.doc)
        self.assertIn("hosted_weights_layout.md", self.doc)
        self.assertIn("preconverted_weights_redistribution_audit.md", self.doc)
        self.assertIn("MLX RF-DiT inference + upstream PyTorch DACVAE encode/decode bridge", self.doc)

    def test_hosted_quick_path_and_local_layout_are_documented(self):
        required_terms = [
            "--weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
            "--weights-repo <approved-v3-repo-id>",
            "hosted_rf_dit_artifacts.md",
            "--weights-dir /models/Irodori-TTS-MLX-500M-v2-VoiceDesign",
            "license_review.status: \"approved\"",
            "irodori_mlx_manifest.json",
            "model_config.json",
            "tokenizer_config.json",
            "weights.npz",
            "huggingface_hub",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_local_conversion_fallback_and_boundaries_are_clear(self):
        required_terms = [
            "Fallback: local conversion",
            "scripts/inspect_checkpoint.py",
            "scripts/convert_weights.py",
            "scripts/generate_wav.py",
            "--weights \"$WORK/weights.npz\"",
            "--model-config-json \"$WORK/model_config.json\"",
            "unaudited, third-party, fine-tuned, quantized, LoRA",
            "local-conversion-only",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_readmes_link_usage_doc_and_expose_quick_path(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        readme_ja = (self.root / "README.ja.md").read_text(encoding="utf-8")

        for text in (readme, readme_ja):
            with self.subTest(document=text[:40]):
                self.assertIn("docs/hosted_weights_usage.md", text)
                self.assertIn("docs/hosted_rf_dit_artifacts.md", text)
                self.assertIn("--weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign", text)
                self.assertIn("upstream PyTorch DACVAE bridge", text)

        self.assertIn("local conversion fallback", readme)
        self.assertIn("local conversion fallback", readme_ja)


if __name__ == "__main__":
    unittest.main()
