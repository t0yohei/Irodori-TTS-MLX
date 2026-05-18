from __future__ import annotations

import re
import unittest
from pathlib import Path


class PublicDocsSanitizedTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_public_docs_do_not_reference_private_environment(self):
        private_user = "ko" + "uka"
        private_tool = "open" + "claw"
        private_downstream = "local-" + "assistant"
        private_rollup = "TO" + "Y-5"
        private_tracker = "linear.app/" + "toyontech"
        forbidden = re.compile(
            rf"/Users/{private_user}|\.{private_tool}|{private_tool}-workspace|"
            rf"{private_user}-voice-playback|{private_tool}|{private_downstream}|"
            rf"{private_rollup}|{re.escape(private_tracker)}",
            flags=re.IGNORECASE,
        )
        paths = [
            self.root / "README.md",
            self.root / "README.ja.md",
            *sorted((self.root / "docs").rglob("*.md")),
        ]

        violations: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if forbidden.search(line):
                    rel_path = path.relative_to(self.root)
                    violations.append(f"{rel_path}:{line_number}: {line}")

        self.assertEqual([], violations)

    def test_readme_keeps_concise_support_boundaries(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")

        required_terms = [
            "## What Works Now",
            "## Install",
            "## Quickstart",
            "## Support Boundary",
            "alpha, CLI-first inference prototype",
            "VoiceDesign v2",
            "v3",
            "t0yohei/Irodori-TTS-MLX-DACVAE-Codec",
            "--codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78",
            "local Gradio UI",
            "local conversion",
            "stable public Python API",
            "arbitrary third-party, fine-tuned, quantized, LoRA, renamed, or architecture-modified checkpoints",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, readme)

        self.assertLess(readme.count("\n"), 240)

    def test_public_api_boundary_is_cli_only_for_alpha(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        readme_ja = (self.root / "README.ja.md").read_text(encoding="utf-8")
        api_doc = (self.root / "docs" / "public_api_stability.md").read_text(encoding="utf-8")
        packaging_doc = (self.root / "docs" / "packaging.md").read_text(encoding="utf-8")
        architecture_doc = (self.root / "docs" / "architecture.md").read_text(encoding="utf-8")

        for text in (readme, api_doc, packaging_doc):
            self.assertIn("installed console scripts", text)
            self.assertIn("stable public Python API", text)
            self.assertIn("irodori-tts-generate", text)
            self.assertIn("irodori-tts-adapt-mlx-audio", text)

        self.assertIn("alpha 期間中に stable-ish な対象", readme_ja)
        self.assertIn("public Python API として使うこと", readme_ja)
        self.assertIn("No `irodori_mlx` module, class, function, dataclass", api_doc)
        self.assertIn("treat them as internal implementation details", api_doc)
        self.assertIn("documented artifact layouts", api_doc)
        self.assertIn("not Python module imports", architecture_doc)
        self.assertIn("internal reusable Python modules", architecture_doc)
        self.assertIn("not a stable public Python API", (self.root / "docs" / "dacvae_bridge.md").read_text(encoding="utf-8"))
        self.assertNotIn("IrodoriGenerator.from_pretrained", architecture_doc)

    def test_readme_first_run_troubleshooting_documents_preflight(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")

        required_terms = [
            "## If It Fails",
            "--preflight",
            "exits before tokenizer loading, MLX weight loading, DACVAE bridge construction, or WAV generation",
            "text_tokenizer_repo",
            "caption_tokenizer_repo",
            "irodori_mlx_manifest.json",
            "license_review.status: \"approved\"",
            "irodori_dacvae_codec_manifest.json",
            "approved hosted DACVAE codec artifact",
            "codec runtime mode",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, readme)


if __name__ == "__main__":
    unittest.main()
