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

    def test_readme_support_matrix_states_alpha_boundaries(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")

        required_terms = [
            "## Current Support Matrix",
            "| Surface | Status | Public support boundary |",
            "Project maturity | Alpha",
            "VoiceDesign v2 hosted RF-DiT artifact | Supported",
            "v3 hosted RF-DiT artifact | Supported",
            "Base v2 speaker-conditioned generation | Experimental",
            "Standalone MLX DACVAE codec artifact path | Supported default",
            "PyTorch bridge-backed DACVAE codec path | Supported fallback",
            "MLX DACVAE decode for no-reference generation | Supported",
            "Fully MLX DACVAE encode/decode for reference audio | Experimental",
            "Local Web UI | Optional",
            "Hosted artifacts outside the approved layouts | Blocked",
            "Unsupported upstream product features | Non-goal",
            "Training, LoRA fine-tuning, hosted demo operation, watermark guarantees, arbitrary checkpoint compatibility, and stable public Python API guarantees",
            "not a hosted demo or a stable public Python API boundary",
            "not references to private caches, local maintainer machines, or unpublished public artifacts",
            "Local conversion is a user-managed fallback",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, readme)

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

        self.assertIn("stable-ish な user contract", readme_ja)
        self.assertIn("stable public Python API としてはまだ support しません", readme_ja)
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
            "### If the quickstart fails",
            "--preflight",
            "skips tokenizer loading, MLX weight loading, DACVAE bridge construction, and WAV generation",
            "text_tokenizer_repo",
            "caption_tokenizer_repo",
            "irodori_mlx_manifest.json",
            "license_review.status: \"approved\"",
            "irodori_dacvae_codec_manifest.json",
            "default approved hosted DACVAE codec artifact",
            "--codec-runtime-mode persistent",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, readme)


if __name__ == "__main__":
    unittest.main()
