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
            "PyTorch bridge-backed DACVAE codec path | Supported default",
            "MLX DACVAE decode for no-reference generation | Experimental",
            "Fully MLX DACVAE encode/decode for reference audio | Experimental",
            "Hosted artifacts outside the approved layouts | Blocked",
            "Unsupported upstream product features | Non-goal",
            "Training, LoRA fine-tuning, Gradio/UI hosting, watermark guarantees, arbitrary checkpoint compatibility, and stable public Python API guarantees",
            "not references to private caches, local maintainer machines, or unpublished public artifacts",
            "Local conversion is a user-managed fallback",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, readme)


if __name__ == "__main__":
    unittest.main()
