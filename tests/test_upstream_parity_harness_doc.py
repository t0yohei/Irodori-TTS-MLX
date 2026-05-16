from __future__ import annotations

import unittest
from pathlib import Path


class UpstreamParityHarnessDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "upstream_parity_harness.md").read_text(encoding="utf-8")
        self.baseline = (
            self.root / "docs" / "baseline-reports" / "2026-05-16-upstream-mlx-parity-baseline.md"
        ).read_text(encoding="utf-8")

    def test_harness_doc_links_current_baseline_and_setup_levels(self):
        self.assertIn("2026-05-16-upstream-mlx-parity-baseline.md", self.doc)
        for term in [
            "Fixture contract",
            "Partial setup audit",
            "Real upstream-vs-MLX run",
            "missing_upstream_root",
            "missing_mlx_weights",
        ]:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_harness_doc_has_copy_pasteable_supported_scenario_commands(self):
        required_commands = [
            "--scenario v3-no-reference",
            "--scenario voicedesign-contrastive-caption",
            "--fixture",
            "--run-upstream",
            "--run-mlx",
            "--mlx-weights",
            "--mlx-model-config-json",
        ]
        for command in required_commands:
            with self.subTest(command=command):
                self.assertIn(command, self.doc)

    def test_baseline_report_summarizes_current_results_and_runtime_boundary(self):
        for term in [
            "#121",
            "#123",
            "expected_drift",
            "v3-no-reference",
            "voicedesign-contrastive-caption",
            "missing_upstream_root",
            "missing_mlx_weights",
            "Real runtime boundary",
        ]:
            with self.subTest(term=term):
                self.assertIn(term, self.baseline)

    def test_real_v3_reference_command_documents_local_reference_wav(self):
        real_v3_section = self.doc.split("## Real v3 Command", 1)[1].split("## Real VoiceDesign Command", 1)[0]

        self.assertIn("--scenario v3-reference-predicted", real_v3_section)
        self.assertIn("--reference-wav /tmp/irodori-parity/v3-reference.wav", real_v3_section)
        self.assertIn("this repository does not ship a real speaker sample", real_v3_section)
        self.assertIn("Do not commit this local reference audio", real_v3_section)


if __name__ == "__main__":
    unittest.main()
