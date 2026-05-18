from __future__ import annotations

import unittest
from pathlib import Path


class DACVAEDecodeParityDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "dacvae_decode_parity.md").read_text(encoding="utf-8")

    def test_decode_parity_doc_links_gate_issue_and_parent(self):
        self.assertIn("Issue #184", self.doc)
        self.assertIn("parent epic #169", self.doc)
        self.assertIn("scripts/check_dacvae_decode_parity.py", self.doc)
        self.assertIn("parity-reports/2026-05-18-dacvae-decode-parity.json", self.doc)

    def test_decode_parity_doc_defines_gate_inputs_outputs_and_tolerances(self):
        for term in (
            "`(1, T, 32)`",
            "`--expected-sample-rate`",
            "`--expected-hop-length`",
            "`--expected-latent-dim`",
            "MLX-only artifact evidence check",
            "mlx-decode.wav",
            "dacvae-decode-parity.json",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_decode_parity_doc_requires_complete_passing_report_for_parity_backed_status(self):
        self.assertIn("has_executable_mlx_decode=true", self.doc)
        self.assertIn("report is complete and passed", self.doc)
        self.assertIn("does not", self.doc)
        self.assertIn("broad acoustic parity", self.doc)
        lower_doc = self.doc.lower()
        self.assertIn("keep generated wavs", lower_doc)
        self.assertIn("artifacts out of git", lower_doc)


if __name__ == "__main__":
    unittest.main()
