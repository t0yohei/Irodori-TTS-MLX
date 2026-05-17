from __future__ import annotations

import unittest
from pathlib import Path


class DACVAEDecodeParityDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "dacvae_decode_parity.md").read_text(encoding="utf-8")

    def test_decode_parity_doc_links_gate_issue_and_parent(self):
        self.assertIn("Issue #172", self.doc)
        self.assertIn("parent epic #169", self.doc)
        self.assertIn("scripts/check_dacvae_decode_parity.py", self.doc)

    def test_decode_parity_doc_defines_gate_inputs_outputs_and_tolerances(self):
        for term in (
            "`(1, T, 32)`",
            "`--expected-latent-dim`",
            "`max_abs <= 5e-3`",
            "`mean_abs <= 1e-3`",
            "`rmse <= 2e-3`",
            "`cosine >= 0.999`",
            "upstream-decode.wav",
            "mlx-decode.wav",
            "dacvae-decode-parity.json",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_decode_parity_doc_keeps_executable_decode_unvalidated_until_pass(self):
        self.assertIn("has_executable_mlx_decode=true", self.doc)
        self.assertIn("available_unvalidated", self.doc)
        self.assertIn("passing report", self.doc)
        lower_doc = self.doc.lower()
        self.assertIn("keep generated wavs", lower_doc)
        self.assertIn("artifacts out of git", lower_doc)


if __name__ == "__main__":
    unittest.main()
