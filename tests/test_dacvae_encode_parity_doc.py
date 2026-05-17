from __future__ import annotations

import unittest
from pathlib import Path


class DACVAEEncodeParityDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "dacvae_encode_parity.md").read_text(encoding="utf-8")

    def test_doc_links_current_child_and_parent_issues(self):
        self.assertIn("Issue #185", self.doc)
        self.assertIn("parent epic #169", self.doc)
        self.assertNotIn("Issue #174 tracks", self.doc)
        self.assertNotIn("Issue #155", self.doc)
        self.assertNotIn("Issue #115 tracks", self.doc)

    def test_doc_records_current_real_validation_status(self):
        for term in (
            'run.status = "complete"',
            'comparison.status = "passed"',
            "47376ee24834d7a05a48ebabfe3cde29b3c5e214",
            "414c20785fc3a28373073ea8ef7a1316eeeaca6e",
            "latent shape: `[1, 13, 32]`",
            "No threshold change was needed.",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_defines_complete_failed_partial_report_contract(self):
        for term in (
            "`passed`: upstream and MLX encode both ran",
            "`failed`: upstream and MLX encode both ran",
            "`partial`: preflight could not reach comparison",
            "--allow-partial",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)


if __name__ == "__main__":
    unittest.main()
