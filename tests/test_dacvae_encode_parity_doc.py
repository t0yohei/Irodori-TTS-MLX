from __future__ import annotations

import unittest
from pathlib import Path


class DACVAEEncodeParityDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "dacvae_encode_parity.md").read_text(encoding="utf-8")

    def test_doc_links_current_child_and_parent_issues(self):
        self.assertIn("Issue #155", self.doc)
        self.assertIn("parent epic #160", self.doc)
        self.assertNotIn("Issue #115 tracks", self.doc)

    def test_doc_defines_complete_failed_partial_report_contract(self):
        for term in (
            "`complete`: upstream and MLX encode both ran",
            "`failed`: upstream and MLX encode both ran",
            "`partial`: setup or encode could not reach comparison",
            "real Semantic-DACVAE encoder conversion remains",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)


if __name__ == "__main__":
    unittest.main()
