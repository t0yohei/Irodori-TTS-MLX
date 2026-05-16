from __future__ import annotations

import unittest
from pathlib import Path


class V02DeliveryPlanDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "v0_2_delivery_plan.md").read_text(encoding="utf-8")

    def test_plan_links_github_issue_cluster(self):
        for issue in range(105, 122):
            with self.subTest(issue=issue):
                self.assertIn(f"/issues/{issue}", self.doc)

    def test_plan_covers_required_workstreams_and_validation_gates(self):
        required_terms = [
            "Runtime UX and duration handling",
            "DACVAE MLX port and artifact policy",
            "Upstream-vs-MLX parity harness",
            "Downstream consumer handoff",
            "Release and runbook cleanup",
            "Completion criteria",
            "Dependency order",
            "Current risks",
            "Tracking policy",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_readmes_link_delivery_plan(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        readme_ja = (self.root / "README.ja.md").read_text(encoding="utf-8")

        for text in (readme, readme_ja):
            with self.subTest(document=text[:40]):
                self.assertIn("docs/v0_2_delivery_plan.md", text)
                self.assertIn("downstream", text)


if __name__ == "__main__":
    unittest.main()
