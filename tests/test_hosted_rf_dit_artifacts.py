from __future__ import annotations

import re
import unittest
from pathlib import Path

from irodori_mlx.config import CHECKPOINT_FAMILY_V3, CHECKPOINT_FAMILY_VOICEDESIGN_V2
from irodori_mlx.hosted_artifacts import (
    approved_hosted_rf_dit_artifacts,
    approved_hosted_rf_dit_repo,
    hosted_rf_dit_artifacts,
)


class HostedRfDitArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "hosted_rf_dit_artifacts.md").read_text(encoding="utf-8")

    def test_voicedesign_public_artifact_is_discoverable(self):
        artifacts = hosted_rf_dit_artifacts()
        voicedesign = artifacts[CHECKPOINT_FAMILY_VOICEDESIGN_V2]

        self.assertTrue(voicedesign.is_approved_public)
        self.assertEqual(voicedesign.repo_id, "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign")
        self.assertEqual(voicedesign.publication_status, "approved-public")
        self.assertEqual(voicedesign.license_review_status, "approved")
        self.assertRegex(voicedesign.revision or "", r"^[0-9a-f]{40}$")
        self.assertEqual(
            approved_hosted_rf_dit_repo(CHECKPOINT_FAMILY_VOICEDESIGN_V2),
            "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
        )

    def test_v3_is_an_explicit_blocker_not_a_private_path(self):
        v3 = hosted_rf_dit_artifacts()[CHECKPOINT_FAMILY_V3]

        self.assertFalse(v3.is_approved_public)
        self.assertEqual(v3.publication_status, "blocked")
        self.assertIsNone(v3.repo_id)
        self.assertIsNone(v3.revision)
        self.assertIn("No approved public hosted v3 RF-DiT artifact location", v3.blocker or "")
        with self.assertRaisesRegex(RuntimeError, "No approved public hosted v3 RF-DiT artifact location"):
            approved_hosted_rf_dit_repo(CHECKPOINT_FAMILY_V3)

    def test_approved_registry_contains_no_private_or_local_paths(self):
        forbidden = re.compile(r"(/Users/|/tmp/|file://|localhost|private|staging)", re.IGNORECASE)
        for artifact in approved_hosted_rf_dit_artifacts().values():
            values = [
                artifact.repo_id or "",
                artifact.revision or "",
                artifact.review_reference,
                artifact.issue_url,
                artifact.parent_issue_url,
            ]
            for value in values:
                with self.subTest(value=value):
                    self.assertNotRegex(value, forbidden)

    def test_doc_records_public_status_and_blocker(self):
        required_terms = [
            "#157",
            "#160",
            "irodori_mlx.hosted_artifacts.HOSTED_RF_DIT_ARTIFACTS",
            "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
            "bf877a3beb7d921dc6bfb2b6812d02be07f39f2a",
            "license_review.status: \"approved\"",
            "v3 hosted artifact is intentionally marked blocked",
            "Do not replace the blocked status with a local filesystem path",
            "local conversion fallback",
            'mkdir -p "$WORK"',
            'python - "$WORK/checkpoint-inspect.json" > "$WORK/model_config.json"',
            "from irodori_mlx.config import ModelConfig",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)


if __name__ == "__main__":
    unittest.main()
