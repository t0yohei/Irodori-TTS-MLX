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

    def test_v3_public_artifact_is_discoverable(self):
        v3 = hosted_rf_dit_artifacts()[CHECKPOINT_FAMILY_V3]

        self.assertTrue(v3.is_approved_public)
        self.assertEqual(v3.repo_id, "t0yohei/Irodori-TTS-MLX-500M-v3")
        self.assertEqual(v3.revision, "078ffb11ffad92e6dde237a6abef730f4341b359")
        self.assertEqual(v3.publication_status, "approved-public")
        self.assertEqual(v3.license_review_status, "approved")
        self.assertIsNone(v3.blocker)
        self.assertEqual(
            approved_hosted_rf_dit_repo(CHECKPOINT_FAMILY_V3),
            "t0yohei/Irodori-TTS-MLX-500M-v3",
        )

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

    def test_doc_records_public_status_and_smoke_commands(self):
        required_terms = [
            "#187",
            "#160",
            "irodori_mlx.hosted_artifacts.HOSTED_RF_DIT_ARTIFACTS",
            "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
            "bf877a3beb7d921dc6bfb2b6812d02be07f39f2a",
            "--weights-revision bf877a3beb7d921dc6bfb2b6812d02be07f39f2a",
            "t0yohei/Irodori-TTS-MLX-500M-v3",
            "078ffb11ffad92e6dde237a6abef730f4341b359",
            "--weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359",
            "supports_predicted_duration: true",
            "license_review.status: \"approved\"",
            "local conversion fallback",
            'mkdir -p "$WORK"',
            'python - "$WORK/checkpoint-inspect.json" > "$WORK/model_config.json"',
            "from irodori_mlx.config import ModelConfig",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_readme_v3_smoke_uses_explicit_converted_paths(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        readme_ja = (self.root / "README.ja.md").read_text(encoding="utf-8")

        for text in (readme, readme_ja):
            with self.subTest(document=text[:40]):
                self.assertIn("--weights /path/to/converted-v3/weights.npz", text)
                self.assertIn("--model-config-json /path/to/converted-v3/model_config.json", text)


if __name__ == "__main__":
    unittest.main()
