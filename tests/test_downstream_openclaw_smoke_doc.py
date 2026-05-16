from __future__ import annotations

import unittest
from pathlib import Path


class DownstreamOpenClawSmokeDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "downstream_openclaw_smoke.md").read_text(encoding="utf-8")

    def test_doc_links_issue_and_names_downstream_entry_points(self):
        required_terms = [
            "#146",
            "#123",
            "toyon-tech/openclaw-workspace-redacted",
            "apps/physical-client/clients/client-core/integrations/tts.js",
            "skills/kouka-voice-playback/scripts/aivis_playback.py",
            "tools/irodori-tts/openai-compatible-irodori-server-runtime.py",
            "irodori-tts-generate",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_covers_required_inputs_command_metadata_and_wav_checks(self):
        required_terms = [
            "IRODORI_MLX_REPO",
            "IRODORI_UPSTREAM_REPO",
            "IRODORI_SMOKE_DIR",
            "IRODORI_WEIGHTS",
            "IRODORI_MODEL_CONFIG",
            "--metadata-json",
            "openclaw-smoke.wav",
            "openclaw-smoke.metadata.json",
            "checkpoint_family",
            "checkpoint_capabilities",
            "duration_mode",
            "codec_backend",
            "codec_encode_backend",
            "codec_decode_backend",
            "timings_ms",
            'metadata_payload["result"]',
            'stdout_payload["result"]',
            "wave.open",
            "getnframes",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_defines_fallbacks_and_artifact_policy(self):
        required_terms = [
            "Hosted weights unavailable",
            "Missing upstream irodori_tts",
            "Missing MLX codec artifact",
            "Missing MLX runtime dependencies",
            "Missing local speaker or playback tools",
            "Do not commit generated audio",
            "downloaded checkpoint caches",
            "DACVAE codec artifacts",
            "secrets",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_readmes_and_delivery_plan_link_smoke_doc(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        readme_ja = (self.root / "README.ja.md").read_text(encoding="utf-8")
        plan = (self.root / "docs" / "v0_2_delivery_plan.md").read_text(encoding="utf-8")

        for text in (readme, readme_ja, plan):
            with self.subTest(document=text[:40]):
                self.assertIn("docs/downstream_openclaw_smoke.md", text)
                self.assertIn("local-assistant/OpenClaw smoke path", text)


if __name__ == "__main__":
    unittest.main()
