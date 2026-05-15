from __future__ import annotations

import re
import unittest
from pathlib import Path


class HostedWeightsLayoutDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "hosted_weights_layout.md").read_text(encoding="utf-8")

    def test_layout_defines_required_top_level_files(self):
        required_files = [
            "README.md",
            "LICENSE.md",
            "irodori_mlx_manifest.json",
            "model_config.json",
            "tokenizer_config.json",
            "conversion_metadata.json",
            "weights.npz",
            "checksums.sha256",
        ]
        for filename in required_files:
            with self.subTest(filename=filename):
                self.assertIn(filename, self.doc)

        self.assertIn("required loader inputs must stay at the top level", self.doc)
        self.assertIn("loader-required files", self.doc)
        self.assertIn("excluding itself", self.doc)
        self.assertIn("hf_hub_download", self.doc)
        self.assertIn("snapshot_download", self.doc)

    def test_manifest_contract_covers_runtime_and_provenance(self):
        expected_manifest_terms = [
            '"format": "irodori-tts-mlx-weights"',
            '"format_version": "0.2"',
            '"family": "v3"',
            '"upstream_checkpoint": "Aratako/Irodori-TTS-500M-v3"',
            '"minimum_irodori_tts_mlx_version": "0.2.0"',
            '"requires_upstream_dacvae_bridge": true',
            '"supports_predicted_duration": true',
            '"license_review"',
            "names every loader-required artifact plus the checksum file",
        ]
        for term in expected_manifest_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_local_directory_and_hosted_repo_share_contract(self):
        self.assertIn("--weights-dir /models/irodori-tts-mlx-v3-500m", self.doc)
        self.assertIn("--weights-repo t0yohei/irodori-tts-mlx-v3-500m", self.doc)
        self.assertIn("--weights /path/to/irodori-v3.npz", self.doc)
        self.assertIn("local fallback", self.doc)

        normalization_steps = [
            "resolve a local directory or remote snapshot",
            "read `irodori_mlx_manifest.json`",
            "validate required files, schema version, family, runtime flags, and license-review status",
            "load `model_config.json`, `tokenizer_config.json`, and `weights.npz` through the same internal runtime path",
        ]
        for step in normalization_steps:
            with self.subTest(step=step):
                self.assertIn(step, self.doc)

    def test_family_differences_and_redistribution_boundary_are_explicit(self):
        family_requirements = [
            "v3 (`Aratako/Irodori-TTS-500M-v3`)",
            "VoiceDesign v2 (`Aratako/Irodori-TTS-500M-v2-VoiceDesign`)",
            "base v2 (`Aratako/Irodori-TTS-500M-v2`)",
            "supports_caption: true",
            "supports_predicted_duration: true",
            "requires_reference_audio: true",
            "Recommended first candidate",
            "Conditional candidate after #81/#82",
        ]
        for term in family_requirements:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

        self.assertRegex(self.doc, re.compile(r"does not upload, publish, or bless redistribution", re.IGNORECASE))
        self.assertIn("license_review.status` is `approved`", self.doc)
        self.assertIn("license review", self.doc)

    def test_readmes_and_license_policy_link_to_layout_contract(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        readme_ja = (self.root / "README.ja.md").read_text(encoding="utf-8")
        policy = (self.root / "docs" / "license_and_distribution.md").read_text(encoding="utf-8")

        for text in (readme, readme_ja, policy):
            with self.subTest(document=text[:40]):
                self.assertIn("hosted_weights_layout.md", text)

        self.assertIn("v0.2 hosted/pre-converted MLX weights layout contract", readme)
        self.assertIn("does not approve publishing converted weights", policy)


if __name__ == "__main__":
    unittest.main()
