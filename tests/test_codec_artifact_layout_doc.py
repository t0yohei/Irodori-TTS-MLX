from __future__ import annotations

import unittest
from pathlib import Path


class CodecArtifactLayoutDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "codec_artifact_layout.md").read_text(encoding="utf-8")

    def test_doc_defines_local_codec_artifact_files_and_provenance(self):
        for term in (
            "dacvae-codec.npz",
            "sample_rate",
            "hop_length",
            "latent_dim",
            "decode_basis",
            "decode_bias",
            "encode_basis",
            "encode_bias",
            "semantic_encoder_manifest_json",
            "artifact_kind",
            "metadata_json",
            "scripts/convert_dacvae_decoder.py",
            "dacvae_decoder/<state-dict-key>",
            "upstream codec repo id, source file, and exact revision",
            "license-review status",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_defines_hosted_companion_pointer_and_no_bundling_policy(self):
        for term in (
            '"codec"',
            '"artifact_format": "irodori-tts-mlx-dacvae-codec"',
            '"source_repo": "Aratako/Semantic-DACVAE-Japanese-32dim"',
            '"runtime_modes": ["mlx-decode", "mlx"]',
            "They do not bundle Semantic-DACVAE weights by default.",
            "PyTorch bridge fallback",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_defines_runtime_capability_and_family_fallback_policy(self):
        for term in (
            "describe_codec_capabilities()",
            "boundaries.codec.capabilities",
            "persistent",
            "subprocess",
            "mlx-decode",
            "mlx-decode-subprocess",
            "mlx",
            "base_v2",
            "voicedesign",
            "v3",
            "reference encode still uses the PyTorch bridge",
            "cannot yet execute the full",
            "--no-reference",
            "irodori-tts-convert-dacvae-codec",
            "blocked conversion status",
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_related_docs_and_readme_link_to_codec_layout(self):
        for relpath in ("README.md", "docs/dacvae_bridge.md", "docs/hosted_weights_layout.md"):
            with self.subTest(relpath=relpath):
                text = (self.root / relpath).read_text(encoding="utf-8")
                self.assertIn("codec_artifact_layout.md", text)


if __name__ == "__main__":
    unittest.main()
