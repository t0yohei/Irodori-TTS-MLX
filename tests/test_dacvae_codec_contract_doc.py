from __future__ import annotations

import json
import unittest
from pathlib import Path


class DACVAECodecContractDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "dacvae_architecture.md").read_text(encoding="utf-8")
        self.contract = json.loads(
            (self.root / "docs" / "dacvae_codec_contract.json").read_text(encoding="utf-8")
        )

    def test_contract_pins_runtime_constants(self):
        codec = self.contract["codec"]
        self.assertEqual(codec["repo_id"], "Aratako/Semantic-DACVAE-Japanese-32dim")
        self.assertEqual(codec["filename"], "weights.pth")
        self.assertEqual(codec["sample_rate"], 48000)
        self.assertEqual(codec["hop_length"], 1920)
        self.assertEqual(codec["latent_dim"], 32)
        self.assertEqual(codec["latent_layout_runtime"], "B,T,D")
        self.assertEqual(codec["latent_layout_dacvae"], "B,D,T")

    def test_doc_lists_architecture_and_preprocessing_contract(self):
        required_terms = [
            "encoder rates `[2, 8, 10, 12]`",
            "decoder rates `[12,10,8,2]`",
            "VAEBottleneck.in_proj",
            "mean[32] + scale[32]",
            "decoder.alpha = 0.0",
            "reflect-pad to a hop-length multiple",
            "audiotools.AudioSignal",
            "transpose to `(B,32,T)`",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_required_tensor_groups_are_actionable_for_encode_and_decode(self):
        groups = {entry["group"]: entry for entry in self.contract["required_logical_tensors"]}
        for group in ("encoder", "quantizer_in_proj", "quantizer_out_proj", "decoder"):
            with self.subTest(group=group):
                self.assertIn(group, groups)
                self.assertIn("pattern", groups[group])
                self.assertIn("shape_notes", groups[group])

        self.assertIn("encode", groups["encoder"]["required_for"])
        self.assertIn("encode", groups["quantizer_in_proj"]["required_for"])
        self.assertIn("decode", groups["quantizer_out_proj"]["required_for"])
        self.assertIn("decode", groups["decoder"]["required_for"])

    def test_family_sharing_and_unknowns_are_explicit(self):
        families = self.contract["shared_checkpoint_families"]
        self.assertIn("base_v2", families)
        self.assertIn("voicedesign_v2", families)
        self.assertIn("v3", families)

        for issue in (112, 113, 114, 115):
            with self.subTest(issue=issue):
                self.assertIn(f"/issues/{issue}", self.doc)
                self.assertIn(
                    f"https://github.com/t0yohei/Irodori-TTS-MLX/issues/{issue}",
                    self.contract["implementation_consumers"],
                )

        blocker_text = "\n".join(self.contract["unknowns_and_blockers"])
        self.assertIn("weights.pth", blocker_text)
        self.assertIn("scripts/convert_dacvae_codec.py", blocker_text)
        self.assertIn("fixed waveform/latent fixtures", blocker_text)
        self.assertIn("Redistribution", blocker_text)
        self.assertIn("https://github.com/t0yohei/Irodori-TTS-MLX/issues/154", self.contract["implementation_consumers"])
        self.assertIn("issue #184", blocker_text)
        self.assertIn("https://github.com/t0yohei/Irodori-TTS-MLX/issues/172", self.contract["implementation_consumers"])
        self.assertIn("https://github.com/t0yohei/Irodori-TTS-MLX/issues/184", self.contract["implementation_consumers"])


if __name__ == "__main__":
    unittest.main()
