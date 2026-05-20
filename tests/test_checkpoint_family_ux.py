from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from irodori_mlx.config import ModelConfig
import scripts.generate_wav as generate_wav


class CheckpointFamilyUXTests(unittest.TestCase):
    def _args(self):
        return generate_wav.parse_args([
            "--weights",
            "weights.npz",
            "--output",
            "out.wav",
            "--text",
            "hello",
            "--no-reference",
        ])

    def test_model_config_reports_family_and_capabilities(self):
        cases = [
            (ModelConfig(), "base_v2", ("speaker-reference", "manual-or-fallback-duration")),
            (ModelConfig(use_caption_condition=True), "voicedesign", ("caption", "no-reference")),
            (ModelConfig(use_duration_predictor=True), "v3", ("speaker-reference", "predicted-duration")),
        ]

        for config, family, capabilities in cases:
            with self.subTest(family=family):
                self.assertEqual(config.checkpoint_family, family)
                for capability in capabilities:
                    self.assertIn(capability, config.checkpoint_capabilities)

    def test_non_caption_family_rejects_caption_with_actionable_family_message(self):
        args = self._args()
        args.caption = "calm voice"

        with self.assertRaisesRegex(SystemExit, "--caption is only supported by VoiceDesign v2.*base_v2"):
            generate_wav.validate_checkpoint_family_request(
                model_config=ModelConfig(),
                args=args,
                overrides={},
                index=1,
            )

    def test_voicedesign_rejects_reference_wav_no_reference_only_flow(self):
        args = self._args()
        args.no_reference = False
        args.reference_wav = "ref.wav"
        args.caption = "calm voice"

        with self.assertRaisesRegex(SystemExit, "VoiceDesign v2 caption.*no-reference only"):
            generate_wav.validate_checkpoint_family_request(
                model_config=ModelConfig(use_caption_condition=True),
                args=args,
                overrides={},
                index=1,
            )

    def test_base_family_without_seconds_warns_about_fallback_duration(self):
        args = self._args()
        args.seconds = None

        with patch("sys.stderr") as stderr:
            generate_wav.validate_checkpoint_family_request(
                model_config=ModelConfig(),
                args=args,
                overrides={},
                index=1,
            )

        self.assertIn("has no duration predictor", "".join(call.args[0] for call in stderr.write.call_args_list))

    def test_checkpoint_family_docs_cover_three_cli_examples(self):
        doc = (Path(__file__).resolve().parents[1] / "docs" / "checkpoint_support.md").read_text(encoding="utf-8")

        for term in (
            "checkpoint_family",
            "checkpoint_capabilities",
            "--weights /path/to/base-v2.npz",
            "--weights /path/to/voicedesign-v2.npz",
            "--weights /path/to/v3.npz",
            "--caption \"落ち着いた女性の声\"",
            "--reference-wav /path/to/reference.wav",
            "Omit `--seconds` to use predicted duration",
        ):
            with self.subTest(term=term):
                self.assertIn(term, doc)

    def test_checkpoint_family_docs_cover_merged_lora_boundary(self):
        doc = (Path(__file__).resolve().parents[1] / "docs" / "checkpoint_support.md").read_text(encoding="utf-8")

        for term in (
            "Upstream-merged LoRA export, layout-compatible",
            "Merged LoRA checkpoints are **experimental** and layout-bound",
            "not dynamic `--lora-adapter` inference",
            "scripts/inspect_checkpoint.py \"$MERGED\" --json",
            "scripts/convert_weights.py \"$MERGED\" \"$WORK/weights.npz\" --dry-run --json",
            "scripts/generate_wav.py",
            "missing keys, unexpected keys, shape mismatches, dtype mismatches, or config errors",
            "dynamic LoRA adapter inference",
        ):
            with self.subTest(term=term):
                self.assertIn(term, doc)


if __name__ == "__main__":
    unittest.main()
