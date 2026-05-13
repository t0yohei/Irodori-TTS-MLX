from __future__ import annotations

import unittest

import scripts.convert_weights as convert_weights


class ConvertWeightsScriptTests(unittest.TestCase):
    def _config(self, *, family: str) -> dict[str, object]:
        if family == convert_weights.CHECKPOINT_FAMILY_BASE:
            return {
                "latent_dim": 32,
                "model_dim": 1280,
                "num_layers": 12,
                "text_layers": 10,
                "speaker_layers": 8,
                "speaker_dim": 768,
            }
        if family == convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN:
            return {
                "latent_dim": 32,
                "model_dim": 1280,
                "num_layers": 12,
                "text_layers": 10,
                "use_caption_condition": True,
                "caption_layers": 10,
                "caption_dim": 512,
                "caption_heads": 8,
                "caption_vocab_size": 99574,
            }
        raise AssertionError(f"unknown family: {family}")

    def _records(self, *, family: str):
        return {
            key: convert_weights.TensorRecord(name=key, shape=shape, dtype="F32")
            for key, shape in convert_weights.EXPECTED_SHAPES_BY_FAMILY[family].items()
        }

    def test_validate_records_accepts_base_checkpoint(self):
        validation = convert_weights.validate_records(
            self._records(family=convert_weights.CHECKPOINT_FAMILY_BASE),
            self._config(family=convert_weights.CHECKPOINT_FAMILY_BASE),
        )
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["checkpoint_family"], convert_weights.CHECKPOINT_FAMILY_BASE)
        self.assertEqual(
            validation["supported_checkpoint"],
            convert_weights.SUPPORTED_CHECKPOINTS[convert_weights.CHECKPOINT_FAMILY_BASE],
        )

    def test_validate_records_accepts_voicedesign_checkpoint(self):
        validation = convert_weights.validate_records(
            self._records(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN),
            self._config(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN),
        )
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["checkpoint_family"], convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN)
        self.assertEqual(
            validation["supported_checkpoint"],
            convert_weights.SUPPORTED_CHECKPOINTS[convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN],
        )

    def test_validate_records_rejects_ambiguous_mixed_layout(self):
        records = self._records(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN)
        records["speaker_norm.weight"] = convert_weights.TensorRecord(
            name="speaker_norm.weight",
            shape=(768,),
            dtype="F32",
        )
        validation = convert_weights.validate_records(
            records,
            self._config(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN),
        )
        self.assertFalse(validation["ok"])
        self.assertIsNone(validation["checkpoint_family"])
        self.assertTrue(any("ambiguous" in error for error in validation["config_errors"]))

    def test_validate_records_reports_missing_voicedesign_caption_keys(self):
        records = self._records(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN)
        records.pop("caption_norm.weight")
        validation = convert_weights.validate_records(
            records,
            self._config(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN),
        )
        self.assertFalse(validation["ok"])
        self.assertEqual(validation["checkpoint_family"], convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN)
        self.assertIn("caption_norm.weight", validation["missing_keys"])

    def test_validate_records_rejects_voicedesign_with_speaker_metadata(self):
        config = self._config(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN)
        config["speaker_layers"] = 8
        config["speaker_dim"] = 768
        validation = convert_weights.validate_records(
            self._records(family=convert_weights.CHECKPOINT_FAMILY_VOICEDESIGN),
            config,
        )
        self.assertFalse(validation["ok"])
        self.assertIsNone(validation["checkpoint_family"])
        self.assertTrue(any("speaker fields" in error or "ambiguous" in error for error in validation["config_errors"]))

    def test_validation_error_message_mentions_checkpoint_family_when_known(self):
        records = self._records(family=convert_weights.CHECKPOINT_FAMILY_BASE)
        records.pop("out_proj.bias")
        validation = convert_weights.validate_records(
            records,
            self._config(family=convert_weights.CHECKPOINT_FAMILY_BASE),
        )
        message = convert_weights.validation_error_message(validation)
        self.assertIn("checkpoint_family: base_v2", message)
        self.assertIn("out_proj.bias", message)


if __name__ == "__main__":
    unittest.main()
