from __future__ import annotations

import unittest

import scripts.convert_weights as convert_weights


class ConvertWeightsScriptTests(unittest.TestCase):
    def test_validate_base_config_rejects_caption_conditioned_checkpoint(self):
        errors = convert_weights.validate_base_config(
            {
                "latent_dim": 32,
                "model_dim": 1280,
                "num_layers": 12,
                "text_layers": 10,
                "speaker_layers": 8,
                "use_caption_condition": True,
            }
        )
        self.assertIn("VoiceDesign/caption checkpoints are not supported: use_caption_condition=true", errors)

    def test_validate_records_reports_caption_keys_as_unsupported(self):
        records = {
            key: convert_weights.TensorRecord(name=key, shape=shape, dtype="F32")
            for key, shape in convert_weights.EXPECTED_SHAPES.items()
        }
        records["caption_norm.weight"] = convert_weights.TensorRecord(
            name="caption_norm.weight",
            shape=(512,),
            dtype="F32",
        )
        validation = convert_weights.validate_records(
            records,
            {
                "latent_dim": 32,
                "model_dim": 1280,
                "num_layers": 12,
                "text_layers": 10,
                "speaker_layers": 8,
                "speaker_dim": 768,
            },
        )
        self.assertFalse(validation["ok"])
        self.assertIn("caption_norm.weight", validation["unexpected_keys"])
        self.assertIn("caption_norm.weight", validation["unsupported_keys"])


if __name__ == "__main__":
    unittest.main()
