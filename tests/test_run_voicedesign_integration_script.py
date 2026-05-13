from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_voicedesign_integration as run_voicedesign_integration


class RunVoiceDesignIntegrationScriptTests(unittest.TestCase):
    def test_run_integration_reports_successful_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = str(Path(td) / "model.safetensors")
            fake_records = {"tensor": object()}
            fake_validation = {
                "ok": True,
                "checkpoint_family": "voicedesign",
                "supported_checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
                "missing_keys": [],
                "unexpected_keys": [],
                "unsupported_keys": [],
                "shape_mismatches": [],
                "dtype_mismatches": [],
                "config_errors": [],
            }
            fake_report = {
                "checkpoint_family": "voicedesign",
                "supported_checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
                "validation": fake_validation,
            }
            with patch.object(
                run_voicedesign_integration,
                "_require_hf_hub_download",
                return_value=lambda **kwargs: checkpoint_path,
            ), patch.object(
                run_voicedesign_integration,
                "inspect_local_safetensors",
                return_value=type("Inspection", (), {"tensors": [1, 2], "config": {"use_caption_condition": True}, "source": {"type": "local"}})(),
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "load_checkpoint",
                return_value=({"use_caption_condition": True}, fake_records),
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "validate_records",
                return_value=fake_validation,
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "build_report",
                return_value=fake_report,
            ):
                result = run_voicedesign_integration.run_integration(
                    repo_id="repo",
                    filename="model.safetensors",
                    revision="main",
                    download_dir=None,
                    full_conversion=False,
                )

        self.assertEqual(result["report"]["checkpoint_family"], "voicedesign")
        self.assertEqual(result["inspection"]["tensor_count"], 2)
        self.assertFalse(result["full_conversion"])
        self.assertNotIn("full_conversion_export", result)

    def test_run_integration_includes_full_conversion_export_when_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = str(Path(td) / "model.safetensors")
            fake_records = {"tensor": object()}
            fake_validation = {
                "ok": True,
                "checkpoint_family": "voicedesign",
                "supported_checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
                "missing_keys": [],
                "unexpected_keys": [],
                "unsupported_keys": [],
                "shape_mismatches": [],
                "dtype_mismatches": [],
                "config_errors": [],
            }
            with patch.object(
                run_voicedesign_integration,
                "_require_hf_hub_download",
                return_value=lambda **kwargs: checkpoint_path,
            ), patch.object(
                run_voicedesign_integration,
                "inspect_local_safetensors",
                return_value=type("Inspection", (), {"tensors": [1], "config": {"use_caption_condition": True}, "source": {"type": "local"}})(),
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "load_checkpoint",
                return_value=({"use_caption_condition": True}, fake_records),
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "validate_records",
                return_value=fake_validation,
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "build_report",
                return_value={"checkpoint_family": "voicedesign", "supported_checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign", "validation": fake_validation},
            ), patch.object(
                run_voicedesign_integration.convert_weights,
                "records_to_arrays",
                return_value={"a": 1, "b": 2},
            ):
                result = run_voicedesign_integration.run_integration(
                    repo_id="repo",
                    filename="model.safetensors",
                    revision="main",
                    download_dir=None,
                    full_conversion=True,
                )

        self.assertEqual(result["full_conversion_export"]["array_count"], 2)
        self.assertEqual(result["full_conversion_export"]["sample_keys"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
