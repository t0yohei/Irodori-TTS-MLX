from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import scripts.convert_dacvae_codec as convert_dacvae_codec


SEMANTIC_KEYS = [
    "encoder.block.0.bias",
    "encoder.block.0.parametrizations.weight.original0",
    "quantizer.in_proj.bias",
    "quantizer.in_proj.parametrizations.weight.original0",
    "quantizer.out_proj.bias",
    "decoder.block.0.bias",
]


class ConvertDACVAECodecScriptTests(unittest.TestCase):
    def test_blocked_report_records_encoder_contract_and_no_output_artifact(self):
        report = convert_dacvae_codec.build_blocked_conversion_report(
            source="/tmp/weights.pth",
            output="/tmp/dacvae-codec.npz",
            state_keys=SEMANTIC_KEYS,
        )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["state_dict"]["encode_contract_present"])
        self.assertTrue(report["state_dict"]["decode_contract_present"])
        self.assertTrue(report["artifact"]["not_written"])
        self.assertIn("VAEBottleneck", report["blocker"])
        self.assertIn("semantic_encoder_manifest_json", report["artifact"]["would_require"])

    def test_blocked_report_detects_missing_quantizer_in_proj(self):
        report = convert_dacvae_codec.build_blocked_conversion_report(
            source="/tmp/weights.pth",
            output="/tmp/dacvae-codec.npz",
            state_keys=["encoder.block.0.bias", "decoder.block.0.bias"],
        )

        self.assertFalse(report["state_dict"]["encode_contract_present"])
        self.assertTrue(report["state_dict"]["encode_groups_present"]["encoder"])
        self.assertFalse(report["state_dict"]["encode_groups_present"]["quantizer_in_proj"])

    def test_model_state_dict_wrapper_feeds_accurate_blocker_diagnosis(self):
        fake_torch = types.SimpleNamespace(
            load=mock.Mock(
                return_value={
                    "epoch": 12,
                    "model_state_dict": {key: object() for key in SEMANTIC_KEYS},
                }
            )
        )

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            keys = convert_dacvae_codec.load_state_dict_keys("/tmp/weights.pth")

        report = convert_dacvae_codec.build_blocked_conversion_report(
            source="/tmp/weights.pth",
            output="/tmp/dacvae-codec.npz",
            state_keys=keys,
        )

        self.assertEqual(keys, sorted(SEMANTIC_KEYS))
        self.assertEqual(report["state_dict"]["key_count"], len(SEMANTIC_KEYS))
        self.assertTrue(report["state_dict"]["encode_contract_present"])
        self.assertTrue(report["state_dict"]["decode_contract_present"])

    def test_main_inspect_only_writes_report_and_returns_success(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "report.json"
            output_path = Path(td) / "dacvae-codec.npz"
            with mock.patch.object(convert_dacvae_codec, "load_state_dict_keys", return_value=SEMANTIC_KEYS):
                rc = convert_dacvae_codec.main(
                    [
                        "/tmp/weights.pth",
                        str(output_path),
                        "--inspect-only",
                        "--report-json",
                        str(report_path),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertFalse(output_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["requested_output"], str(output_path))

    def test_main_conversion_attempt_returns_blocked_status(self):
        with mock.patch.object(convert_dacvae_codec, "load_state_dict_keys", return_value=SEMANTIC_KEYS):
            rc = convert_dacvae_codec.main(["/tmp/weights.pth", "/tmp/dacvae-codec.npz"])

        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
