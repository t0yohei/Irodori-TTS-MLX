from __future__ import annotations

import json
import unittest
from pathlib import Path


class FirstAudioLatencyDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "first_audio_latency.md").read_text(encoding="utf-8")
        self.schema = json.loads((self.root / "docs" / "first_audio_latency_report_schema.json").read_text(encoding="utf-8"))

    def test_doc_records_complete_wav_decision_and_architecture_limits(self):
        for term in [
            "complete-WAV request latency",
            "first playable audio is not currently earlier than complete-WAV availability",
            "RF-DiT sampling is full-sequence",
            "decode_to_wav(latents, output_path)",
            "sentence-level segmentation",
            "complete WAV available for playback after a warmed persistent request",
        ]:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_doc_defines_evidence_fields_and_repro_command(self):
        for term in [
            "one_shot_wall_clock_ms",
            "persistent_request_latency_ms",
            "rf_sampling_ms",
            "dacvae_decode_ms",
            "audio_serialization_write_ms",
            "decode_to_wav_ms",
            "first_audio_available_ms",
            "complete_wav_available_ms",
            "scripts/benchmark_persistent_batch.py",
            "--case-label issue-222-first-audio-v3-no-reference",
        ]:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_schema_requires_latency_contract_fields(self):
        latency = self.schema["properties"]["latency_ms"]
        self.assertEqual(
            latency["required"],
            [
                "first_audio_available_ms",
                "complete_wav_available_ms",
                "rf_sampling_ms",
                "decode_to_wav_ms",
            ],
        )
        for field in [
            "one_shot_wall_clock_ms",
            "persistent_request_latency_ms",
            "rf_sampling_ms",
            "dacvae_decode_ms",
            "audio_serialization_write_ms",
            "decode_to_wav_ms",
            "first_audio_available_ms",
            "complete_wav_available_ms",
        ]:
            with self.subTest(field=field):
                self.assertIn(field, latency["properties"])


if __name__ == "__main__":
    unittest.main()
