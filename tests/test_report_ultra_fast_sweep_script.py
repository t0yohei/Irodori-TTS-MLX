from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import scripts.report_ultra_fast_sweep as sweep


class UltraFastSweepReportScriptTests(unittest.TestCase):
    def _write_summary(self, root: Path, label: str, *, steps: int, cfg: float, total: float) -> Path:
        path = root / label / "persistent-batch-summary.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "invocation": {
                        "case_label": label,
                        "num_steps": steps,
                        "cfg_guidance_mode": "independent",
                        "cfg_scale_text": cfg,
                    },
                    "process": {"wall_seconds": 10.0, "measured_generation_throughput_rps": 0.5},
                    "aggregates": {
                        "measured_total_to_decode_ms": {"median": total},
                        "measured_sample_rf_ms": {"median": total - 800.0},
                        "measured_decode_dacvae_ms": {"median": 750.0},
                        "measured_decode_dacvae_model_ms": {"median": 735.0},
                        "measured_audio_write_ms": {"median": 4.0},
                        "measured_output_duration_seconds": {"median": 1.24},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_parse_summary_extracts_latency_and_cfg(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_summary(Path(td), "steps-6", steps=6, cfg=1.0, total=1220.0)
            candidate = sweep.parse_summary(path)

        self.assertEqual(candidate.num_steps, 6)
        self.assertEqual(candidate.cfg_scale_text, 1.0)
        self.assertEqual(candidate.audio_write_ms, 4.0)
        self.assertEqual(candidate.output_duration_seconds, 1.24)
        self.assertEqual(candidate.quality_proxy, "experimental-fastest")

    def test_build_report_ranks_by_request_wall_latency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            slow = sweep.parse_summary(self._write_summary(root, "steps-12", steps=12, cfg=3.0, total=1670.0))
            fast = sweep.parse_summary(self._write_summary(root, "steps-6", steps=6, cfg=1.0, total=1220.0))

        report = sweep.build_report(
            [slow, fast],
            args=argparse.Namespace(
                issue_url="issue",
                parent_url="parent",
                baseline_url="baseline",
                persistent_baseline_url="persistent",
            ),
        )

        self.assertLess(report.index("steps-6"), report.index("steps-12"))
        self.assertIn("Audio write", report)
        self.assertIn("Public preset recommendation", report)


if __name__ == "__main__":
    unittest.main()
