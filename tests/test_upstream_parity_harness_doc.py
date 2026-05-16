from __future__ import annotations

import unittest
from pathlib import Path


class UpstreamParityHarnessDocTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "upstream_parity_harness.md").read_text(encoding="utf-8")

    def test_real_v3_reference_command_documents_local_reference_wav(self):
        real_v3_section = self.doc.split("## Real v3 Command", 1)[1].split("## Real VoiceDesign Command", 1)[0]

        self.assertIn("--scenario v3-reference-predicted", real_v3_section)
        self.assertIn("--reference-wav /tmp/irodori-parity/v3-reference.wav", real_v3_section)
        self.assertIn("this repository does not ship a real speaker sample", real_v3_section)
        self.assertIn("Do not commit this local reference audio", real_v3_section)


if __name__ == "__main__":
    unittest.main()
