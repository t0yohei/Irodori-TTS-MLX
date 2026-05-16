from __future__ import annotations

import re
import unittest
from pathlib import Path


class PublicDocsSanitizedTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_public_docs_do_not_reference_private_environment(self):
        private_user = "ko" + "uka"
        private_tool = "open" + "claw"
        private_downstream = "local-" + "assistant"
        private_rollup = "TO" + "Y-5"
        private_tracker = "linear.app/" + "toyontech"
        forbidden = re.compile(
            rf"/Users/{private_user}|\.{private_tool}|{private_tool}-workspace|"
            rf"{private_user}-voice-playback|{private_tool}|{private_downstream}|"
            rf"{private_rollup}|{re.escape(private_tracker)}",
            flags=re.IGNORECASE,
        )
        paths = [
            self.root / "README.md",
            self.root / "README.ja.md",
            *sorted((self.root / "docs").rglob("*.md")),
        ]

        violations: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if forbidden.search(line):
                    rel_path = path.relative_to(self.root)
                    violations.append(f"{rel_path}:{line_number}: {line}")

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
