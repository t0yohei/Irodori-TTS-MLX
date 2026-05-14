from __future__ import annotations

import unittest
from pathlib import Path


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_declares_supported_python_and_extras(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('name = "irodori-tts-mlx"', pyproject)
        self.assertIn('requires-python = ">=3.11,<3.15"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.11"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.12"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.13"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.14"', pyproject)
        self.assertIn('"mlx>=0.25,<1"', pyproject)
        self.assertIn('"numpy>=1.26,<3"', pyproject)

        self.assertIn('[project.optional-dependencies]', pyproject)
        self.assertIn('runtime = [', pyproject)
        self.assertIn('bench = [', pyproject)
        self.assertIn('dev = [', pyproject)
        self.assertIn('"torch>=2.6,<3"', pyproject)
        self.assertIn('"transformers>=4.51,<5"', pyproject)
        self.assertIn('"safetensors>=0.4,<1"', pyproject)
        self.assertIn('"pytest>=8,<9"', pyproject)

    def test_packaging_doc_references_editable_install_targets(self):
        root = Path(__file__).resolve().parents[1]
        packaging_doc = (root / "docs" / "packaging.md").read_text(encoding="utf-8")
        self.assertIn('pip install -e ".[runtime]"', packaging_doc)
        self.assertIn('pip install -e ".[bench]"', packaging_doc)
        self.assertIn('pip install -e ".[dev]"', packaging_doc)
        self.assertIn("Python 3.11", packaging_doc)
        self.assertIn("Python 3.12", packaging_doc)
        self.assertIn("Python 3.13", packaging_doc)
        self.assertIn("Python 3.14", packaging_doc)


if __name__ == "__main__":
    unittest.main()
