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

    def test_upstream_dependency_boundary_docs_name_import_paths_and_split(self):
        root = Path(__file__).resolve().parents[1]
        dependency_doc = (root / "docs" / "upstream_dependency.md").read_text(encoding="utf-8")
        packaging_doc = (root / "docs" / "packaging.md").read_text(encoding="utf-8")
        dacvae_doc = (root / "docs" / "dacvae_bridge.md").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")

        for doc in (dependency_doc, packaging_doc, dacvae_doc, readme):
            self.assertIn("irodori_tts.codec.DACVAECodec", doc)
            self.assertIn("pip install -e /path/to/Irodori-TTS", doc)
            self.assertIn("PYTHONPATH=/path/to/Irodori-TTS", doc)

        self.assertIn("this repository owns the MLX text/caption conditioning", dependency_doc)
        self.assertIn("upstream `irodori_tts` still owns the PyTorch", dependency_doc)
        self.assertIn("full MLX DACVAE port is not required", dependency_doc)
        self.assertIn("does **not** provide standalone v0.1 WAV generation", dependency_doc)


if __name__ == "__main__":
    unittest.main()
