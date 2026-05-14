from __future__ import annotations

import subprocess
import sys
import tempfile
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

        self.assertIn('[project.scripts]', pyproject)
        self.assertIn('irodori-tts-generate = "scripts.generate_wav:cli_main"', pyproject)
        self.assertIn('irodori-tts-convert = "scripts.convert_weights:cli_main"', pyproject)
        self.assertIn('irodori-tts-inspect = "scripts.inspect_checkpoint:cli_main"', pyproject)

    def test_packaging_doc_references_editable_install_targets(self):
        root = Path(__file__).resolve().parents[1]
        packaging_doc = (root / "docs" / "packaging.md").read_text(encoding="utf-8")
        self.assertIn('pip install -e ".[runtime]"', packaging_doc)
        self.assertIn('pip install -e ".[bench]"', packaging_doc)
        self.assertIn('pip install -e ".[dev]"', packaging_doc)
        self.assertIn("irodori-tts-generate", packaging_doc)
        self.assertIn("irodori-tts-convert", packaging_doc)
        self.assertIn("irodori-tts-inspect", packaging_doc)
        self.assertIn("Python 3.11", packaging_doc)
        self.assertIn("Python 3.12", packaging_doc)
        self.assertIn("Python 3.13", packaging_doc)
        self.assertIn("Python 3.14", packaging_doc)

    @unittest.skipIf(sys.version_info < (3, 11), "project console scripts require Python >= 3.11")
    def test_console_entry_point_help_smoke_after_editable_install(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            venv = Path(td) / "venv"
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, cwd=root)
            python = venv / "bin" / "python"
            bin_dir = venv / "bin"
            subprocess.run(
                [str(python), "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(root)],
                check=True,
                cwd=root,
            )
            for command, expected in (
                ("irodori-tts-generate", "Generate a WAV"),
                ("irodori-tts-convert", "Convert a local Irodori-TTS safetensors checkpoint"),
                ("irodori-tts-inspect", "Inspect a local safetensors checkpoint"),
            ):
                script = bin_dir / command
                result = subprocess.run(
                    [str(script), "--help"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertIn(expected, result.stdout)

            convert_missing_source = subprocess.run(
                [str(bin_dir / "irodori-tts-convert")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(convert_missing_source.returncode, 0)
            self.assertIn("error: source path is required unless --self-test is used", convert_missing_source.stderr)


if __name__ == "__main__":
    unittest.main()
