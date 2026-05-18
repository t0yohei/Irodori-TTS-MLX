from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    HAS_BUILD_MODULE = importlib.util.find_spec("build.__main__") is not None
except ModuleNotFoundError:
    HAS_BUILD_MODULE = False


def _pyproject_extra_entries(pyproject: str, extra: str) -> list[str]:
    marker = f"{extra} = ["
    start = pyproject.index(marker)
    end = pyproject.index("]", start)
    entries: list[str] = []
    for line in pyproject[start:end].splitlines()[1:]:
        stripped = line.strip().rstrip(",")
        if stripped.startswith('"') and stripped.endswith('"'):
            entries.append(stripped[1:-1])
    return entries


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_declares_supported_python_and_extras(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        runtime_extra = _pyproject_extra_entries(pyproject, "runtime")

        self.assertIn('requires = ["setuptools>=77", "wheel"]', pyproject)
        self.assertIn('name = "irodori-tts-mlx"', pyproject)
        self.assertIn('version = "0.3.0a1"', pyproject)
        self.assertNotIn('version = "0.0.0"', pyproject)
        self.assertIn('requires-python = ">=3.11,<3.15"', pyproject)
        self.assertIn('license = "MIT"', pyproject)
        self.assertNotIn('"License :: Other/Proprietary License"', pyproject)
        self.assertNotIn('"License :: OSI Approved :: MIT License"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.11"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.12"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.13"', pyproject)
        self.assertIn('"Programming Language :: Python :: 3.14"', pyproject)
        self.assertIn('"mlx>=0.25,<1"', pyproject)
        self.assertIn('"numpy>=1.26,<3"', pyproject)

        self.assertIn('[project.optional-dependencies]', pyproject)
        self.assertIn('runtime = [', pyproject)
        self.assertIn('bench = [', pyproject)
        self.assertIn('web = [', pyproject)
        self.assertIn('dev = [', pyproject)
        self.assertIn('"torch>=2.6,<3"', pyproject)
        self.assertIn('"transformers>=4.51,<5"', pyproject)
        self.assertIn('"gradio>=4.44,<6"', pyproject)
        self.assertIn('"sentencepiece>=0.1.99,<0.2; python_version < \'3.12\'"', pyproject)
        self.assertIn('"sentencepiece>=0.2,<1; python_version >= \'3.12\'"', pyproject)
        self.assertNotIn('"sentencepiece>=0.2,<1",', pyproject)
        self.assertIn('"safetensors>=0.4,<1"', pyproject)
        self.assertIn('"pytest>=8,<9"', pyproject)
        self.assertNotIn("torch>=2.6,<3", runtime_extra)
        self.assertNotIn("torchaudio>=2.6,<3", runtime_extra)

        self.assertIn('[project.scripts]', pyproject)
        self.assertIn('irodori-tts-generate = "scripts.generate_wav:cli_main"', pyproject)
        self.assertIn('irodori-tts-web = "scripts.web_ui:cli_main"', pyproject)
        self.assertIn('irodori-tts-convert = "scripts.convert_weights:cli_main"', pyproject)
        self.assertIn('irodori-tts-convert-dacvae-codec = "scripts.convert_dacvae_codec:cli_main"', pyproject)
        self.assertIn('irodori-tts-convert-dacvae-decoder = "scripts.convert_dacvae_decoder:main"', pyproject)
        self.assertIn('irodori-tts-inspect = "scripts.inspect_checkpoint:cli_main"', pyproject)
        self.assertIn('irodori-tts-adapt-mlx-audio = "scripts.adapt_mlx_audio:cli_main"', pyproject)

    def test_source_distribution_manifest_keeps_release_docs_and_tests(self):
        root = Path(__file__).resolve().parents[1]
        manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("include LICENSE", manifest)
        self.assertIn("include README.md", manifest)
        self.assertIn("include README.ja.md", manifest)
        self.assertIn("recursive-include docs *.md *.json", manifest)
        self.assertIn("recursive-include tests *.py", manifest)

    def test_packaging_doc_references_editable_install_targets(self):
        root = Path(__file__).resolve().parents[1]
        packaging_doc = (root / "docs" / "packaging.md").read_text(encoding="utf-8")
        self.assertIn('pip install -e ".[runtime]"', packaging_doc)
        self.assertIn('pip install -e ".[bench]"', packaging_doc)
        self.assertIn('pip install -e ".[dev]"', packaging_doc)
        self.assertIn("irodori-tts-generate", packaging_doc)
        self.assertIn("irodori-tts-convert", packaging_doc)
        self.assertIn("irodori-tts-inspect", packaging_doc)
        self.assertIn("irodori-tts-adapt-mlx-audio", packaging_doc)
        self.assertIn("0.3.0a1", packaging_doc)
        self.assertIn("python -m build", packaging_doc)
        self.assertIn("Python 3.11", packaging_doc)
        self.assertIn("Python 3.12", packaging_doc)
        self.assertIn("Python 3.13", packaging_doc)
        self.assertIn("Python 3.14", packaging_doc)
        self.assertIn("installed console scripts", packaging_doc)
        self.assertIn("not a stable public Python API", packaging_doc)
        self.assertIn("public_api_stability.md", packaging_doc)
        self.assertIn("sentencepiece>=0.1.99,<0.2", packaging_doc)
        self.assertIn("Python 3.11", packaging_doc)
        self.assertIn("Python 3.12 and newer", packaging_doc)
        self.assertIn("standalone MLX runtime", packaging_doc)
        self.assertIn("audited artifacts", packaging_doc)

    @unittest.skipIf(sys.version_info < (3, 11), "project console scripts require Python >= 3.11")
    @unittest.skipUnless(HAS_BUILD_MODULE, "clean wheel smoke requires the optional build package")
    def test_console_entry_point_help_smoke_after_clean_wheel_install(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "dist"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--sdist",
                    "--outdir",
                    str(build_dir),
                ],
                check=True,
                cwd=root,
            )
            wheels = sorted(build_dir.glob("irodori_tts_mlx-0.3.0a1-py3-none-any.whl"))
            sdists = sorted(build_dir.glob("irodori_tts_mlx-0.3.0a1.tar.gz"))
            self.assertEqual(1, len(wheels))
            self.assertEqual(1, len(sdists))

            venv = Path(td) / "wheel-venv"
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, cwd=root)
            python = venv / "bin" / "python"
            bin_dir = venv / "bin"
            subprocess.run(
                [str(python), "-m", "pip", "install", "--quiet", "--no-deps", str(wheels[0])],
                check=True,
                cwd=root,
            )
            for command, expected in (
                ("irodori-tts-generate", "Generate a WAV"),
                ("irodori-tts-convert", "Convert a local Irodori-TTS safetensors checkpoint"),
                ("irodori-tts-convert-dacvae-codec", "Semantic-DACVAE weights.pth"),
                ("irodori-tts-convert-dacvae-decoder", "Convert Aratako/Semantic-DACVAE-Japanese-32dim"),
                ("irodori-tts-inspect", "Inspect a local safetensors checkpoint"),
                ("irodori-tts-adapt-mlx-audio", "Convert an unquantized mlx-audio Irodori directory"),
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

    def test_upstream_dependency_boundary_docs_name_import_paths_and_split(self):
        root = Path(__file__).resolve().parents[1]
        dependency_doc = (root / "docs" / "upstream_dependency.md").read_text(encoding="utf-8")
        packaging_doc = (root / "docs" / "packaging.md").read_text(encoding="utf-8")
        dacvae_doc = (root / "docs" / "dacvae_bridge.md").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")

        for doc in (dependency_doc, packaging_doc, dacvae_doc, readme):
            self.assertIn("irodori_tts.codec.DACVAECodec", doc)
            self.assertIn("does not require upstream", doc)
            self.assertIn("--codec-runtime-mode mlx", doc)

        self.assertIn("The old bridge-backed generation modes", dependency_doc)
        self.assertIn("this MLX repo owns the text/caption conditioning", dependency_doc)
        self.assertIn("standalone v0.3 alpha WAV generation", dependency_doc)

    def test_license_and_distribution_policy_is_documented(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        readme_ja = (root / "README.ja.md").read_text(encoding="utf-8")
        policy_doc = (root / "docs" / "license_and_distribution.md").read_text(encoding="utf-8")
        license_file = (root / "LICENSE").read_text(encoding="utf-8")

        self.assertIn("MIT License", license_file)
        self.assertIn("[MIT License](LICENSE)", readme)
        self.assertIn("docs/license_and_distribution.md", readme)
        self.assertIn("docs/license_and_distribution.md", readme_ja)

        for text in (readme, policy_doc):
            self.assertIn("does **not** redistribute", text)
            self.assertIn("converted `.npz`", text)
            self.assertIn("generated audio", text)

        self.assertIn("Irodori-TTS 500M v3 checkpoint", policy_doc)
        self.assertIn("Semantic-DACVAE Japanese codec weights", policy_doc)


if __name__ == "__main__":
    unittest.main()
