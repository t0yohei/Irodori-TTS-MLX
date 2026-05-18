from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.web_ui as web_ui


class WebUiScriptTests(unittest.TestCase):
    def test_voice_design_preset_builds_hosted_generation_argv(self):
        argv = web_ui.build_generate_argv(
            web_ui.WebGenerationConfig(),
            output_wav="/tmp/out.wav",
            metadata_json="/tmp/meta.json",
        )

        self.assertIn("--weights-repo", argv)
        self.assertEqual(argv[argv.index("--weights-repo") + 1], web_ui.VOICE_DESIGN_WEIGHTS_REPO)
        self.assertEqual(argv[argv.index("--weights-revision") + 1], web_ui.VOICE_DESIGN_WEIGHTS_REVISION)
        self.assertIn("--caption", argv)
        self.assertIn("--no-reference", argv)
        self.assertIn("--preset", argv)
        self.assertEqual(argv[argv.index("--preset") + 1], "balanced")
        self.assertIn("--metadata-json", argv)
        self.assertIn("--json", argv)

    def test_v3_preset_includes_pinned_revision(self):
        argv = web_ui.build_generate_argv(
            web_ui.WebGenerationConfig(artifact_preset="v3 hosted"),
            output_wav="/tmp/out.wav",
            metadata_json="/tmp/meta.json",
        )

        self.assertEqual(argv[argv.index("--weights-repo") + 1], web_ui.V3_WEIGHTS_REPO)
        self.assertEqual(argv[argv.index("--weights-revision") + 1], web_ui.V3_WEIGHTS_REVISION)
        self.assertNotIn("--caption", argv)

    def test_v3_preset_allows_reference_audio(self):
        argv = web_ui.build_generate_argv(
            web_ui.WebGenerationConfig(
                artifact_preset="v3 hosted",
                caption="",
                no_reference=False,
                reference_wav="/tmp/ref.wav",
            ),
            output_wav="/tmp/out.wav",
            metadata_json="/tmp/meta.json",
        )

        self.assertIn("--reference-wav", argv)
        self.assertEqual(argv[argv.index("--reference-wav") + 1], "/tmp/ref.wav")
        self.assertNotIn("--no-reference", argv)

    def test_custom_config_requires_exactly_one_weight_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one weights source"):
            web_ui.build_generate_argv(
                web_ui.WebGenerationConfig(artifact_preset="Custom", weights="a.npz", weights_dir="layout"),
                output_wav="/tmp/out.wav",
                metadata_json="/tmp/meta.json",
            )

    def test_run_generation_restores_sys_argv_and_reads_metadata(self):
        old_argv = list(sys.argv)

        def fake_cli_main() -> int:
            self.assertEqual(sys.argv[0], "irodori-tts-generate")
            metadata_path = Path(sys.argv[sys.argv.index("--metadata-json") + 1])
            metadata_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            print("generated")
            return 0

        with tempfile.TemporaryDirectory() as td, patch.object(web_ui.generate_wav, "cli_main", side_effect=fake_cli_main):
            audio, metadata, logs = web_ui.run_generation(
                web_ui.WebGenerationConfig(output_dir=td),
            )

        self.assertEqual(sys.argv, old_argv)
        self.assertTrue(str(audio).endswith("irodori-web-output.wav"))
        self.assertIn('"ok": true', metadata)
        self.assertIn("generated", logs)

    def test_run_generation_reports_generator_failure(self):
        with patch.object(web_ui.generate_wav, "cli_main", return_value=2):
            audio, _metadata, logs = web_ui.run_generation(web_ui.WebGenerationConfig())

        self.assertIsNone(audio)
        self.assertIn("generation failed with exit code 2", logs)

    def test_run_generation_clears_stale_metadata_before_failure(self):
        with tempfile.TemporaryDirectory() as td:
            metadata_path = Path(td) / "irodori-web-metadata.json"
            metadata_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

            with patch.object(web_ui.generate_wav, "cli_main", return_value=2):
                audio, metadata, logs = web_ui.run_generation(web_ui.WebGenerationConfig(output_dir=td))

        self.assertIsNone(audio)
        self.assertEqual(metadata, "")
        self.assertIn("generation failed with exit code 2", logs)

    def test_run_generation_preserves_system_exit_message(self):
        with patch.object(web_ui.generate_wav, "cli_main", side_effect=SystemExit("bad request")):
            audio, _metadata, logs = web_ui.run_generation(web_ui.WebGenerationConfig())

        self.assertIsNone(audio)
        self.assertIn("generation failed with exit code 1", logs)
        self.assertIn("bad request", logs)

    def test_zero_numeric_values_do_not_fall_back_to_defaults(self):
        self.assertEqual(web_ui._int_or_default(0, 1), 0)
        self.assertEqual(web_ui._float_or_default(0, 1.0), 0.0)

    def test_gradio_is_lazy_optional(self):
        parsed = web_ui.build_parser().parse_args(["--host", "0.0.0.0", "--port", "9999"])

        self.assertEqual(parsed.host, "0.0.0.0")
        self.assertEqual(parsed.port, 9999)

    def test_missing_gradio_error_mentions_web_extra(self):
        def fake_import(name: str):
            if name == "gradio":
                raise ImportError("missing")
            return types.SimpleNamespace()

        with patch.object(web_ui.importlib, "import_module", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "runtime,web"):
                web_ui.build_ui()


if __name__ == "__main__":
    unittest.main()
