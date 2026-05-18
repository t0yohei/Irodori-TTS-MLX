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
        self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], "mlx-decode")
        self.assertEqual(argv[argv.index("--codec-artifact-repo") + 1], web_ui.DEFAULT_CODEC_ARTIFACT_REPO)
        self.assertNotIn("--cfg-scale-text", argv)
        self.assertNotIn("--cfg-scale-caption", argv)
        self.assertNotIn("--cfg-scale-speaker", argv)
        self.assertNotIn("--cfg-guidance-mode", argv)

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

    def test_local_codec_path_overrides_default_hosted_codec_artifact(self):
        argv = web_ui.build_generate_argv(
            web_ui.WebGenerationConfig(codec_path="/tmp/codec.npz", codec_artifact_revision="abc123"),
            output_wav="/tmp/out.wav",
            metadata_json="/tmp/meta.json",
        )

        self.assertIn("--codec-path", argv)
        self.assertNotIn("--codec-artifact-repo", argv)
        self.assertNotIn("--codec-artifact-revision", argv)

    def test_pytorch_codec_modes_do_not_inject_hosted_codec_artifact(self):
        for mode in ("persistent", "subprocess"):
            with self.subTest(mode=mode):
                argv = web_ui.build_generate_argv(
                    web_ui.WebGenerationConfig(codec_runtime_mode=mode, codec_artifact_revision="abc123"),
                    output_wav="/tmp/out.wav",
                    metadata_json="/tmp/meta.json",
                )

                self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], mode)
                self.assertNotIn("--codec-artifact-repo", argv)
                self.assertNotIn("--codec-artifact-revision", argv)

    def test_mlx_codec_modes_include_hosted_codec_artifact_and_revision(self):
        for mode in web_ui.MLX_CODEC_RUNTIME_MODES:
            with self.subTest(mode=mode):
                argv = web_ui.build_generate_argv(
                    web_ui.WebGenerationConfig(codec_runtime_mode=mode, codec_artifact_revision="abc123"),
                    output_wav="/tmp/out.wav",
                    metadata_json="/tmp/meta.json",
                )

                self.assertEqual(argv[argv.index("--codec-runtime-mode") + 1], mode)
                self.assertEqual(argv[argv.index("--codec-artifact-repo") + 1], web_ui.DEFAULT_CODEC_ARTIFACT_REPO)
                self.assertEqual(argv[argv.index("--codec-artifact-revision") + 1], "abc123")

    def test_explicit_zero_cfg_values_are_preserved(self):
        argv = web_ui.build_generate_argv(
            web_ui.WebGenerationConfig(
                cfg_scale_text=0,
                cfg_scale_caption=0,
                cfg_scale_speaker=0,
                cfg_guidance_mode="joint",
            ),
            output_wav="/tmp/out.wav",
            metadata_json="/tmp/meta.json",
        )

        self.assertEqual(argv[argv.index("--cfg-scale-text") + 1], "0.0")
        self.assertEqual(argv[argv.index("--cfg-scale-caption") + 1], "0.0")
        self.assertEqual(argv[argv.index("--cfg-scale-speaker") + 1], "0.0")
        self.assertEqual(argv[argv.index("--cfg-guidance-mode") + 1], "joint")

    def test_run_generation_uses_subprocess_and_reads_metadata(self):
        old_argv = list(sys.argv)

        def fake_run(command, **kwargs):
            metadata_path = Path(command[command.index("--metadata-json") + 1])
            metadata_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="generated\n", stderr="")

        with tempfile.TemporaryDirectory() as td, patch.object(web_ui.subprocess, "run", side_effect=fake_run) as run:
            audio, metadata, logs = web_ui.run_generation(
                web_ui.WebGenerationConfig(output_dir=td),
            )

        self.assertEqual(sys.argv, old_argv)
        self.assertEqual(run.call_args.args[0][:3], [sys.executable, "-m", "scripts.generate_wav"])
        self.assertTrue(str(audio).endswith("irodori-web-output.wav"))
        self.assertIn('"ok": true', metadata)
        self.assertIn("generated", logs)

    def test_run_generation_reports_generator_failure(self):
        completed = types.SimpleNamespace(returncode=2, stdout="", stderr="failed")
        with patch.object(web_ui.subprocess, "run", return_value=completed):
            audio, _metadata, logs = web_ui.run_generation(web_ui.WebGenerationConfig())

        self.assertIsNone(audio)
        self.assertIn("generation failed with exit code 2", logs)
        self.assertIn("failed", logs)

    def test_run_generation_clears_stale_metadata_before_failure(self):
        with tempfile.TemporaryDirectory() as td:
            metadata_path = Path(td) / "irodori-web-metadata.json"
            metadata_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

            completed = types.SimpleNamespace(returncode=2, stdout="", stderr="failed")
            with patch.object(web_ui.subprocess, "run", return_value=completed):
                audio, metadata, logs = web_ui.run_generation(web_ui.WebGenerationConfig(output_dir=td))

        self.assertIsNone(audio)
        self.assertEqual(metadata, "")
        self.assertIn("generation failed with exit code 2", logs)

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

    def test_main_launches_with_pwa_enabled(self):
        class FakeDemo:
            def __init__(self):
                self.launch_kwargs = None

            def launch(self, **kwargs):
                self.launch_kwargs = kwargs

        demo = FakeDemo()

        with patch.object(web_ui, "build_ui", return_value=demo):
            rc = web_ui.main(["--host", "127.0.0.1", "--port", "7861"])

        self.assertEqual(rc, 0)
        self.assertTrue(demo.launch_kwargs["pwa"])
        self.assertEqual(demo.launch_kwargs["server_name"], "127.0.0.1")
        self.assertEqual(demo.launch_kwargs["server_port"], 7861)


if __name__ == "__main__":
    unittest.main()
