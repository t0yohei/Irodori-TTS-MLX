from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from irodori_mlx.config import ModelConfig
import scripts.generate_wav as generate_wav


class _FakeRuntime:
    def __init__(self, config):
        self.config = config
        self.requests = []

    def describe_boundaries(self):
        return {"ok": True}

    def generate(self, request):
        self.requests.append(request)
        output = Path(request.output_wav)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake wav")
        return type(
            "Result",
            (),
            {
                "output_wav": str(output),
                "sample_rate": 24000,
                "samples": 2400,
                "latent_steps": 12,
                "patched_steps": 3,
                "seed": request.seed,
                "timings_ms": {"sample_rf": 12.5, "total_to_decode": 20.0},
                "messages": ["ok"],
            },
        )()


class GenerateWavScriptTests(unittest.TestCase):
    def _write_hosted_layout(self, root: Path, *, license_status: str = "approved") -> None:
        files = {
            "weights": "weights.npz",
            "model_config": "model_config.json",
            "tokenizer_config": "tokenizer_config.json",
            "conversion_metadata": "conversion_metadata.json",
            "checksums": "checksums.sha256",
        }
        manifest = {
            "schema_version": 1,
            "format": "irodori-tts-mlx-weights",
            "format_version": "0.2",
            "family": "v3",
            "upstream_checkpoint": "Aratako/Irodori-TTS-500M-v3",
            "files": files,
            "runtime": {
                "minimum_irodori_tts_mlx_version": "0.2.0",
                "requires_upstream_dacvae_bridge": True,
                "requires_reference_audio": False,
                "supports_no_reference": True,
                "supports_caption": False,
                "supports_predicted_duration": True,
            },
            "license_review": {"status": license_status, "review_reference": "https://github.com/t0yohei/Irodori-TTS-MLX/issues/80"},
        }
        (root / "irodori_mlx_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "model_config.json").write_text('{"use_duration_predictor": true}', encoding="utf-8")
        (root / "tokenizer_config.json").write_text(
            '{"schema_version": 1, "text_tokenizer": {"repo": "dummy"}}', encoding="utf-8"
        )
        (root / "conversion_metadata.json").write_text(
            '{"schema_version": 1, "detected_family": "v3", "converter": {"name": "test"}, "upstream": {"repo": "dummy"}}',
            encoding="utf-8",
        )
        (root / "weights.npz").write_bytes(b"fake npz")
        listed = ["irodori_mlx_manifest.json", *(value for key, value in files.items() if key != "checksums")]
        lines = []
        for name in listed:
            digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
        (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _args(self, output_wav: str) -> Namespace:
        return Namespace(
            weights="weights.npz",
            weights_dir=None,
            weights_repo=None,
            weights_revision=None,
            output=output_wav,
            text="hello",
            preset=None,
            reference_wav=None,
            no_reference=False,
            caption="calm",
            model_config_json='{"use_caption_condition": true}',
            text_tokenizer_repo=None,
            caption_tokenizer_repo=None,
            text_max_length=256,
            caption_max_length=None,
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            codec_path=None,
            codec_device="cpu",
            codec_runtime_mode="subprocess",
            disable_codec_normalize=False,
            enable_watermark=False,
            seconds=0.1,
            duration_scale=1.0,
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            cfg_guidance_mode="independent",
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            seed=0,
            max_reference_seconds=30.0,
            no_context_kv_cache=False,
            print_boundaries=False,
            config_json=None,
            requests_json=None,
            metadata_json=None,
            json_output=False,
        )

    def test_caption_conditioned_checkpoint_allows_missing_reference_without_no_reference_flag(self):
        runtime_holder = {}

        def fake_runtime_factory(*, config):
            runtime_holder["runtime"] = _FakeRuntime(config)
            return runtime_holder["runtime"]

        with tempfile.TemporaryDirectory() as td:
            args = self._args(str(Path(td) / "out.wav"))
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "load_model_config_json", return_value=ModelConfig(use_caption_condition=True)
            ), patch.object(generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory), patch.object(
                generate_wav, "iter_messages", return_value=iter([])
            ):
                rc = generate_wav.main()

        self.assertEqual(rc, 0)
        runtime = runtime_holder["runtime"]
        self.assertEqual(len(runtime.requests), 1)
        request = runtime.requests[0]
        self.assertIsNone(request.reference_wav)
        self.assertFalse(request.no_reference)
        self.assertEqual(runtime.config.codec.runtime_mode, "subprocess")

    def test_build_runtime_config_accepts_mlx_codec_artifact_path(self):
        args = self._args("out.wav")
        args.codec_runtime_mode = "mlx"
        args.codec_path = "codec.npz"

        runtime_config = generate_wav.build_runtime_config(args, ModelConfig(use_caption_condition=True))

        self.assertEqual(runtime_config.codec.runtime_mode, "mlx")
        self.assertEqual(runtime_config.codec.codec_path, "codec.npz")

    def test_parse_args_config_json_accepts_mlx_codec_artifact_path(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "weights": "weights.npz",
                        "output": "out.wav",
                        "text": "hello",
                        "codec_runtime_mode": "mlx",
                        "codec_path": "codec.npz",
                    }
                ),
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(cfg_path)])

        self.assertEqual(args.codec_runtime_mode, "mlx")
        self.assertEqual(args.codec_path, "codec.npz")

    def test_parse_args_config_json_supplies_defaults_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "from-config.npz", "output": "from-config.wav", "text": "hello", "seconds": 1.5, "num_steps": 8}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args([
                "--config-json",
                str(cfg_path),
                "--output",
                str(Path(td) / "override.wav"),
                "--seconds",
                "2.5",
            ])

        self.assertEqual(args.weights, "from-config.npz")
        self.assertTrue(str(args.output).endswith("override.wav"))
        self.assertEqual(args.text, "hello")
        self.assertEqual(args.seconds, 2.5)
        self.assertEqual(args.num_steps, 8)

    def test_parse_args_accepts_weights_dir_instead_of_npz(self):
        args = generate_wav.parse_args(
            [
                "--weights-dir",
                "converted-layout",
                "--output",
                "out.wav",
                "--text",
                "hello",
            ]
        )

        self.assertIsNone(args.weights)
        self.assertEqual(args.weights_dir, "converted-layout")
        self.assertIsNone(args.weights_repo)

    def test_parse_args_cli_weight_source_overrides_config_weight_source(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "from-config.npz", "model_config_json": "config-model.json", "text_tokenizer_repo": "current/text", "caption_tokenizer_repo": "current/caption", "output": "from-config.wav", "text": "hello"}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(cfg_path), "--weights-dir", "converted-layout"])

        self.assertIsNone(args.weights)
        self.assertEqual(args.weights_dir, "converted-layout")
        self.assertIsNone(args.model_config_json)
        self.assertIsNone(args.text_tokenizer_repo)
        self.assertIsNone(args.caption_tokenizer_repo)

    def test_parse_args_preserves_explicit_tokenizer_overrides_with_layout(self):
        args = generate_wav.parse_args(
            [
                "--weights-dir",
                "converted-layout",
                "--text-tokenizer-repo",
                "custom/text",
                "--caption-tokenizer-repo",
                "custom/caption",
                "--output",
                "out.wav",
                "--text",
                "hello",
            ]
        )

        self.assertEqual(args.weights_dir, "converted-layout")
        self.assertEqual(args.text_tokenizer_repo, "custom/text")
        self.assertEqual(args.caption_tokenizer_repo, "custom/caption")

    def test_parse_args_rejects_layout_with_explicit_model_config_override(self):
        with self.assertRaises(SystemExit):
            generate_wav.parse_args(
                [
                    "--weights-dir",
                    "converted-layout",
                    "--model-config-json",
                    "model.json",
                    "--output",
                    "out.wav",
                    "--text",
                    "hello",
                ]
            )

    def test_parse_args_cli_repo_override_clears_config_revision(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights_repo": "owner/old", "weights_revision": "oldrev", "output": "from-config.wav", "text": "hello"}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(cfg_path), "--weights-repo", "owner/new"])

        self.assertEqual(args.weights_repo, "owner/new")
        self.assertIsNone(args.weights_revision)

    def test_parse_args_rejects_weights_revision_without_repo(self):
        with self.assertRaises(SystemExit):
            generate_wav.parse_args(
                [
                    "--weights",
                    "weights.npz",
                    "--weights-revision",
                    "abc123",
                    "--output",
                    "out.wav",
                    "--text",
                    "hello",
                ]
            )

    def test_main_uses_resolved_layout_weights_and_model_config(self):
        runtime_holder = {}

        def fake_runtime_factory(*, config):
            runtime_holder["runtime"] = _FakeRuntime(config)
            return runtime_holder["runtime"]

        with tempfile.TemporaryDirectory() as td:
            out_wav = str(Path(td) / "out.wav")
            args = self._args(out_wav)
            args.weights = None
            args.weights_dir = str(Path(td) / "layout")
            resolved = type(
                "ResolvedLayout",
                (),
                {
                    "weights_path": Path(td) / "layout" / "weights.npz",
                    "model_config": ModelConfig(use_caption_condition=True, text_tokenizer_repo="layout/text"),
                    "manifest": {"runtime": {"requires_reference_audio": False, "supports_no_reference": True}},
                },
            )()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "resolve_weights_layout_source", return_value=resolved
            ), patch.object(generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory), patch.object(
                generate_wav, "iter_messages", return_value=iter([])
            ):
                rc = generate_wav.main()

        self.assertEqual(rc, 0)
        self.assertEqual(runtime_holder["runtime"].config.weights_path, str(resolved.weights_path))
        self.assertEqual(runtime_holder["runtime"].config.model_config.text_tokenizer_repo, "layout/text")

    def test_main_rejects_layout_no_reference_when_manifest_requires_reference(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._args(str(Path(td) / "out.wav"))
            args.weights = None
            args.weights_dir = str(Path(td) / "layout")
            args.no_reference = True
            resolved = type(
                "ResolvedLayout",
                (),
                {
                    "weights_path": Path(td) / "layout" / "weights.npz",
                    "model_config": ModelConfig(use_caption_condition=False),
                    "manifest": {"runtime": {"requires_reference_audio": True, "supports_no_reference": False}},
                },
            )()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "resolve_weights_layout_source", return_value=resolved
            ), self.assertRaisesRegex(SystemExit, "requires reference_wav"):
                generate_wav.main()

    def test_parse_args_applies_preset_num_steps(self):
        args = generate_wav.parse_args(
            [
                "--weights",
                "weights.npz",
                "--output",
                "out.wav",
                "--text",
                "hello",
                "--preset",
                "fast",
            ]
        )

        self.assertEqual(args.preset, "fast")
        self.assertEqual(args.num_steps, 12)

    def test_parse_args_manual_num_steps_overrides_preset(self):
        args = generate_wav.parse_args(
            [
                "--weights",
                "weights.npz",
                "--output",
                "out.wav",
                "--text",
                "hello",
                "--preset",
                "balanced",
                "--num-steps",
                "16",
            ]
        )

        self.assertEqual(args.preset, "balanced")
        self.assertEqual(args.num_steps, 16)

    def test_parse_args_config_preset_supplies_default_num_steps(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "from-config.npz", "output": "from-config.wav", "text": "hello", "preset": "quality"}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(cfg_path)])

        self.assertEqual(args.preset, "quality")
        self.assertEqual(args.num_steps, 40)

    def test_parse_args_cli_preset_overrides_config_num_steps(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "from-config.npz", "output": "from-config.wav", "text": "hello", "num_steps": 8}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(cfg_path), "--preset", "balanced"])

        self.assertEqual(args.preset, "balanced")
        self.assertEqual(args.num_steps, 24)

    def test_parse_args_allows_batch_requests_to_supply_text_and_output(self):
        with tempfile.TemporaryDirectory() as td:
            requests_path = Path(td) / "requests.json"
            requests_path.write_text(
                '[{"text": "first", "output": "first.wav"}, {"text": "second", "output": "second.wav"}]',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(
                [
                    "--weights",
                    "weights.npz",
                    "--requests-json",
                    str(requests_path),
                ]
            )

        self.assertEqual(args.weights, "weights.npz")
        self.assertIsNone(args.text)
        self.assertIsNone(args.output)
        self.assertEqual(args.requests_json, str(requests_path))

    def test_load_generation_requests_json_validates_request_keys(self):
        with tempfile.TemporaryDirectory() as td:
            requests_path = Path(td) / "requests.json"
            requests_path.write_text('[{"text": "hello", "output": "out.wav", "unexpected": true}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_wav.load_generation_requests_json(str(requests_path))

    def test_build_generation_request_applies_per_request_preset(self):
        args = self._args("default.wav")
        args.num_steps = 40
        request = generate_wav.build_generation_request(
            args,
            {
                "text": "override",
                "output": "override.wav",
                "preset": "fast",
                "seed": 123,
                "no_context_kv_cache": True,
            },
        )

        self.assertEqual(request.text, "override")
        self.assertEqual(request.output_wav, "override.wav")
        self.assertEqual(request.num_steps, 12)
        self.assertEqual(request.seed, 123)
        self.assertFalse(request.use_context_kv_cache)

    def test_parse_args_allows_omitted_seconds_for_auto_duration(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "from-config.npz", "output": "from-config.wav", "text": "hello", "seconds": null, "duration_scale": 1.2}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(cfg_path)])

        self.assertIsNone(args.seconds)
        self.assertEqual(args.duration_scale, 1.2)

    def test_parse_args_config_boolean_can_be_disabled_from_cli(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "from-config.npz", "output": "from-config.wav", "text": "hello", "no_reference": true}',
                encoding="utf-8",
            )
            args = generate_wav.parse_args(
                [
                    "--config-json",
                    str(cfg_path),
                    "--use-reference",
                    "--reference-wav",
                    "ref.wav",
                ]
            )

        self.assertFalse(args.no_reference)
        self.assertEqual(args.reference_wav, "ref.wav")

    def test_parse_args_rejects_conflicting_reference_flags(self):
        with self.assertRaises(SystemExit):
            generate_wav.parse_args(
                [
                    "--weights",
                    "weights.npz",
                    "--output",
                    "out.wav",
                    "--text",
                    "hello",
                    "--reference-wav",
                    "ref.wav",
                    "--no-reference",
                ]
            )

    def test_parse_args_rejects_invalid_boolean_config_value(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "weights.npz", "output": "out.wav", "text": "hello", "no_reference": "false"}',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                generate_wav.parse_args(["--config-json", str(cfg_path)])

    def test_parse_args_rejects_invalid_numeric_config_value(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "weights.npz", "output": "out.wav", "text": "hello", "seconds": "fast"}',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                generate_wav.parse_args(["--config-json", str(cfg_path)])

    def test_parse_args_rejects_null_required_config_value(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": null, "output": "out.wav", "text": "hello"}',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                generate_wav.parse_args(["--config-json", str(cfg_path)])

    def test_parse_args_rejects_non_positive_duration_scale(self):
        with self.assertRaises(SystemExit):
            generate_wav.parse_args(
                [
                    "--weights",
                    "weights.npz",
                    "--output",
                    "out.wav",
                    "--text",
                    "hello",
                    "--duration-scale",
                    "0",
                ]
            )

    def test_parse_args_rejects_invalid_metadata_json_type(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "generate.json"
            cfg_path.write_text(
                '{"weights": "weights.npz", "output": "out.wav", "text": "hello", "metadata_json": true}',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                generate_wav.parse_args(["--config-json", str(cfg_path)])


    def test_parse_args_uses_weights_repo_without_model_alias(self):
        help_text = generate_wav.build_parser().format_help()
        self.assertIn("--weights-repo", help_text)
        self.assertNotIn("--model ", help_text)
        self.assertIn("converted .npz fallback", help_text)

        args = generate_wav.parse_args(["--weights-repo", "org/repo", "--output", "out.wav", "--text", "hello"])

        self.assertIsNone(args.weights)
        self.assertEqual(args.weights_repo, "org/repo")

        with self.assertRaises(SystemExit):
            generate_wav.parse_args(["--model", "org/repo", "--output", "out.wav", "--text", "hello"])

    def test_resolve_preconverted_weights_dir_supplies_weights_and_model_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="pending")
            args = generate_wav.parse_args(["--weights-dir", str(root), "--output", "out.wav", "--text", "hello"])

            generate_wav.resolve_preconverted_weights_args(args)

        self.assertTrue(args.weights.endswith("weights.npz"))
        self.assertTrue(args.model_config_json.endswith("model_config.json"))

    def test_resolve_preconverted_weights_repo_uses_snapshot_download_and_requires_approved_license(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="approved")
            args = generate_wav.parse_args(["--weights-repo", "org/repo", "--output", "out.wav", "--text", "hello"])
            with patch.object(generate_wav, "_download_weights_repo_snapshot", return_value=root):
                generate_wav.resolve_preconverted_weights_args(args)

        self.assertTrue(args.weights.endswith("weights.npz"))
        self.assertTrue(args.model_config_json.endswith("model_config.json"))

    def test_resolve_preconverted_weights_source_overrides_config_model_config_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            self._write_hosted_layout(root, license_status="approved")
            config_path = Path(td) / "generate.json"
            stale_config = Path(td) / "stale_model_config.json"
            stale_config.write_text("{}", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "weights": "stale.npz",
                        "model_config_json": str(stale_config),
                        "output": "out.wav",
                        "text": "hello",
                    }
                ),
                encoding="utf-8",
            )
            args = generate_wav.parse_args(["--config-json", str(config_path), "--weights-dir", str(root)])

            generate_wav.resolve_preconverted_weights_args(args)

        self.assertTrue(args.weights.endswith("weights.npz"))
        self.assertTrue(args.model_config_json.endswith("model_config.json"))
        self.assertNotEqual(args.model_config_json, str(stale_config))

    def test_resolve_preconverted_weights_repo_rejects_unapproved_license_with_fallback_hint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="pending")
            args = generate_wav.parse_args(["--weights-repo", "org/repo", "--output", "out.wav", "--text", "hello"])
            with patch.object(generate_wav, "_download_weights_repo_snapshot", return_value=root), self.assertRaisesRegex(
                ValueError, "locally converted .npz fallback"
            ):
                generate_wav.resolve_preconverted_weights_args(args)

    def test_download_weights_repo_snapshot_honors_manifest_declared_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {
                "files": {
                    "weights": "artifacts/weights-v3.npz",
                    "model_config": "configs/model-v3.json",
                    "tokenizer_config": "configs/tokenizer.json",
                    "conversion_metadata": "metadata/conversion.json",
                    "checksums": "metadata/checksums.sha256",
                },
                "license_review": {"status": "approved"},
            }
            manifest_path = root / "irodori_mlx_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            captured = {}

            class FakeHfApi:
                def model_info(self, *, repo_id):
                    captured["model_info_repo_id"] = repo_id
                    return type("ModelInfo", (), {"sha": "abc123"})()

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.HfApi = FakeHfApi

            def fake_hf_hub_download(*, repo_id, filename, revision):
                captured["manifest_revision"] = revision
                return str(manifest_path)

            def fake_snapshot_download(*, repo_id, revision, allow_patterns):
                captured["snapshot_revision"] = revision
                captured["allow_patterns"] = allow_patterns
                return str(root)

            fake_hub.hf_hub_download = fake_hf_hub_download
            fake_hub.snapshot_download = fake_snapshot_download
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                snapshot = generate_wav._download_weights_repo_snapshot("org/repo")

        self.assertEqual(snapshot, root)
        self.assertEqual(captured["model_info_repo_id"], "org/repo")
        self.assertEqual(captured["manifest_revision"], "abc123")
        self.assertEqual(captured["snapshot_revision"], "abc123")
        self.assertIn("artifacts/weights-v3.npz", captured["allow_patterns"])
        self.assertIn("configs/model-v3.json", captured["allow_patterns"])
        self.assertIn("irodori_mlx_manifest.json", captured["allow_patterns"])

    def test_resolve_preconverted_weights_dir_allows_snapshot_symlink_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snapshot"
            root.mkdir()
            blob_dir = Path(td) / "blobs"
            blob_dir.mkdir()
            blob_weights = blob_dir / "weights-blob.npz"
            blob_weights.write_bytes(b"fake")
            self._write_hosted_layout(root, license_status="approved")
            (root / "weights.npz").unlink()
            (root / "weights.npz").symlink_to(blob_weights)
            checksums = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
            rewritten = []
            for line in checksums:
                rewritten.append(f"{hashlib.sha256(blob_weights.read_bytes()).hexdigest()}  weights.npz" if line.endswith("  weights.npz") else line)
            (root / "checksums.sha256").write_text("\n".join(rewritten) + "\n", encoding="utf-8")
            args = generate_wav.parse_args(["--weights-dir", str(root), "--output", "out.wav", "--text", "hello"])

            generate_wav.resolve_preconverted_weights_args(args)

            self.assertEqual(Path(args.weights).name, "weights.npz")
            self.assertTrue(Path(args.weights).is_symlink())

    def test_resolve_preconverted_weights_dir_rejects_manifest_paths_outside_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            self._write_hosted_layout(root, license_status="approved")
            manifest_path = root / "irodori_mlx_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["weights"] = "../outside.npz"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root.parent / "outside.npz").write_bytes(b"not hosted")
            args = generate_wav.parse_args(
                ["--weights-dir", str(root), "--output", "out.wav", "--text", "hello"]
            )

            with self.assertRaisesRegex(ValueError, "must stay inside the hosted weights layout"):
                generate_wav.resolve_preconverted_weights_args(args)

    def test_resolve_preconverted_weights_dir_requires_exact_checksum_filenames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="approved")
            (root / "checksums.sha256").write_text(
                "\n".join(
                    [
                        f"{'0' * 64}  irodori_mlx_manifest.json",
                        f"{'0' * 64}  model_config.json",
                        f"{'0' * 64}  tokenizer_config.json",
                        f"{'0' * 64}  conversion_metadata.json",
                        f"{'0' * 64}  weights.npz.bak",
                    ]
                ),
                encoding="utf-8",
            )
            args = generate_wav.parse_args(
                ["--weights-dir", str(root), "--output", "out.wav", "--text", "hello"]
            )

            with self.assertRaisesRegex(
                ValueError, "checksums.sha256 does not name required files: weights.npz"
            ):
                generate_wav.resolve_preconverted_weights_args(args)

    def test_main_smoke_uses_mocked_repo_id_resolution(self):
        runtime_holder = {}

        def fake_runtime_factory(*, config):
            runtime_holder["runtime"] = _FakeRuntime(config)
            return runtime_holder["runtime"]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repo"
            root.mkdir()
            self._write_hosted_layout(root, license_status="approved")
            out_wav = str(Path(td) / "out.wav")
            args = generate_wav.parse_args([
                "--weights-repo",
                "org/repo",
                "--output",
                out_wav,
                "--text",
                "hello",
                "--no-reference",
                "--json",
            ])
            stdout = StringIO()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "_download_weights_repo_snapshot", return_value=root
            ), patch.object(generate_wav, "load_model_config_json", return_value=ModelConfig(use_duration_predictor=True)), patch.object(
                generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory
            ), redirect_stdout(stdout):
                rc = generate_wav.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["result"]["output_wav"], out_wav)
        self.assertTrue(runtime_holder["runtime"].config.weights_path.endswith("weights.npz"))

    def test_main_json_output_and_metadata_file(self):
        runtime_holder = {}

        def fake_runtime_factory(*, config):
            runtime_holder["runtime"] = _FakeRuntime(config)
            return runtime_holder["runtime"]

        with tempfile.TemporaryDirectory() as td:
            out_wav = str(Path(td) / "out.wav")
            metadata_path = str(Path(td) / "metadata.json")
            args = self._args(out_wav)
            args.metadata_json = metadata_path
            args.json_output = True
            stdout = StringIO()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "load_model_config_json", return_value=ModelConfig(use_caption_condition=True)
            ), patch.object(generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory), redirect_stdout(stdout):
                rc = generate_wav.main()

            payload = json.loads(stdout.getvalue())
            written = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(payload["result"]["output_wav"], out_wav)
        self.assertEqual(payload["result"]["timings_ms"]["sample_rf"], 12.5)
        self.assertEqual(payload["request"]["caption"], "calm")
        self.assertEqual(payload["request"]["duration_scale"], 1.0)
        self.assertEqual(written["result"]["samples"], 2400)

    def test_main_batch_requests_reuse_one_runtime(self):
        runtime_holder = {}

        def fake_runtime_factory(*, config):
            runtime_holder["runtime"] = _FakeRuntime(config)
            return runtime_holder["runtime"]

        with tempfile.TemporaryDirectory() as td:
            requests_path = Path(td) / "requests.json"
            first_wav = str(Path(td) / "first.wav")
            second_wav = str(Path(td) / "second.wav")
            requests_path.write_text(
                json.dumps(
                    [
                        {"text": "first", "output": first_wav, "preset": "fast"},
                        {"text": "second", "output": second_wav, "num_steps": 7, "seed": 42},
                    ]
                ),
                encoding="utf-8",
            )
            args = self._args("")
            args.output = None
            args.text = None
            args.requests_json = str(requests_path)
            args.json_output = True
            stdout = StringIO()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "load_model_config_json", return_value=ModelConfig(use_caption_condition=True)
            ), patch.object(generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory), redirect_stdout(stdout):
                rc = generate_wav.main()

            payload = json.loads(stdout.getvalue())

        runtime = runtime_holder["runtime"]
        self.assertEqual(rc, 0)
        self.assertEqual(len(runtime.requests), 2)
        self.assertEqual(runtime.requests[0].text, "first")
        self.assertEqual(runtime.requests[0].num_steps, 12)
        self.assertEqual(runtime.requests[1].text, "second")
        self.assertEqual(runtime.requests[1].num_steps, 7)
        self.assertEqual(payload["batch"]["count"], 2)
        self.assertEqual(len(payload["results"]), 2)

    def test_main_single_batch_request_keeps_batch_envelope(self):
        runtime_holder = {}

        def fake_runtime_factory(*, config):
            runtime_holder["runtime"] = _FakeRuntime(config)
            return runtime_holder["runtime"]

        with tempfile.TemporaryDirectory() as td:
            requests_path = Path(td) / "requests.json"
            output_wav = str(Path(td) / "only.wav")
            requests_path.write_text(json.dumps([{"text": "only", "output": output_wav}]), encoding="utf-8")
            args = self._args("")
            args.output = None
            args.text = None
            args.requests_json = str(requests_path)
            args.json_output = True
            stdout = StringIO()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "load_model_config_json", return_value=ModelConfig(use_caption_condition=True)
            ), patch.object(generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory), redirect_stdout(stdout):
                rc = generate_wav.main()

            payload = json.loads(stdout.getvalue())

        runtime = runtime_holder["runtime"]
        self.assertEqual(rc, 0)
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(payload["batch"]["count"], 1)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["result"]["output_wav"], output_wav)

    def test_main_print_boundaries_uses_stderr_in_json_mode(self):
        def fake_runtime_factory(*, config):
            return _FakeRuntime(config)

        with tempfile.TemporaryDirectory() as td:
            args = self._args(str(Path(td) / "out.wav"))
            args.print_boundaries = True
            args.json_output = True
            stdout = StringIO()
            stderr = StringIO()
            with patch.object(generate_wav, "parse_args", return_value=args), patch.object(
                generate_wav, "load_model_config_json", return_value=ModelConfig(use_caption_condition=True)
            ), patch.object(generate_wav, "MLXDACVAERuntime", side_effect=fake_runtime_factory), redirect_stdout(stdout), redirect_stderr(stderr):
                rc = generate_wav.main()

        payload = json.loads(stdout.getvalue())
        boundaries = json.loads(stderr.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["result"]["output_wav"], args.output)
        self.assertEqual(boundaries, {"ok": True})


if __name__ == "__main__":
    unittest.main()
