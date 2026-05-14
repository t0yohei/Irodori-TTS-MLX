from __future__ import annotations

import json
import tempfile
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
    def _args(self, output_wav: str) -> Namespace:
        return Namespace(
            weights="weights.npz",
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
