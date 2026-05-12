from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
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
        return type("Result", (), {"messages": ["ok"]})()


class GenerateWavScriptTests(unittest.TestCase):
    def _args(self, output_wav: str) -> Namespace:
        return Namespace(
            weights="weights.npz",
            output=output_wav,
            text="hello",
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
            disable_codec_normalize=False,
            enable_watermark=False,
            seconds=0.1,
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


if __name__ == "__main__":
    unittest.main()
