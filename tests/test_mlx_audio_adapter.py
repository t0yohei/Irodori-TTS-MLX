from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from irodori_mlx.hosted_weights import validate_weights_layout
from irodori_mlx.mlx_audio_adapter import (
    MlxAudioAdapterError,
    adapt_mlx_audio_layout,
    remap_mlx_audio_tensor_name,
    translate_mlx_audio_config,
)


MLX_AUDIO_CONFIG: dict[str, Any] = {
    "model_type": "irodori_tts",
    "sample_rate": 48000,
    "dacvae_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
    "dit": {
        "latent_dim": 32,
        "latent_patch_size": 1,
        "model_dim": 1280,
        "num_layers": 12,
        "num_heads": 20,
        "mlp_ratio": 2.875,
        "text_mlp_ratio": 2.6,
        "speaker_mlp_ratio": 2.6,
        "text_vocab_size": 99574,
        "text_tokenizer_repo": "llm-jp/llm-jp-3-150m",
        "text_add_bos": True,
        "text_dim": 512,
        "text_layers": 10,
        "text_heads": 8,
        "speaker_dim": 768,
        "speaker_layers": 8,
        "speaker_heads": 12,
        "speaker_patch_size": 1,
        "timestep_embed_dim": 512,
        "adaln_rank": 192,
        "norm_eps": 1e-5,
    },
    "sampler": {"num_steps": 40, "cfg_guidance_mode": "independent", "sequence_length": 750},
}


def _write_mlx_audio_layout(root: Path, config: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"fixture")
    (root / "dacvae").mkdir()
    (root / "dacvae" / "config.json").write_text('{"sample_rate": 48000}\n', encoding="utf-8")


def _fixture_converter(source_safetensors: Path, output_npz: Path, model_config: Mapping[str, Any]) -> Mapping[str, Any]:
    assert source_safetensors.name == "model.safetensors"
    assert model_config["latent_dim"] == 32
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **{"text_norm.weight": np.ones((1,), dtype=np.float32)})
    family = "voicedesign" if model_config.get("use_caption_condition") else "base_v2"
    return {"ok": True, "checkpoint_family": family}


class MlxAudioAdapterTests(unittest.TestCase):
    def test_translates_nested_mlx_audio_config_to_flat_model_config(self):
        translated = translate_mlx_audio_config(MLX_AUDIO_CONFIG)

        self.assertEqual(translated["latent_dim"], 32)
        self.assertEqual(translated["text_tokenizer_repo"], "llm-jp/llm-jp-3-150m")
        self.assertFalse(translated["use_caption_condition"])

    def test_adapts_unquantized_mlx_audio_layout_to_valid_hosted_layout(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "mlx-audio"
            output = Path(td) / "hosted"
            _write_mlx_audio_layout(source, MLX_AUDIO_CONFIG)

            result = adapt_mlx_audio_layout(
                source,
                output,
                source_repo="mlx-community/Irodori-TTS-500M-v2-fp16",
                source_revision="abc123",
                weight_converter=_fixture_converter,
            )
            resolved = validate_weights_layout(result.output_dir)

        self.assertEqual(result.checkpoint_family, "base_v2")
        self.assertEqual(resolved.manifest["upstream_checkpoint"], "mlx-community/Irodori-TTS-500M-v2-fp16")
        self.assertEqual(resolved.conversion_metadata["upstream"]["checkpoint_revision"], "abc123")
        self.assertEqual(resolved.model_config.text_tokenizer_repo, "llm-jp/llm-jp-3-150m")
        self.assertTrue(resolved.conversion_metadata["upstream"]["dacvae_present"])

    def test_quantized_mlx_audio_metadata_is_rejected_with_targeted_error(self):
        config = dict(MLX_AUDIO_CONFIG)
        config["quantization"] = {"bits": 4, "group_size": 64}
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "mlx-audio"
            output = Path(td) / "hosted"
            _write_mlx_audio_layout(source, config)

            with self.assertRaisesRegex(MlxAudioAdapterError, "quantized mlx-audio Irodori artifacts.*4-bit"):
                adapt_mlx_audio_layout(source, output, weight_converter=_fixture_converter)

    def test_remaps_common_mlx_audio_tensor_prefixes(self):
        self.assertEqual(remap_mlx_audio_tensor_name("model.blocks.0.attention.wq.weight"), "blocks.0.attention.wq.weight")
        self.assertEqual(remap_mlx_audio_tensor_name("dit.text_norm.weight"), "text_norm.weight")
        self.assertEqual(remap_mlx_audio_tensor_name("text_norm.weight"), "text_norm.weight")


if __name__ == "__main__":
    unittest.main()
