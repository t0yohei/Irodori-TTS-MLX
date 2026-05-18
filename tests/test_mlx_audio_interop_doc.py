from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from irodori_mlx.hosted_weights import HostedWeightsError, validate_weights_layout


MLX_AUDIO_CONFIG = {
    "model_type": "irodori_tts",
    "sample_rate": 48000,
    "max_text_length": 256,
    "max_caption_length": 512,
    "max_speaker_latent_length": 6400,
    "audio_downsample_factor": 1920,
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
    "sampler": {
        "num_steps": 40,
        "cfg_guidance_mode": "independent",
        "sequence_length": 750,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_irodori_layout(root: Path, *, model_config: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "weights.npz", **{"text_norm.weight": np.ones((1,), dtype=np.float32)})
    (root / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "text_tokenizer": {
                    "source": "upstream",
                    "normalization_contract": "docs/text_preprocessing.md",
                    "padding": "right",
                    "truncation": "family-defined",
                },
                "caption_tokenizer": None,
            }
        ),
        encoding="utf-8",
    )
    (root / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "converter": {"repository": "https://github.com/t0yohei/Irodori-TTS-MLX"},
                "upstream": {"checkpoint_repo": "mlx-community/Irodori-TTS-500M-v2-fp16"},
                "detected_family": "base_v2",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "format": "irodori-tts-mlx-weights",
        "format_version": "0.2",
        "family": "base_v2",
        "upstream_checkpoint": "mlx-community/Irodori-TTS-500M-v2-fp16",
        "files": {
            "weights": "weights.npz",
            "model_config": "model_config.json",
            "tokenizer_config": "tokenizer_config.json",
            "conversion_metadata": "conversion_metadata.json",
            "checksums": "checksums.sha256",
        },
        "runtime": {
            "minimum_irodori_tts_mlx_version": "0.2.0",
            "requires_reference_audio": True,
            "supports_no_reference": False,
            "supports_caption": False,
            "supports_predicted_duration": False,
        },
        "license_review": {"status": "pending", "review_reference": "local-test"},
    }
    (root / "irodori_mlx_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksum_files = [
        "weights.npz",
        "model_config.json",
        "tokenizer_config.json",
        "conversion_metadata.json",
        "irodori_mlx_manifest.json",
    ]
    (root / "checksums.sha256").write_text(
        "".join(f"{_sha256(root / filename)}  {filename}\n" for filename in checksum_files),
        encoding="utf-8",
    )


class MlxAudioInteropDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.doc = (self.root / "docs" / "mlx_audio_interop.md").read_text(encoding="utf-8")

    def test_report_names_evaluated_mlx_audio_artifacts_and_boundary(self):
        expected_terms = [
            "Blaizzy/mlx-audio",
            "mlx-community/Irodori-TTS-500M-v2-fp16",
            "mlx-community/Irodori-TTS-500M-v2-8bit",
            "mlx-community/Irodori-TTS-500M-v2-4bit",
            "mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16",
            "mlx-community/Irodori-TTS-500M-v2-VoiceDesign-8bit",
            "mlx-community/Irodori-TTS-500M-v2-VoiceDesign-4bit",
            "config.json",
            "model.safetensors",
            "dacvae/model.safetensors",
            "irodori_mlx_manifest.json",
            "weights.npz",
            "not drop-in compatible",
        ]
        for term in expected_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_report_records_recommended_adapter_and_dacvae_followups(self):
        expected_terms = [
            "Irodori-TTS-MLX/issues/131",
            "reject 4-bit/8-bit quantized repos",
            "Irodori-TTS-MLX/issues/130",
            "--weights-repo mlx-community/...",
            "fixed latent/audio parity fixtures",
        ]
        for term in expected_terms:
            with self.subTest(term=term):
                self.assertIn(term, self.doc)

    def test_mlx_audio_directory_is_not_a_valid_irodori_mlx_hosted_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mlx-audio-layout"
            (root / "dacvae").mkdir(parents=True)
            (root / "config.json").write_text(json.dumps(MLX_AUDIO_CONFIG), encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"placeholder")
            (root / "dacvae" / "config.json").write_text(
                json.dumps({"sample_rate": 48000, "codebook_dim": 32}),
                encoding="utf-8",
            )
            (root / "dacvae" / "model.safetensors").write_bytes(b"placeholder")

            with self.assertRaisesRegex(HostedWeightsError, "irodori_mlx_manifest.json"):
                validate_weights_layout(root)

    def test_mlx_audio_nested_config_is_rejected_as_irodori_model_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_irodori_layout(root, model_config=MLX_AUDIO_CONFIG)

            with self.assertRaisesRegex(HostedWeightsError, "unsupported keys"):
                validate_weights_layout(root)


if __name__ == "__main__":
    unittest.main()
