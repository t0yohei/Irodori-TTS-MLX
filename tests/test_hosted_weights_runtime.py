from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from irodori_mlx.hosted_weights import HostedWeightsError, resolve_weights_layout_source, validate_weights_layout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_layout(
    root: Path,
    *,
    license_status: str = "pending",
    family: str = "v3",
    caption: bool = False,
    file_prefix: str = "",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_prefix.rstrip('/')}/" if file_prefix else ""
    weights_path = root / prefix / "weights.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, **{"text_norm.weight": np.ones((1,), dtype=np.float32)})
    model_config = {
        "latent_dim": 4,
        "latent_patch_size": 2,
        "model_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "mlp_ratio": 1.5,
        "text_vocab_size": 32,
        "text_tokenizer_repo": "example/text-tokenizer",
        "text_dim": 8,
        "text_layers": 1,
        "text_heads": 2,
        "speaker_dim": 8,
        "speaker_layers": 1,
        "speaker_heads": 2,
        "timestep_embed_dim": 8,
        "adaln_rank": 2,
        "dropout": 0.0,
        "use_duration_predictor": family == "v3",
        "duration_aux_dim": 14,
        "duration_hidden_dim": 8,
        "duration_layers": 1,
        "duration_dropout": 0.0,
        "duration_attention_heads": 2,
    }
    if caption:
        model_config.update(
            {
                "use_caption_condition": True,
                "caption_vocab_size": 32,
                "caption_tokenizer_repo": "example/caption-tokenizer",
                "caption_dim": 8,
                "caption_layers": 1,
                "caption_heads": 2,
                "use_duration_predictor": False,
            }
        )
    (root / prefix / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
    tokenizer_config = {
        "schema_version": 1,
        "text_tokenizer": {
            "source": "upstream",
            "normalization_contract": "docs/text_preprocessing.md",
            "padding": "right",
            "truncation": "family-defined",
        },
        "caption_tokenizer": {"source": "upstream"} if caption else None,
    }
    (root / prefix / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    conversion_metadata = {
        "schema_version": 1,
        "converter": {"repository": "https://github.com/t0yohei/Irodori-TTS-MLX", "version": "git:test"},
        "upstream": {"checkpoint_repo": "Aratako/Irodori-TTS-500M-v3", "checkpoint_revision": "test"},
        "detected_family": family,
        "license_review": {"status": license_status},
    }
    (root / prefix / "conversion_metadata.json").write_text(json.dumps(conversion_metadata), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "format": "irodori-tts-mlx-weights",
        "format_version": "0.2",
        "family": family,
        "upstream_checkpoint": "Aratako/Irodori-TTS-500M-v3",
        "files": {
            "weights": f"{prefix}weights.npz",
            "model_config": f"{prefix}model_config.json",
            "tokenizer_config": f"{prefix}tokenizer_config.json",
            "conversion_metadata": f"{prefix}conversion_metadata.json",
            "checksums": f"{prefix}checksums.sha256",
        },
        "runtime": {
            "minimum_irodori_tts_mlx_version": "0.2.0",
            "requires_reference_audio": False,
            "supports_no_reference": True,
            "supports_caption": caption,
            "supports_predicted_duration": family == "v3",
        },
        "license_review": {"status": license_status, "review_reference": "local-test"},
    }
    (root / "irodori_mlx_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksum_lines = []
    for filename in [
        f"{prefix}weights.npz",
        f"{prefix}model_config.json",
        f"{prefix}tokenizer_config.json",
        f"{prefix}conversion_metadata.json",
        "irodori_mlx_manifest.json",
    ]:
        checksum_lines.append(f"{_sha256(root / filename)}  {filename}")
    (root / prefix / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


class HostedWeightsRuntimeTests(unittest.TestCase):
    def test_validates_local_converted_layout_and_loads_runtime_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root, license_status="pending")

            resolved = validate_weights_layout(root)

        self.assertEqual(resolved.weights_path.name, "weights.npz")
        self.assertEqual(resolved.model_config.text_tokenizer_repo, "example/text-tokenizer")
        self.assertTrue(resolved.model_config.use_duration_predictor)
        self.assertEqual(resolved.source_kind, "local")

    def test_validates_local_converted_layout_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root, license_status="pending")
            archive_path = Path(td) / "layout.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in root.iterdir():
                    archive.write(path, arcname=f"weights/{path.name}")

            resolved = validate_weights_layout(archive_path)
            self.assertEqual(resolved.weights_path.name, "weights.npz")
            self.assertEqual(resolved.source, str(archive_path))
            self.assertTrue(resolved.root.is_dir())

    def test_rejects_tar_archives_with_special_members(self):
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "layout.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("layout/special.fifo")
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)

            with self.assertRaisesRegex(HostedWeightsError, "regular files and directories"):
                validate_weights_layout(archive_path)

    def test_rejects_missing_required_manifest_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root)
            (root / "tokenizer_config.json").unlink()

            with self.assertRaisesRegex(HostedWeightsError, "tokenizer_config"):
                validate_weights_layout(root)

    def test_rejects_weights_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root)
            with (root / "weights.npz").open("ab") as fh:
                fh.write(b"tampered")

            with self.assertRaisesRegex(HostedWeightsError, "checksum mismatch for weights.npz"):
                validate_weights_layout(root)

    def test_accepts_manifest_defined_relative_file_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root, file_prefix="mlx")

            resolved = validate_weights_layout(root)

        self.assertEqual(resolved.weights_path.parent.name, "mlx")
        self.assertEqual(resolved.model_config_path.parent.name, "mlx")

    def test_rejects_caption_layout_without_caption_tokenizer_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root, family="voicedesign", caption=True)
            tokenizer_config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
            tokenizer_config["caption_tokenizer"] = None
            (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
            # keep checksum coverage valid for files that are not under test here
            checksum_lines = []
            for filename in [
                "weights.npz",
                "model_config.json",
                "tokenizer_config.json",
                "conversion_metadata.json",
                "irodori_mlx_manifest.json",
            ]:
                checksum_lines.append(f"{_sha256(root / filename)}  {filename}")
            (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(HostedWeightsError, "caption_tokenizer"):
                validate_weights_layout(root)

    def test_rejects_manifest_family_mismatch_with_model_config_family(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "layout"
            _write_layout(root, family="v3", caption=False)
            model_config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
            model_config["use_duration_predictor"] = False
            (root / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
            checksum_lines = []
            for filename in [
                "weights.npz",
                "model_config.json",
                "tokenizer_config.json",
                "conversion_metadata.json",
                "irodori_mlx_manifest.json",
            ]:
                checksum_lines.append(f"{_sha256(root / filename)}  {filename}")
            (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(HostedWeightsError, "manifest family must match model_config checkpoint family"):
                validate_weights_layout(root)

    def test_hosted_repo_resolution_uses_snapshot_download_and_requires_approved_license(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snapshot"
            _write_layout(root, license_status="approved")

            with patch("irodori_mlx.hosted_weights.snapshot_weights_repo", return_value=root) as snapshot:
                resolved = resolve_weights_layout_source(weights_repo="t0yohei/Irodori-TTS-MLX-500M-v3", revision="abc123")

        snapshot.assert_called_once_with("t0yohei/Irodori-TTS-MLX-500M-v3", revision="abc123")
        self.assertEqual(resolved.source_kind, "repo")
        self.assertEqual(resolved.source, "t0yohei/Irodori-TTS-MLX-500M-v3@abc123")

    def test_hosted_repo_resolution_rejects_pending_license_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snapshot"
            _write_layout(root, license_status="pending")

            with patch("irodori_mlx.hosted_weights.snapshot_weights_repo", return_value=root), self.assertRaisesRegex(
                HostedWeightsError, "license_review.status='approved'"
            ):
                resolve_weights_layout_source(weights_repo="t0yohei/private-layout")


if __name__ == "__main__":
    unittest.main()
