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

from irodori_mlx.hosted_codec import HostedCodecError, resolve_codec_artifact_source, validate_codec_artifact_layout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_codec_layout(root: Path, *, license_status: str = "pending", file_prefix: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_prefix.rstrip('/')}/" if file_prefix else ""
    codec_path = root / prefix / "dacvae-codec.npz"
    codec_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        codec_path,
        sample_rate=np.array(48000, dtype=np.int64),
        hop_length=np.array(1920, dtype=np.int64),
        latent_dim=np.array(32, dtype=np.int64),
        decode_basis=np.zeros((32, 1920), dtype=np.float32),
        decode_bias=np.zeros((1920,), dtype=np.float32),
        metadata_json=np.array(
            json.dumps(
                {
                    "artifact_kind": "linear-fixture",
                    "sample_rate": 48000,
                    "hop_length": 1920,
                    "latent_dim": 32,
                }
            )
        ),
    )
    metadata = {
        "schema_version": 1,
        "artifact_format": "irodori-tts-mlx-dacvae-codec",
        "artifact_format_version": "0.2",
        "provenance": {
            "source_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "source_file": "weights.pth",
            "source_revision": "test-revision",
            "converter_repository": "https://github.com/t0yohei/Irodori-TTS-MLX",
            "converter_version": "git:test",
        },
        "validation": {
            "contract": "transport-only",
            "decode_parity": "not-applicable-to-fixture",
            "encode_parity": "not-applicable-to-fixture",
        },
    }
    (root / prefix / "codec_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_format": "irodori-tts-mlx-dacvae-codec",
        "artifact_format_version": "0.2",
        "files": {
            "codec": f"{prefix}dacvae-codec.npz",
            "metadata": f"{prefix}codec_metadata.json",
            "checksums": f"{prefix}checksums.sha256",
        },
        "codec": {
            "source_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "source_revision": "test-revision",
            "source_file": "weights.pth",
            "artifact_kind": "semantic-dacvae",
            "sample_rate": 48000,
            "hop_length": 1920,
            "latent_dim": 32,
        },
        "runtime": {
            "minimum_irodori_tts_mlx_version": "0.2.0",
            "supports_mlx_decode": True,
            "supports_mlx_encode": False,
            "requires_pytorch_fallback": True,
        },
        "license_review": {"status": license_status, "review_reference": "local-test"},
    }
    (root / "irodori_dacvae_codec_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksum_lines = []
    for filename in [
        f"{prefix}dacvae-codec.npz",
        f"{prefix}codec_metadata.json",
        "irodori_dacvae_codec_manifest.json",
    ]:
        checksum_lines.append(f"{_sha256(root / filename)}  {filename}")
    (root / prefix / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


class HostedCodecArtifactTests(unittest.TestCase):
    def test_validates_local_codec_artifact_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "codec"
            _write_codec_layout(root)

            resolved = validate_codec_artifact_layout(root)

        self.assertEqual(resolved.codec_path.name, "dacvae-codec.npz")
        self.assertEqual(resolved.metadata["artifact_format"], "irodori-tts-mlx-dacvae-codec")
        self.assertEqual(resolved.source_kind, "local")

    def test_validates_archive_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "codec"
            _write_codec_layout(root)
            archive_path = Path(td) / "codec.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in root.iterdir():
                    archive.write(path, arcname=f"codec/{path.name}")

            resolved = validate_codec_artifact_layout(archive_path)

        self.assertEqual(resolved.codec_path.name, "dacvae-codec.npz")
        self.assertTrue(resolved.root.is_dir())

    def test_rejects_tar_archives_with_special_members(self):
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "codec.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("codec/special.fifo")
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)

            with self.assertRaisesRegex(HostedCodecError, "regular files and directories"):
                validate_codec_artifact_layout(archive_path)

    def test_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "codec"
            _write_codec_layout(root)
            with (root / "codec_metadata.json").open("a", encoding="utf-8") as fh:
                fh.write("\n")

            with self.assertRaisesRegex(HostedCodecError, "checksum mismatch for codec_metadata.json"):
                validate_codec_artifact_layout(root)

    def test_accepts_manifest_defined_relative_file_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "codec"
            _write_codec_layout(root, file_prefix="artifacts")

            resolved = validate_codec_artifact_layout(root)

        self.assertEqual(resolved.codec_path.parent.name, "artifacts")
        self.assertEqual(resolved.metadata_path.parent.name, "artifacts")

    def test_hosted_repo_resolution_requires_approved_license(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snapshot"
            _write_codec_layout(root, license_status="approved")

            with patch("irodori_mlx.hosted_codec.snapshot_codec_repo", return_value=root) as snapshot:
                resolved = resolve_codec_artifact_source(codec_artifact_repo="t0yohei/Irodori-TTS-MLX-DACVAE-Codec", revision="abc123")

        snapshot.assert_called_once_with("t0yohei/Irodori-TTS-MLX-DACVAE-Codec", revision="abc123")
        self.assertEqual(resolved.source_kind, "repo")
        self.assertEqual(resolved.source, "t0yohei/Irodori-TTS-MLX-DACVAE-Codec@abc123")

    def test_hosted_repo_resolution_rejects_pending_license_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snapshot"
            _write_codec_layout(root, license_status="pending")

            with patch("irodori_mlx.hosted_codec.snapshot_codec_repo", return_value=root), self.assertRaisesRegex(
                HostedCodecError, "license_review.status='approved'"
            ):
                resolve_codec_artifact_source(codec_artifact_repo="t0yohei/private-codec")

    def test_rejects_non_numeric_codec_dimensions_as_hosted_codec_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "codec"
            _write_codec_layout(root)
            manifest_path = root / "irodori_dacvae_codec_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["codec"]["sample_rate"] = None
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(HostedCodecError, "codec manifest codec.sample_rate must be 48000"):
                validate_codec_artifact_layout(root)


if __name__ == "__main__":
    unittest.main()
