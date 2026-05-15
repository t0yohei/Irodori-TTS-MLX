from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from irodori_mlx.hosted_weights import default_huggingface_snapshot_download, resolve_weights_source, validate_hosted_weights_layout


class HostedWeightsSmokeTests(unittest.TestCase):
    def _write_hosted_layout(
        self,
        root: Path,
        *,
        license_status: str = "approved",
        omit_file: str | None = None,
        omit_checksum_entry: str | None = None,
        corrupt_checksum_entry: str | None = None,
        file_overrides: dict[str, str] | None = None,
    ) -> None:
        files = {
            "weights": "weights.npz",
            "model_config": "model_config.json",
            "tokenizer_config": "tokenizer_config.json",
            "conversion_metadata": "conversion_metadata.json",
            "checksums": "checksums.sha256",
        }
        if file_overrides:
            files.update(file_overrides)
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
            "license_review": {
                "status": license_status,
                "review_reference": "https://github.com/t0yohei/irodori-tts-mlx/issues/80",
            },
        }
        payloads = {
            "irodori_mlx_manifest.json": json.dumps(manifest),
            files["model_config"]: '{"use_duration_predictor": true}',
            files["tokenizer_config"]: '{"schema_version": 1}',
            files["conversion_metadata"]: '{"schema_version": 1}',
            files["weights"]: b"tiny fake npz fixture; not a real model weight",
        }
        for name, payload in payloads.items():
            if name == omit_file:
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                path.write_text(payload, encoding="utf-8")
        listed = ["irodori_mlx_manifest.json", *(value for key, value in files.items() if key != "checksums")]
        if omit_checksum_entry:
            listed.remove(omit_checksum_entry)
        checksum_lines = []
        for name in listed:
            path = root / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "0" * 64
            if name == corrupt_checksum_entry:
                digest = "0" * 64
            checksum_lines.append(f"{digest}  {name}")
        checksum_path = root / files["checksums"]
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text("\n".join(checksum_lines), encoding="utf-8")

    def test_local_hosted_layout_discovers_weights_and_required_metadata_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="pending")

            resolved = validate_hosted_weights_layout(root, source_label="fixture")

        self.assertEqual(resolved.source_kind, "hosted-layout")
        self.assertTrue(str(resolved.weights_path).endswith("weights.npz"))
        self.assertTrue(str(resolved.model_config_path).endswith("model_config.json"))
        self.assertEqual(resolved.manifest["upstream_checkpoint"], "Aratako/Irodori-TTS-500M-v3")

    def test_hosted_weights_module_import_does_not_initialize_mlx(self):
        root = Path(__file__).resolve().parents[1]
        code = """
import importlib.abc
import sys

class BlockMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("hosted weights import should not initialize MLX")
        return None

sys.meta_path.insert(0, BlockMlx())
from irodori_mlx.hosted_weights import validate_hosted_weights_layout
print(validate_hosted_weights_layout.__name__)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.stdout.strip(), "validate_hosted_weights_layout")

    def test_repo_id_resolution_uses_download_abstraction_and_requires_approved_license(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="approved")
            calls: list[str] = []

            def fake_snapshot_download(repo_id: str) -> Path:
                calls.append(repo_id)
                return root

            resolved = resolve_weights_source(weights_repo="org/irodori-v3-mlx", snapshot_downloader=fake_snapshot_download)

        self.assertEqual(calls, ["org/irodori-v3-mlx"])
        self.assertEqual(resolved.source_kind, "hosted-layout")
        self.assertIn("weights.npz", str(resolved.weights_path))

    def test_huggingface_download_expands_allow_patterns_from_manifest_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(
                root,
                license_status="approved",
                file_overrides={
                    "weights": "artifacts/weights.npz",
                    "model_config": "artifacts/config/model_config.json",
                    "tokenizer_config": "artifacts/config/tokenizer_config.json",
                    "conversion_metadata": "artifacts/conversion_metadata.json",
                    "checksums": "artifacts/checksums.sha256",
                },
            )
            calls: list[tuple[str, list[str]]] = []
            model_info_calls: list[str] = []

            class FakeHfApi:
                def model_info(self, *, repo_id: str):
                    model_info_calls.append(repo_id)
                    return SimpleNamespace(sha="abc123")

            def fake_snapshot_download(*, repo_id: str, revision: str, allow_patterns: list[str]) -> str:
                self.assertEqual(repo_id, "org/irodori-v3-mlx")
                calls.append((revision, allow_patterns))
                return str(root)

            previous = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = SimpleNamespace(HfApi=FakeHfApi, snapshot_download=fake_snapshot_download)
            try:
                resolved = default_huggingface_snapshot_download("org/irodori-v3-mlx")
            finally:
                if previous is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = previous

        self.assertEqual(resolved, root)
        self.assertEqual(model_info_calls, ["org/irodori-v3-mlx"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ("abc123", ["README.md", "LICENSE.md", "irodori_mlx_manifest.json"]))
        self.assertEqual(calls[1][0], "abc123")
        self.assertIn("artifacts/weights.npz", calls[1][1])
        self.assertIn("artifacts/config/model_config.json", calls[1][1])
        self.assertIn("artifacts/checksums.sha256", calls[1][1])

    def test_huggingface_manifest_paths_reject_globs_before_second_download(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, file_overrides={"weights": "artifacts/*.npz"})
            calls: list[tuple[str, list[str]]] = []

            class FakeHfApi:
                def model_info(self, *, repo_id: str):
                    return SimpleNamespace(sha="abc123")

            def fake_snapshot_download(*, repo_id: str, revision: str, allow_patterns: list[str]) -> str:
                calls.append((revision, allow_patterns))
                return str(root)

            previous = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = SimpleNamespace(HfApi=FakeHfApi, snapshot_download=fake_snapshot_download)
            try:
                with self.assertRaisesRegex(ValueError, "glob metacharacters"):
                    default_huggingface_snapshot_download("org/irodori-v3-mlx")
            finally:
                if previous is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = previous

        self.assertEqual(calls, [("abc123", ["README.md", "LICENSE.md", "irodori_mlx_manifest.json"])])

    def test_huggingface_manifest_path_validation_ignores_cwd_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snapshot"
            root.mkdir()
            self._write_hosted_layout(root, file_overrides={"weights": "artifacts/weights.npz"})
            cwd = Path(td) / "cwd"
            cwd.mkdir()
            (cwd / "artifacts").symlink_to(Path(td) / "outside")
            calls: list[tuple[str, list[str]]] = []

            class FakeHfApi:
                def model_info(self, *, repo_id: str):
                    return SimpleNamespace(sha="abc123")

            def fake_snapshot_download(*, repo_id: str, revision: str, allow_patterns: list[str]) -> str:
                calls.append((revision, allow_patterns))
                return str(root)

            previous_module = sys.modules.get("huggingface_hub")
            previous_cwd = Path.cwd()
            sys.modules["huggingface_hub"] = SimpleNamespace(HfApi=FakeHfApi, snapshot_download=fake_snapshot_download)
            try:
                os.chdir(cwd)
                resolved = default_huggingface_snapshot_download("org/irodori-v3-mlx")
            finally:
                os.chdir(previous_cwd)
                if previous_module is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = previous_module

        self.assertEqual(resolved, root)
        self.assertEqual(len(calls), 2)
        self.assertIn("artifacts/weights.npz", calls[1][1])

    def test_repo_id_resolution_rejects_unapproved_artifacts_with_local_fallback_hint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, license_status="pending")

            with self.assertRaisesRegex(ValueError, "fallback to locally converted .npz weights"):
                resolve_weights_source(weights_repo="org/unapproved", snapshot_downloader=lambda _repo: root)

    def test_direct_local_npz_fallback_does_not_require_hosted_metadata(self):
        resolved = resolve_weights_source(weights="/tmp/local-converted.npz")

        self.assertEqual(resolved.source_kind, "local-npz")
        self.assertEqual(resolved.source_label, "local converted .npz fallback")
        self.assertIsNone(resolved.model_config_path)
        self.assertIsNone(resolved.manifest)

    def test_missing_layout_file_identifies_component_and_mentions_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, omit_file="tokenizer_config.json")

            with self.assertRaisesRegex(ValueError, "tokenizer_config.json.*Fallback"):
                validate_hosted_weights_layout(root, source_label="fixture")

    def test_checksum_manifest_must_list_weights_and_metadata_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, omit_checksum_entry="weights.npz")

            with self.assertRaisesRegex(ValueError, "checksums file does not list required files: weights.npz"):
                validate_hosted_weights_layout(root, source_label="fixture")

    def test_manifest_file_entries_must_stay_inside_hosted_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, file_overrides={"weights": "../outside.npz"})

            with self.assertRaisesRegex(ValueError, "must stay inside the hosted weights layout"):
                validate_hosted_weights_layout(root, source_label="fixture")

    def test_checksum_manifest_must_match_required_file_digests(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(root, corrupt_checksum_entry="weights.npz")

            with self.assertRaisesRegex(ValueError, "mismatched sha256 digests: weights.npz"):
                validate_hosted_weights_layout(root, source_label="fixture")

    def test_manifest_checksum_always_covers_canonical_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_hosted_layout(
                root,
                file_overrides={"manifest": "alternate_manifest.json"},
                corrupt_checksum_entry="irodori_mlx_manifest.json",
            )
            (root / "alternate_manifest.json").write_text('{"ignored": true}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mismatched sha256 digests: irodori_mlx_manifest.json"):
                validate_hosted_weights_layout(root, source_label="fixture")

    def test_only_one_source_may_be_selected(self):
        with self.assertRaisesRegex(ValueError, "choose only one weights source"):
            resolve_weights_source(weights="local.npz", weights_repo="org/repo")


if __name__ == "__main__":
    unittest.main()
