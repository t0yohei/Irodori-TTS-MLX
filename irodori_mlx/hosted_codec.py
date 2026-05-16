from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CODEC_MANIFEST_NAME = "irodori_dacvae_codec_manifest.json"
REQUIRED_CODEC_MANIFEST_FILES = {"codec", "metadata", "checksums"}
SUPPORTED_CODEC_SCHEMA_VERSION = 1
SUPPORTED_CODEC_FORMAT = "irodori-tts-mlx-dacvae-codec"
SUPPORTED_CODEC_FORMAT_VERSION = "0.2"
HOSTED_CODEC_ALLOW_PATTERNS = (
    "README.md",
    "LICENSE.md",
    CODEC_MANIFEST_NAME,
)
MANIFEST_GLOB_METACHARACTERS = frozenset("*?[]")


class HostedCodecError(ValueError):
    """Raised when a hosted/local DACVAE codec artifact layout is unsafe."""


@dataclass(frozen=True)
class ResolvedCodecArtifact:
    """Validated local snapshot of a DACVAE codec artifact layout."""

    root: Path
    codec_path: Path
    metadata_path: Path
    manifest_path: Path
    checksums_path: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    source: str
    source_kind: str
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = None


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HostedCodecError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HostedCodecError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HostedCodecError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_mapping(payload: Mapping[str, Any], key: str, *, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise HostedCodecError(f"{label}.{key} must be a JSON object")
    return dict(value)


def _manifest_relative_path(entry: str, *, label: str) -> Path:
    if any(char in entry for char in MANIFEST_GLOB_METACHARACTERS):
        raise HostedCodecError(f"{label} manifest file path must not contain glob metacharacters: {entry}")
    relative = Path(entry)
    if relative.is_absolute() or ".." in relative.parts:
        raise HostedCodecError(f"{label} manifest file path must stay inside the hosted codec layout: {entry}")
    return relative


def _relative_file(root: Path, value: object, *, manifest_key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HostedCodecError(f"manifest files.{manifest_key} must be a non-empty relative path")
    return root / _manifest_relative_path(value, label=f"manifest files.{manifest_key}")


def _parse_checksum_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise HostedCodecError(f"checksum file is missing: {path}") from exc
    checksums: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise HostedCodecError(f"invalid checksum line {index} in {path}: expected '<sha256> <file>'")
        digest, filename = parts
        filename = filename.lstrip("*").strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise HostedCodecError(f"invalid sha256 digest for {filename!r} in {path}")
        checksums[filename] = digest.lower()
    return checksums


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum_coverage(*, root: Path, checksums_path: Path, manifest_files: dict[str, str]) -> None:
    checksums = _parse_checksum_file(checksums_path)
    checksum_targets = {role: rel_path for role, rel_path in manifest_files.items() if role != "checksums"}
    checksum_targets["manifest"] = CODEC_MANIFEST_NAME
    missing = sorted(set(checksum_targets.values()) - set(checksums))
    if missing:
        raise HostedCodecError("checksums.sha256 does not name required codec files: " + ", ".join(missing))
    for rel_path in checksum_targets.values():
        target = root / rel_path
        expected = checksums.get(rel_path)
        if expected is None:
            continue
        actual = _sha256_file(target)
        if actual != expected:
            raise HostedCodecError(f"checksum mismatch for {rel_path}: expected {expected}, got {actual}")


def _validate_manifest(manifest: dict[str, Any], *, source_kind: str) -> dict[str, str]:
    if manifest.get("schema_version") != SUPPORTED_CODEC_SCHEMA_VERSION:
        raise HostedCodecError("codec manifest schema_version must be 1")
    if manifest.get("artifact_format") != SUPPORTED_CODEC_FORMAT:
        raise HostedCodecError(f"codec manifest artifact_format must be {SUPPORTED_CODEC_FORMAT!r}")
    if manifest.get("artifact_format_version") != SUPPORTED_CODEC_FORMAT_VERSION:
        raise HostedCodecError(f"codec manifest artifact_format_version must be {SUPPORTED_CODEC_FORMAT_VERSION!r}")
    files = _require_mapping(manifest, "files", label="codec manifest")
    missing_file_keys = sorted(REQUIRED_CODEC_MANIFEST_FILES - set(files))
    if missing_file_keys:
        raise HostedCodecError("codec manifest files is missing required keys: " + ", ".join(missing_file_keys))
    manifest_files = {key: files[key] for key in REQUIRED_CODEC_MANIFEST_FILES}
    for key, entry in manifest_files.items():
        if not isinstance(entry, str) or not entry.strip():
            raise HostedCodecError("codec manifest file entries must be non-empty strings")
        _manifest_relative_path(entry, label=f"codec manifest files.{key}")
    codec = _require_mapping(manifest, "codec", label="codec manifest")
    for key, expected in (("sample_rate", 48000), ("hop_length", 512), ("latent_dim", 32)):
        try:
            actual = int(codec.get(key, -1))
        except (TypeError, ValueError) as exc:
            raise HostedCodecError(f"codec manifest codec.{key} must be {expected}") from exc
        if actual != expected:
            raise HostedCodecError(f"codec manifest codec.{key} must be {expected}")
    if codec.get("source_repo") != "Aratako/Semantic-DACVAE-Japanese-32dim":
        raise HostedCodecError("codec manifest codec.source_repo must be Aratako/Semantic-DACVAE-Japanese-32dim")
    if codec.get("source_file") != "weights.pth":
        raise HostedCodecError("codec manifest codec.source_file must be weights.pth")
    runtime = _require_mapping(manifest, "runtime", label="codec manifest")
    for key in ("supports_mlx_decode", "supports_mlx_encode", "requires_pytorch_fallback"):
        if not isinstance(runtime.get(key), bool):
            raise HostedCodecError(f"codec manifest runtime.{key} must be a boolean")
    license_review = _require_mapping(manifest, "license_review", label="codec manifest")
    status = license_review.get("status")
    if source_kind == "repo" and status != "approved":
        raise HostedCodecError("hosted codec repos require manifest license_review.status='approved'")
    if source_kind == "local" and status not in {"approved", "pending"}:
        raise HostedCodecError("local codec layouts require license_review.status to be 'approved' or 'pending'")
    return manifest_files


def _validate_metadata(metadata: dict[str, Any], *, manifest: dict[str, Any]) -> None:
    if metadata.get("schema_version") != 1:
        raise HostedCodecError("codec_metadata.json schema_version must be 1")
    if metadata.get("artifact_format") != manifest.get("artifact_format"):
        raise HostedCodecError("codec_metadata.json artifact_format must match the manifest")
    if metadata.get("artifact_format_version") != manifest.get("artifact_format_version"):
        raise HostedCodecError("codec_metadata.json artifact_format_version must match the manifest")
    if not isinstance(metadata.get("provenance"), dict):
        raise HostedCodecError("codec_metadata.json must include provenance")
    if not isinstance(metadata.get("validation"), dict):
        raise HostedCodecError("codec_metadata.json must include validation evidence")


def _archive_members_safe(names: list[str]) -> None:
    for name in names:
        member_path = Path(name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise HostedCodecError(f"archive member must stay inside the codec layout: {name!r}")


def _find_extracted_layout_root(extract_root: Path) -> Path:
    if (extract_root / CODEC_MANIFEST_NAME).is_file():
        return extract_root
    children = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / CODEC_MANIFEST_NAME).is_file():
        return children[0]
    raise HostedCodecError(
        f"hosted codec archive must contain {CODEC_MANIFEST_NAME} at the archive root or in one top-level directory"
    )


def _extract_codec_archive(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    archive_path = path.expanduser().resolve(strict=False)
    if not archive_path.is_file():
        raise HostedCodecError(f"codec layout is not a directory or archive: {archive_path}")
    tmp = tempfile.TemporaryDirectory(prefix="irodori-mlx-codec-")
    extract_root = Path(tmp.name)
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                _archive_members_safe(archive.namelist())
                archive.extractall(extract_root)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                members = archive.getmembers()
                _archive_members_safe([member.name for member in members])
                for member in members:
                    if not (member.isfile() or member.isdir()):
                        raise HostedCodecError("codec archives must contain only regular files and directories")
                archive.extractall(extract_root, members=members)
        else:
            raise HostedCodecError(f"unsupported codec archive type: {archive_path}")
        return _find_extracted_layout_root(extract_root), tmp
    except Exception:
        tmp.cleanup()
        raise


def validate_codec_artifact_layout(
    root: str | Path,
    *,
    source: str | None = None,
    source_kind: str = "local",
) -> ResolvedCodecArtifact:
    """Validate a local/hosted DACVAE codec artifact layout."""

    layout_root = Path(root).expanduser().resolve(strict=False)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if layout_root.is_file():
        layout_root, temporary_directory = _extract_codec_archive(layout_root)
    if not layout_root.is_dir():
        raise HostedCodecError(f"codec layout is not a directory or archive: {layout_root}")
    manifest_path = layout_root / CODEC_MANIFEST_NAME
    manifest = _read_json_object(manifest_path, label=CODEC_MANIFEST_NAME)
    manifest_files = _validate_manifest(manifest, source_kind=source_kind)
    paths = {key: _relative_file(layout_root, value, manifest_key=key) for key, value in manifest_files.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise HostedCodecError(f"required codec layout file for {key!r} is missing: {path}")
    metadata = _read_json_object(paths["metadata"], label="codec_metadata.json")
    _validate_metadata(metadata, manifest=manifest)
    _validate_checksum_coverage(root=layout_root, checksums_path=paths["checksums"], manifest_files=manifest_files)
    return ResolvedCodecArtifact(
        root=layout_root,
        codec_path=paths["codec"],
        metadata_path=paths["metadata"],
        manifest_path=manifest_path,
        checksums_path=paths["checksums"],
        manifest=manifest,
        metadata=metadata,
        source=source or str(root),
        source_kind=source_kind,
        _temporary_directory=temporary_directory,
    )


def _hosted_codec_allow_patterns_from_manifest(manifest: Mapping[str, Any], *, label: str) -> list[str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise HostedCodecError(f"{label} manifest must include a files object")
    normalized_files = dict(files)
    normalized_files["manifest"] = CODEC_MANIFEST_NAME
    allow_patterns = [*HOSTED_CODEC_ALLOW_PATTERNS]
    for key in ("manifest", *sorted(REQUIRED_CODEC_MANIFEST_FILES)):
        entry = normalized_files.get(key)
        if not isinstance(entry, str) or not entry.strip():
            raise HostedCodecError(f"{label} manifest is missing file entries: {key}")
        _manifest_relative_path(entry, label=label)
        allow_patterns.append(entry)
    return list(dict.fromkeys(allow_patterns))


def snapshot_codec_repo(repo_id: str, *, revision: str | None = None) -> Path:
    """Download a Hugging Face codec artifact repo snapshot using only required files."""

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise HostedCodecError(
            "huggingface_hub is required for --codec-artifact-repo. Install it or pass "
            "--codec-artifact-dir / --codec-path for a local DACVAE codec artifact."
        ) from exc
    try:
        pinned_revision = revision
        if pinned_revision is None:
            model_info = HfApi().model_info(repo_id=repo_id)
            pinned_revision = getattr(model_info, "sha", None)
            if not isinstance(pinned_revision, str) or not pinned_revision.strip():
                raise HostedCodecError(f"could not determine a pinned revision for hosted codec repo {repo_id!r}")
        manifest_path = Path(hf_hub_download(repo_id=repo_id, filename=CODEC_MANIFEST_NAME, revision=pinned_revision))
        manifest = _read_json_object(manifest_path, label=CODEC_MANIFEST_NAME)
        license_review = _require_mapping(manifest, "license_review", label="codec manifest")
        if license_review.get("status") != "approved":
            raise HostedCodecError("hosted codec repos require manifest license_review.status='approved'")
        allow_patterns = _hosted_codec_allow_patterns_from_manifest(manifest, label=f"hosted codec repo {repo_id!r}")
        return Path(snapshot_download(repo_id=repo_id, revision=pinned_revision, allow_patterns=allow_patterns))
    except Exception as exc:
        rev = f" at revision {revision!r}" if revision else ""
        raise HostedCodecError(f"could not resolve hosted codec repo {repo_id!r}{rev}: {exc}") from exc


def resolve_codec_artifact_source(
    *,
    codec_artifact_dir: str | Path | None = None,
    codec_artifact_repo: str | None = None,
    revision: str | None = None,
) -> ResolvedCodecArtifact | None:
    """Resolve either a local codec layout directory/archive or a hosted Hugging Face repo."""

    if codec_artifact_dir and codec_artifact_repo:
        raise HostedCodecError("choose either codec_artifact_dir or codec_artifact_repo, not both")
    if codec_artifact_dir:
        return validate_codec_artifact_layout(codec_artifact_dir, source=str(codec_artifact_dir), source_kind="local")
    if codec_artifact_repo:
        snapshot = snapshot_codec_repo(str(codec_artifact_repo), revision=revision)
        source = str(codec_artifact_repo) if revision is None else f"{codec_artifact_repo}@{revision}"
        return validate_codec_artifact_layout(snapshot, source=source, source_kind="repo")
    return None
