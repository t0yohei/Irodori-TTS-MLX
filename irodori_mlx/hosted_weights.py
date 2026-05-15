from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig

MANIFEST_NAME = "irodori_mlx_manifest.json"
REQUIRED_MANIFEST_FILES = {
    "weights",
    "model_config",
    "tokenizer_config",
    "conversion_metadata",
    "checksums",
}
EXPECTED_TOP_LEVEL_FILES = {
    "weights": "weights.npz",
    "model_config": "model_config.json",
    "tokenizer_config": "tokenizer_config.json",
    "conversion_metadata": "conversion_metadata.json",
    "checksums": "checksums.sha256",
}
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_FORMAT = "irodori-tts-mlx-weights"
SUPPORTED_FORMAT_VERSION = "0.2"
SUPPORTED_FAMILIES = {"base_v2", "voicedesign", "v3"}


class HostedWeightsError(ValueError):
    """Raised when a hosted/local converted weights layout cannot be used safely."""


@dataclass(frozen=True)
class ResolvedWeightsLayout:
    """Validated local snapshot of an irodori-tts-mlx converted weights layout."""

    root: Path
    weights_path: Path
    model_config_path: Path
    tokenizer_config_path: Path
    conversion_metadata_path: Path
    manifest_path: Path
    checksums_path: Path
    manifest: dict[str, Any]
    tokenizer_config: dict[str, Any]
    conversion_metadata: dict[str, Any]
    model_config: ModelConfig
    source: str
    source_kind: str
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = None


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HostedWeightsError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HostedWeightsError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HostedWeightsError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_model_config_json(path: Path) -> ModelConfig:
    """Load ModelConfig without importing the MLX runtime module."""

    payload = _read_json_object(path, label="model_config.json")
    try:
        return ModelConfig(**payload)
    except TypeError as exc:
        raise HostedWeightsError(
            "model_config.json contains unsupported keys for irodori_mlx.config.ModelConfig"
        ) from exc
    except ValueError as exc:
        raise HostedWeightsError(f"model_config.json is not supported by the MLX runtime: {exc}") from exc


def _require_mapping(payload: dict[str, Any], key: str, *, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise HostedWeightsError(f"{label}.{key} must be a JSON object")
    return value


def _relative_file(root: Path, value: object, *, manifest_key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HostedWeightsError(f"manifest files.{manifest_key} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise HostedWeightsError(f"manifest files.{manifest_key} must stay inside the weights layout: {value!r}")
    return root / candidate


def _parse_checksum_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise HostedWeightsError(f"checksum file is missing: {path}") from exc
    checksums: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise HostedWeightsError(f"invalid checksum line {index} in {path}: expected '<sha256> <file>'")
        digest, filename = parts
        filename = filename.lstrip("*").strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise HostedWeightsError(f"invalid sha256 digest for {filename!r} in {path}")
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
    checksum_targets["manifest"] = MANIFEST_NAME
    missing = sorted(set(checksum_targets.values()) - set(checksums))
    if missing:
        raise HostedWeightsError("checksums.sha256 does not name required files: " + ", ".join(missing))
    for role, rel_path in checksum_targets.items():
        target = root / rel_path
        expected = checksums.get(rel_path)
        if expected is None:
            continue
        actual = _sha256_file(target)
        if actual != expected:
            raise HostedWeightsError(f"checksum mismatch for {rel_path}: expected {expected}, got {actual}")


def _validate_manifest(manifest: dict[str, Any], *, source_kind: str) -> dict[str, str]:
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise HostedWeightsError("manifest schema_version must be 1")
    if manifest.get("format") != SUPPORTED_FORMAT:
        raise HostedWeightsError(f"manifest format must be {SUPPORTED_FORMAT!r}")
    if manifest.get("format_version") != SUPPORTED_FORMAT_VERSION:
        raise HostedWeightsError(f"manifest format_version must be {SUPPORTED_FORMAT_VERSION!r}")
    family = manifest.get("family")
    if family not in SUPPORTED_FAMILIES:
        allowed = ", ".join(sorted(SUPPORTED_FAMILIES))
        raise HostedWeightsError(f"manifest family must be one of: {allowed}")
    files = _require_mapping(manifest, "files", label="manifest")
    missing_file_keys = sorted(REQUIRED_MANIFEST_FILES - set(files))
    if missing_file_keys:
        raise HostedWeightsError("manifest files is missing required keys: " + ", ".join(missing_file_keys))
    manifest_files = {key: files[key] for key in REQUIRED_MANIFEST_FILES}
    if any(not isinstance(value, str) or not value.strip() for value in manifest_files.values()):
        raise HostedWeightsError("manifest file entries must be non-empty strings")
    for key, expected in EXPECTED_TOP_LEVEL_FILES.items():
        if manifest_files[key] != expected:
            raise HostedWeightsError(f"manifest files.{key} must be top-level {expected!r}")
    runtime = _require_mapping(manifest, "runtime", label="manifest")
    for key in ("requires_upstream_dacvae_bridge", "requires_reference_audio", "supports_no_reference", "supports_caption", "supports_predicted_duration"):
        if not isinstance(runtime.get(key), bool):
            raise HostedWeightsError(f"manifest runtime.{key} must be a boolean")
    license_review = _require_mapping(manifest, "license_review", label="manifest")
    status = license_review.get("status")
    if source_kind == "repo" and status != "approved":
        raise HostedWeightsError("hosted weights repos require manifest license_review.status='approved'")
    if source_kind == "local" and status not in {"approved", "pending"}:
        raise HostedWeightsError("local weights layouts require license_review.status to be 'approved' or 'pending'")
    return manifest_files


def _validate_tokenizer_config(tokenizer_config: dict[str, Any], *, model_config: ModelConfig) -> None:
    if tokenizer_config.get("schema_version") != 1:
        raise HostedWeightsError("tokenizer_config.json schema_version must be 1")
    text_tokenizer = tokenizer_config.get("text_tokenizer")
    if not isinstance(text_tokenizer, dict):
        raise HostedWeightsError("tokenizer_config.json must include text_tokenizer metadata")
    if model_config.use_caption_condition and not isinstance(tokenizer_config.get("caption_tokenizer"), dict):
        raise HostedWeightsError("caption-conditioned layouts require caption_tokenizer metadata")


def _validate_conversion_metadata(conversion_metadata: dict[str, Any], *, manifest: dict[str, Any]) -> None:
    if conversion_metadata.get("schema_version") != 1:
        raise HostedWeightsError("conversion_metadata.json schema_version must be 1")
    if conversion_metadata.get("detected_family") != manifest.get("family"):
        raise HostedWeightsError("conversion_metadata.json detected_family must match manifest family")
    if not isinstance(conversion_metadata.get("converter"), dict):
        raise HostedWeightsError("conversion_metadata.json must include converter provenance")
    if not isinstance(conversion_metadata.get("upstream"), dict):
        raise HostedWeightsError("conversion_metadata.json must include upstream provenance")


def _validate_runtime_flags(manifest: dict[str, Any], *, model_config: ModelConfig) -> None:
    runtime = _require_mapping(manifest, "runtime", label="manifest")
    if bool(runtime["supports_caption"]) != bool(model_config.use_caption_condition):
        raise HostedWeightsError("manifest runtime.supports_caption must match model_config.use_caption_condition")
    if bool(runtime["supports_predicted_duration"]) != bool(model_config.use_duration_predictor):
        raise HostedWeightsError("manifest runtime.supports_predicted_duration must match model_config.use_duration_predictor")
    if bool(runtime["requires_reference_audio"]) and bool(runtime["supports_no_reference"]):
        raise HostedWeightsError("manifest cannot both require reference audio and support no-reference generation")


def _archive_members_safe(names: list[str]) -> None:
    for name in names:
        member_path = Path(name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise HostedWeightsError(f"archive member must stay inside the weights layout: {name!r}")


def _find_extracted_layout_root(extract_root: Path) -> Path:
    if (extract_root / MANIFEST_NAME).is_file():
        return extract_root
    children = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / MANIFEST_NAME).is_file():
        return children[0]
    raise HostedWeightsError(
        f"converted weights archive must contain {MANIFEST_NAME} at the archive root or in one top-level directory"
    )


def _extract_weights_archive(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    archive_path = path.expanduser().resolve(strict=False)
    if not archive_path.is_file():
        raise HostedWeightsError(f"converted weights layout is not a directory or archive: {archive_path}")
    tmp = tempfile.TemporaryDirectory(prefix="irodori-mlx-weights-")
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
                        raise HostedWeightsError("converted weights archives must contain only regular files and directories")
                archive.extractall(extract_root, members=members)
        else:
            raise HostedWeightsError(f"unsupported converted weights archive type: {archive_path}")
        return _find_extracted_layout_root(extract_root), tmp
    except Exception:
        tmp.cleanup()
        raise


def validate_weights_layout(root: str | Path, *, source: str | None = None, source_kind: str = "local") -> ResolvedWeightsLayout:
    """Validate a local converted weights directory and return resolved runtime inputs."""

    layout_root = Path(root).expanduser().resolve(strict=False)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if layout_root.is_file():
        layout_root, temporary_directory = _extract_weights_archive(layout_root)
    if not layout_root.is_dir():
        raise HostedWeightsError(f"converted weights layout is not a directory or archive: {layout_root}")
    manifest_path = layout_root / MANIFEST_NAME
    manifest = _read_json_object(manifest_path, label=MANIFEST_NAME)
    manifest_files = _validate_manifest(manifest, source_kind=source_kind)
    paths = {key: _relative_file(layout_root, value, manifest_key=key) for key, value in manifest_files.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise HostedWeightsError(f"required layout file for {key!r} is missing: {path}")
    model_config = _load_model_config_json(paths["model_config"])
    tokenizer_config = _read_json_object(paths["tokenizer_config"], label="tokenizer_config.json")
    conversion_metadata = _read_json_object(paths["conversion_metadata"], label="conversion_metadata.json")
    _validate_tokenizer_config(tokenizer_config, model_config=model_config)
    _validate_conversion_metadata(conversion_metadata, manifest=manifest)
    _validate_runtime_flags(manifest, model_config=model_config)
    _validate_checksum_coverage(root=layout_root, checksums_path=paths["checksums"], manifest_files=manifest_files)
    return ResolvedWeightsLayout(
        root=layout_root,
        weights_path=paths["weights"],
        model_config_path=paths["model_config"],
        tokenizer_config_path=paths["tokenizer_config"],
        conversion_metadata_path=paths["conversion_metadata"],
        manifest_path=manifest_path,
        checksums_path=paths["checksums"],
        manifest=manifest,
        tokenizer_config=tokenizer_config,
        conversion_metadata=conversion_metadata,
        model_config=model_config,
        source=source or str(root),
        source_kind=source_kind,
        _temporary_directory=temporary_directory,
    )


def snapshot_weights_repo(repo_id: str, *, revision: str | None = None) -> Path:
    """Download a Hugging Face weights repo snapshot using only required layout files."""

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise HostedWeightsError(
            "huggingface_hub is required for --weights-repo. Install the runtime/bench dependency or pass "
            "--weights-dir for a local converted layout."
        ) from exc
    try:
        pinned_revision = revision
        if pinned_revision is None:
            model_info = HfApi().model_info(repo_id=repo_id)
            pinned_revision = getattr(model_info, "sha", None)
            if not isinstance(pinned_revision, str) or not pinned_revision.strip():
                raise HostedWeightsError(f"could not determine a pinned revision for hosted weights repo {repo_id!r}")
        manifest_path = Path(hf_hub_download(repo_id=repo_id, filename=MANIFEST_NAME, revision=pinned_revision))
        manifest = _read_json_object(manifest_path, label=MANIFEST_NAME)
        license_review = _require_mapping(manifest, "license_review", label="manifest")
        if license_review.get("status") != "approved":
            raise HostedWeightsError("hosted weights repos require manifest license_review.status='approved'")
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=pinned_revision,
                allow_patterns=[
                    "README.md",
                    "LICENSE.md",
                    MANIFEST_NAME,
                    "model_config.json",
                    "tokenizer_config.json",
                    "conversion_metadata.json",
                    "weights.npz",
                    "checksums.sha256",
                ],
            )
        )
    except Exception as exc:
        rev = f" at revision {revision!r}" if revision else ""
        raise HostedWeightsError(f"could not resolve hosted weights repo {repo_id!r}{rev}: {exc}") from exc


def resolve_weights_layout_source(
    *,
    weights_dir: str | Path | None = None,
    weights_repo: str | None = None,
    revision: str | None = None,
) -> ResolvedWeightsLayout | None:
    """Resolve either a local layout directory or a hosted Hugging Face repo layout."""

    if weights_dir and weights_repo:
        raise HostedWeightsError("choose either weights_dir or weights_repo, not both")
    if weights_dir:
        return validate_weights_layout(weights_dir, source=str(weights_dir), source_kind="local")
    if weights_repo:
        snapshot = snapshot_weights_repo(str(weights_repo), revision=revision)
        source = str(weights_repo) if revision is None else f"{weights_repo}@{revision}"
        return validate_weights_layout(snapshot, source=source, source_kind="repo")
    return None
