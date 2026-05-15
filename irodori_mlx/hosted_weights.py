from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Union


HOSTED_WEIGHTS_MANIFEST = "irodori_mlx_manifest.json"
HOSTED_WEIGHTS_REQUIRED_FILES = (
    "manifest",
    "weights",
    "model_config",
    "tokenizer_config",
    "conversion_metadata",
    "checksums",
)
HOSTED_WEIGHTS_ALLOW_PATTERNS = (
    "README.md",
    "LICENSE.md",
    HOSTED_WEIGHTS_MANIFEST,
)
HOSTED_WEIGHTS_GLOB_METACHARACTERS = frozenset("*?[]")


@dataclass(frozen=True)
class WeightsSourceResolution:
    """Resolved runtime inputs for either hosted-layout or direct local weights."""

    weights_path: Path
    model_config_path: Path | None
    layout_dir: Path | None
    source_label: str
    source_kind: str
    manifest: Mapping[str, Any] | None = None


SnapshotDownloader = Callable[[str], Union[str, Path]]


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _manifest_relative_path(entry: str, *, label: str) -> Path:
    if any(char in entry for char in HOSTED_WEIGHTS_GLOB_METACHARACTERS):
        raise ValueError(f"{label} manifest file path must not contain glob metacharacters: {entry}")
    relative = Path(entry)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} manifest file path must stay inside the hosted weights layout: {entry}")
    return relative


def _layout_file_path(root: Path, entry: str, *, label: str) -> Path:
    return root / _manifest_relative_path(entry, label=label)


def _hosted_weights_allow_patterns_from_manifest(manifest: Mapping[str, Any], *, label: str) -> list[str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{label} manifest must include a files object")

    normalized_files = dict(files)
    normalized_files["manifest"] = HOSTED_WEIGHTS_MANIFEST
    allow_patterns = [*HOSTED_WEIGHTS_ALLOW_PATTERNS]
    for key in HOSTED_WEIGHTS_REQUIRED_FILES:
        entry = normalized_files.get(key)
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"{label} manifest is missing file entries: {key}")
        _layout_file_path(Path("."), entry, label=label)
        allow_patterns.append(entry)
    return list(dict.fromkeys(allow_patterns))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha256sum_lines(payload: str, *, label: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{label} checksums file has invalid sha256 entry on line {line_number}")
        digest, filename = parts
        if not all(char in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError(f"{label} checksums file has invalid sha256 entry on line {line_number}")
        entries[filename.lstrip("*")] = digest.lower()
    return entries


def default_huggingface_snapshot_download(repo_id: str) -> Path:
    """Download a hosted weights snapshot without requiring the dependency at import time."""

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional user setup.
        raise ValueError(
            "Hosted pre-converted MLX weights by repo id require huggingface_hub. Install it, use a local "
            "hosted-layout directory, or fall back to locally converted .npz weights."
        ) from exc
    try:
        repo_info = HfApi().model_info(repo_id=repo_id)
        revision = repo_info.sha
        manifest_snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=list(HOSTED_WEIGHTS_ALLOW_PATTERNS),
            )
        )
        manifest = _read_json_object(manifest_snapshot / HOSTED_WEIGHTS_MANIFEST, label=f"hosted repo {repo_id!r}")
        allow_patterns = _hosted_weights_allow_patterns_from_manifest(manifest, label=f"hosted repo {repo_id!r}")
        return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns))
    except Exception as exc:
        raise ValueError(
            f"Could not resolve hosted pre-converted MLX weights repo {repo_id!r}: {exc}. "
            "Check the repo id, network/cache access, and license approval; fallback to locally converted .npz "
            "weights if needed."
        ) from exc


def validate_hosted_weights_layout(
    layout_dir: str | Path,
    *,
    source_label: str | None = None,
    require_approved_license: bool = False,
) -> WeightsSourceResolution:
    """Validate the v0.2 hosted/pre-converted layout and return runtime paths.

    The validator is intentionally small and filesystem-oriented so tests can cover hosted layout
    contract behavior with tiny fixtures and without downloading real model weights.
    """

    root = Path(layout_dir).expanduser()
    label = source_label or str(root)
    manifest = _read_json_object(root / HOSTED_WEIGHTS_MANIFEST, label="hosted weights manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{label} has unsupported {HOSTED_WEIGHTS_MANIFEST} schema_version; expected 1")
    if manifest.get("format") != "irodori-tts-mlx-weights":
        raise ValueError(f"{label} is not an irodori-tts-mlx pre-converted weights layout")
    if manifest.get("format_version") != "0.2":
        raise ValueError(f"{label} has unsupported weights format_version {manifest.get('format_version')!r}; expected '0.2'")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{label} manifest must include a files object")
    normalized_files = dict(files)
    normalized_files["manifest"] = HOSTED_WEIGHTS_MANIFEST

    missing_entries = [
        key
        for key in HOSTED_WEIGHTS_REQUIRED_FILES
        if not isinstance(normalized_files.get(key), str) or not normalized_files[key].strip()
    ]
    if missing_entries:
        raise ValueError(f"{label} manifest is missing file entries: {', '.join(missing_entries)}")

    layout_files = {
        key: _layout_file_path(root, str(normalized_files[key]), label=label)
        for key in HOSTED_WEIGHTS_REQUIRED_FILES
    }
    missing_files = [
        str(normalized_files[key])
        for key in HOSTED_WEIGHTS_REQUIRED_FILES
        if not layout_files[key].is_file()
    ]
    if missing_files:
        raise ValueError(
            f"{label} pre-converted weights layout is missing required files: {', '.join(missing_files)}. "
            "Fallback: run local conversion and pass the converted .npz weights path."
        )

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} manifest must include runtime metadata")
    license_review = manifest.get("license_review")
    if not isinstance(license_review, dict):
        raise ValueError(f"{label} manifest must include license_review metadata")
    status = license_review.get("status")
    if require_approved_license and status != "approved":
        raise ValueError(
            f"{label} hosted weights license_review.status is {status!r}, expected 'approved'. "
            "Do not use unpublished or unapproved hosted weights; fallback to locally converted .npz weights."
        )

    checksums = _parse_sha256sum_lines(layout_files["checksums"].read_text(encoding="utf-8"), label=label)
    checksum_required = [key for key in HOSTED_WEIGHTS_REQUIRED_FILES if key != "checksums"]
    not_listed = [str(normalized_files[key]) for key in checksum_required if str(normalized_files[key]) not in checksums]
    if not_listed:
        raise ValueError(f"{label} checksums file does not list required files: {', '.join(not_listed)}")
    mismatched = [
        str(normalized_files[key])
        for key in checksum_required
        if checksums[str(normalized_files[key])] != _sha256(layout_files[key])
    ]
    if mismatched:
        raise ValueError(f"{label} checksums file has mismatched sha256 digests: {', '.join(mismatched)}")

    return WeightsSourceResolution(
        weights_path=layout_files["weights"],
        model_config_path=layout_files["model_config"],
        layout_dir=root,
        source_label=label,
        source_kind="hosted-layout",
        manifest=manifest,
    )


def resolve_weights_source(
    *,
    weights: str | Path | None = None,
    weights_dir: str | Path | None = None,
    weights_repo: str | None = None,
    snapshot_downloader: SnapshotDownloader | None = None,
) -> WeightsSourceResolution:
    """Resolve exactly one supported source: direct local .npz, local hosted layout, or repo id."""

    selected = [value for value in (weights, weights_dir, weights_repo) if value is not None and str(value).strip()]
    if not selected:
        raise ValueError("choose one weights source: local .npz, hosted-layout directory, or hosted repo id")
    if len(selected) > 1:
        raise ValueError("choose only one weights source; local .npz remains the fallback path")

    if weights is not None and str(weights).strip():
        return WeightsSourceResolution(
            weights_path=Path(weights).expanduser(),
            model_config_path=None,
            layout_dir=None,
            source_label="local converted .npz fallback",
            source_kind="local-npz",
            manifest=None,
        )
    if weights_dir is not None and str(weights_dir).strip():
        return validate_hosted_weights_layout(weights_dir, source_label="local hosted-layout directory")

    assert weights_repo is not None
    downloader = snapshot_downloader or default_huggingface_snapshot_download
    snapshot = Path(downloader(str(weights_repo))).expanduser()
    return validate_hosted_weights_layout(
        snapshot,
        source_label=f"hosted repo {weights_repo!r}",
        require_approved_license=True,
    )
