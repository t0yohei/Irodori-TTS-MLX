#!/usr/bin/env python3
"""Run real-checkpoint VoiceDesign integration checks.

This helper downloads the public VoiceDesign checkpoint from Hugging Face,
verifies that checkpoint inspection succeeds, and then exercises the converter
validation path against the real artifact. Full `.npz` export is optional,
because it is much heavier than the header-only validation path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import convert_weights  # noqa: E402
from scripts.inspect_checkpoint import inspect_local_safetensors  # noqa: E402

DEFAULT_REPO_ID = "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
DEFAULT_FILENAME = "model.safetensors"
DEFAULT_REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real VoiceDesign checkpoint integration check against inspect + convert paths."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id to validate.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Checkpoint filename inside the repo.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision to download.")
    parser.add_argument(
        "--download-dir",
        help="Optional directory for the downloaded checkpoint. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--full-conversion",
        action="store_true",
        help="Load tensor payloads and verify the full `.npz` export path in addition to dry-run validation.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON output.")
    return parser.parse_args()


def _require_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on CI/runtime environment.
        raise RuntimeError(
            "huggingface_hub is required for the real-checkpoint integration check. Install it before running this script."
        ) from exc
    return hf_hub_download


def run_integration(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    download_dir: str | None,
    full_conversion: bool,
) -> dict[str, Any]:
    hf_hub_download = _require_hf_hub_download()
    with tempfile.TemporaryDirectory(prefix="voicedesign-integration-") as tmp:
        local_dir = Path(download_dir) if download_dir else Path(tmp)
        local_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        )

        inspection = inspect_local_safetensors(checkpoint_path)
        config, records = convert_weights.load_checkpoint(checkpoint_path, load_arrays=full_conversion)
        validation = convert_weights.validate_records(records, config)
        report = convert_weights.build_report(
            checkpoint_path,
            None,
            records,
            validation,
            dry_run=not full_conversion,
        )

        result: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "checkpoint_path": str(checkpoint_path),
            "full_conversion": full_conversion,
            "inspection": {
                "tensor_count": len(inspection.tensors),
                "has_config": inspection.config is not None,
                "source": inspection.source,
            },
            "report": report,
        }

        if not validation["ok"]:
            raise RuntimeError(convert_weights.validation_error_message(validation))

        if full_conversion:
            arrays = convert_weights.records_to_arrays(records, checkpoint_family=validation["checkpoint_family"])
            result["full_conversion_export"] = {
                "array_count": len(arrays),
                "sample_keys": list(sorted(arrays))[:5],
            }
        return result


def main() -> int:
    args = parse_args()
    result = run_integration(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        download_dir=args.download_dir,
        full_conversion=args.full_conversion,
    )
    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        report = result["report"]
        print(f"repo_id: {result['repo_id']}")
        print(f"revision: {result['revision']}")
        print(f"checkpoint_path: {result['checkpoint_path']}")
        print(f"full_conversion: {result['full_conversion']}")
        print(f"inspection_tensor_count: {result['inspection']['tensor_count']}")
        print(f"checkpoint_family: {report['checkpoint_family']}")
        print(f"supported_checkpoint: {report['supported_checkpoint']}")
        print(f"validation: {'ok' if report['validation']['ok'] else 'failed'}")
        if args.full_conversion:
            print(f"full_conversion_array_count: {result['full_conversion_export']['array_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI guard.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
