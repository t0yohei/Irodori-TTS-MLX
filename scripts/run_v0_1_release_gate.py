#!/usr/bin/env python3
"""Run the v0.1 end-to-end WAV-generation release gate.

The required gate is the public v3 checkpoint path because it is the shortest
v0.1-supported fresh-environment route to a WAV: download, inspect, convert,
generate with predicted duration, and preserve metadata/artifacts. VoiceDesign
coverage is available as an optional heavier companion check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v3_generation_ci, run_voicedesign_generation_ci  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v0.1 release gate that proves a fresh environment can "
            "download a supported checkpoint, inspect it, convert it, generate "
            "a WAV, and validate generation metadata."
        )
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for release-gate artifacts. Defaults to a newly-created temporary directory kept after exit.",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory for downloaded checkpoints. Defaults to a newly-created temporary directory kept after exit.",
    )
    parser.add_argument(
        "--upstream-root",
        help="Optional upstream Irodori-TTS checkout path used by generation subprocesses.",
    )
    parser.add_argument("--codec-device", default=run_v3_generation_ci.DEFAULT_CODEC_DEVICE)
    parser.add_argument("--num-steps", type=int, default=run_v3_generation_ci.DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--include-optional-voicedesign",
        action="store_true",
        help="Also run the heavier VoiceDesign caption-conditioned hosted generation check.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON output.")
    return parser.parse_args()


def _artifact_summary(result: dict[str, Any]) -> dict[str, Any]:
    generation = result.get("generation", {})
    request = generation.get("request", {}) if isinstance(generation, dict) else {}
    generated = generation.get("result", {}) if isinstance(generation, dict) else {}
    return {
        "repo_id": result["repo_id"],
        "revision": result["revision"],
        "checkpoint_family": result["report"]["checkpoint_family"],
        "tensor_count": result["inspection"]["tensor_count"],
        "weights_path": result["weights_path"],
        "weights_bytes": result["weights_bytes"],
        "output_wav": result["output_wav"],
        "output_wav_bytes": result["output_wav_bytes"],
        "metadata_json": result["metadata_json"],
        "stdout_path": result["stdout_path"],
        "stderr_path": result["stderr_path"],
        "duration_mode": generated.get("duration_mode"),
        "samples": generated.get("samples"),
        "seconds_request": request.get("seconds"),
        "has_caption": bool(request.get("caption")),
    }


def _assert_required_v3_metadata(result: dict[str, Any]) -> None:
    generation = result.get("generation", {})
    request = generation.get("request", {}) if isinstance(generation, dict) else {}
    generated = generation.get("result", {}) if isinstance(generation, dict) else {}
    duration_mode = generated.get("duration_mode")
    if duration_mode != "predicted":
        raise RuntimeError(f"v0.1 required gate expected duration_mode='predicted', got {duration_mode!r}.")
    if request.get("seconds") is not None:
        raise RuntimeError("v0.1 required gate expected the v3 generation request to omit manual seconds.")
    if int(result.get("output_wav_bytes", 0)) <= 0:
        raise RuntimeError("v0.1 required gate generated an empty WAV artifact.")
    if int(result.get("weights_bytes", 0)) <= 0:
        raise RuntimeError("v0.1 required gate generated an empty converted-weight artifact.")
    metadata_json = result.get("metadata_json")
    if not metadata_json:
        raise RuntimeError("v0.1 required gate did not report a metadata JSON artifact path.")
    metadata_path = Path(str(metadata_json))
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        raise RuntimeError(f"v0.1 required gate generated an empty or missing metadata artifact: {metadata_path}")


def run_release_gate(
    *,
    output_dir: str | None,
    download_dir: str | None,
    upstream_root: str | None,
    codec_device: str,
    num_steps: int,
    include_optional_voicedesign: bool,
) -> dict[str, Any]:
    root_download_dir = Path(download_dir) if download_dir else Path(tempfile.mkdtemp(prefix="v0.1-release-gate-download-"))
    root_output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="v0.1-release-gate-output-"))
    root_download_dir.mkdir(parents=True, exist_ok=True)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    v3_download_dir = root_download_dir / "v3"
    v3_output_dir = root_output_dir / "v3"
    v3_result = run_v3_generation_ci.run_generation(
        repo_id=run_v3_generation_ci.DEFAULT_REPO_ID,
        filename=run_v3_generation_ci.DEFAULT_FILENAME,
        revision=run_v3_generation_ci.DEFAULT_REVISION,
        download_dir=str(v3_download_dir),
        output_dir=str(v3_output_dir),
        text=run_v3_generation_ci.DEFAULT_TEXT,
        num_steps=num_steps,
        codec_device=codec_device,
        upstream_root=upstream_root,
    )
    _assert_required_v3_metadata(v3_result)

    checks: dict[str, Any] = {
        "required_v3": {
            "status": "pass",
            "stages": ["download", "inspect", "convert", "generate_wav", "metadata_validate"],
            "artifacts": _artifact_summary(v3_result),
        }
    }

    if include_optional_voicedesign:
        voicedesign_result = run_voicedesign_generation_ci.run_generation(
            repo_id=run_voicedesign_generation_ci.DEFAULT_REPO_ID,
            filename=run_voicedesign_generation_ci.DEFAULT_FILENAME,
            revision=run_voicedesign_generation_ci.DEFAULT_REVISION,
            download_dir=str(root_download_dir / "voicedesign"),
            output_dir=str(root_output_dir / "voicedesign"),
            text=run_voicedesign_generation_ci.DEFAULT_TEXT,
            caption=run_voicedesign_generation_ci.DEFAULT_CAPTION,
            seconds=run_voicedesign_generation_ci.DEFAULT_SECONDS,
            num_steps=num_steps,
            codec_device=codec_device,
            upstream_root=upstream_root,
        )
        checks["optional_voicedesign"] = {
            "status": "pass",
            "stages": ["download", "inspect", "convert", "generate_wav", "metadata_validate"],
            "artifacts": _artifact_summary(voicedesign_result),
        }
    else:
        checks["optional_voicedesign"] = {
            "status": "skipped",
            "reason": "Run with --include-optional-voicedesign for the heavier caption-conditioned gate.",
        }

    summary = {
        "release_gate": "v0.1-wav-generation",
        "status": "pass",
        "required_check": "required_v3",
        "output_dir": str(root_output_dir),
        "download_dir": str(root_download_dir),
        "checks": checks,
    }
    summary_path = root_output_dir / "v0.1-release-gate-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary

def main() -> int:
    args = parse_args()
    summary = run_release_gate(
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        upstream_root=args.upstream_root,
        codec_device=args.codec_device,
        num_steps=args.num_steps,
        include_optional_voicedesign=args.include_optional_voicedesign,
    )
    if args.json_output:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"release_gate: {summary['release_gate']}")
        print(f"status: {summary['status']}")
        print(f"required_check: {summary['required_check']}")
        print(f"summary_json: {summary['summary_json']}")
        required = summary["checks"]["required_v3"]["artifacts"]
        print(f"required_v3_output_wav: {required['output_wav']}")
        print(f"required_v3_metadata_json: {required['metadata_json']}")
        print(f"optional_voicedesign: {summary['checks']['optional_voicedesign']['status']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI guard.
        print(exc.stdout or "", end="", file=sys.stdout)
        print(exc.stderr or f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:  # pragma: no cover - CLI guard.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
