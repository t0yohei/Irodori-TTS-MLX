#!/usr/bin/env python3
"""Run hosted Apple Silicon VoiceDesign end-to-end generation coverage.

This helper is intended for GitHub Actions on Apple Silicon runners. It:

1. downloads the public VoiceDesign checkpoint from Hugging Face
2. inspects and validates the checkpoint layout
3. converts the checkpoint into an MLX-friendly `.npz`
4. runs `scripts/generate_wav.py --caption ...` against the converted weights
5. emits machine-readable metadata for debugging and artifact upload
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from irodori_mlx.config import ModelConfig  # noqa: E402
from scripts import convert_weights  # noqa: E402
from scripts.inspect_checkpoint import inspect_local_safetensors  # noqa: E402

DEFAULT_REPO_ID = "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
DEFAULT_FILENAME = "model.safetensors"
DEFAULT_REVISION = "main"
DEFAULT_TEXT = "こんにちは。今日は良い天気です。"
DEFAULT_CAPTION = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"
DEFAULT_SECONDS = 2.0
DEFAULT_NUM_STEPS = 8
DEFAULT_CODEC_DEVICE = "cpu"
MODEL_CONFIG_KEYS = {field.name for field in fields(ModelConfig)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the public VoiceDesign checkpoint, convert it, and run the "
            "full generate_wav.py caption-conditioned path."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id to validate.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Checkpoint filename inside the repo.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision to download.")
    parser.add_argument(
        "--download-dir",
        help="Optional directory for the downloaded checkpoint. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for converted weights, metadata, logs, and generated WAV. Defaults to a temporary directory.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text prompt for the generation run.")
    parser.add_argument("--caption", default=DEFAULT_CAPTION, help="Caption/style prompt for VoiceDesign generation.")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="Requested output duration in seconds.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="RF sampling steps for smoke coverage.")
    parser.add_argument("--codec-device", default=DEFAULT_CODEC_DEVICE, help="PyTorch codec device to pass to generate_wav.py.")
    parser.add_argument(
        "--upstream-root",
        help="Optional upstream Irodori-TTS checkout path to prepend to PYTHONPATH for the generation subprocess.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON output.")
    return parser.parse_args()


def _require_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on CI/runtime environment.
        raise RuntimeError(
            "huggingface_hub is required for the hosted VoiceDesign generation helper. Install it before running this script."
        ) from exc
    return hf_hub_download


def _model_config_payload(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint metadata did not expose a model config JSON object.")
    payload = {key: value for key, value in config.items() if key in MODEL_CONFIG_KEYS}
    if payload.get("use_caption_condition") is not True:
        raise RuntimeError("Hosted VoiceDesign generation requires a caption-conditioned checkpoint config.")
    return payload


def _load_generation_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as direct_error:
        start = stdout.rfind("\n{")
        start = start + 1 if start >= 0 else stdout.find("{")
        if start < 0:
            raise direct_error
        payload = json.loads(stdout[start:])
    if not isinstance(payload, dict):
        raise RuntimeError("generate_wav.py --json must emit a JSON object.")
    return payload


def _build_generation_command(
    *,
    python_executable: str,
    weights_path: Path,
    model_config_path: Path,
    output_wav: Path,
    metadata_json: Path,
    text: str,
    caption: str,
    seconds: float,
    num_steps: int,
    codec_device: str,
) -> list[str]:
    return [
        python_executable,
        str(ROOT / "scripts" / "generate_wav.py"),
        "--weights",
        str(weights_path),
        "--model-config-json",
        str(model_config_path),
        "--text",
        text,
        "--caption",
        caption,
        "--no-reference",
        "--output",
        str(output_wav),
        "--seconds",
        str(seconds),
        "--num-steps",
        str(num_steps),
        "--codec-device",
        codec_device,
        "--json",
        "--metadata-json",
        str(metadata_json),
    ]


def run_generation(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    download_dir: str | None,
    output_dir: str | None,
    text: str,
    caption: str,
    seconds: float,
    num_steps: int,
    codec_device: str,
    upstream_root: str | None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    hf_hub_download = _require_hf_hub_download()
    python_executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="voicedesign-generation-download-") as tmp_download, tempfile.TemporaryDirectory(
        prefix="voicedesign-generation-output-"
    ) as tmp_output:
        local_download_dir = Path(download_dir) if download_dir else Path(tmp_download)
        local_output_dir = Path(output_dir) if output_dir else Path(tmp_output)
        local_download_dir.mkdir(parents=True, exist_ok=True)
        local_output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(local_download_dir),
            )
        )
        inspection = inspect_local_safetensors(checkpoint_path)
        config, records = convert_weights.load_checkpoint(checkpoint_path, load_arrays=True)
        validation = convert_weights.validate_records(records, config)
        if not validation["ok"]:
            raise RuntimeError(convert_weights.validation_error_message(validation))

        report = convert_weights.build_report(
            checkpoint_path,
            local_output_dir / "irodori-voicedesign.npz",
            records,
            validation,
            dry_run=False,
        )
        arrays = convert_weights.records_to_arrays(records, checkpoint_family=validation["checkpoint_family"])
        weights_path = local_output_dir / "irodori-voicedesign.npz"
        convert_weights.write_npz_atomic(weights_path, arrays)

        model_config_payload = _model_config_payload(config)
        model_config_path = local_output_dir / "voicedesign-model-config.json"
        model_config_path.write_text(
            json.dumps(model_config_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        output_wav = local_output_dir / "voicedesign-hosted.wav"
        metadata_json = local_output_dir / "voicedesign-generation-metadata.json"
        stdout_path = local_output_dir / "voicedesign-generate.stdout.json"
        stderr_path = local_output_dir / "voicedesign-generate.stderr.txt"
        command = _build_generation_command(
            python_executable=python_executable,
            weights_path=weights_path,
            model_config_path=model_config_path,
            output_wav=output_wav,
            metadata_json=metadata_json,
            text=text,
            caption=caption,
            seconds=seconds,
            num_steps=num_steps,
            codec_device=codec_device,
        )

        env = os.environ.copy()
        if upstream_root:
            current = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(Path(upstream_root)) + (os.pathsep + current if current else "")

        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        generation_payload = _load_generation_payload(completed.stdout)

        return {
            "repo_id": repo_id,
            "revision": revision,
            "checkpoint_path": str(checkpoint_path),
            "inspection": {
                "tensor_count": len(inspection.tensors),
                "has_config": inspection.config is not None,
                "source": inspection.source,
            },
            "report": report,
            "weights_path": str(weights_path),
            "weights_bytes": weights_path.stat().st_size,
            "model_config_path": str(model_config_path),
            "output_wav": str(output_wav),
            "output_wav_bytes": output_wav.stat().st_size,
            "metadata_json": str(metadata_json),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "command": {
                "argv": command,
                "shell": shlex.join(command),
            },
            "generation": generation_payload,
        }


def main() -> int:
    args = parse_args()
    result = run_generation(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        download_dir=args.download_dir,
        output_dir=args.output_dir,
        text=args.text,
        caption=args.caption,
        seconds=args.seconds,
        num_steps=args.num_steps,
        codec_device=args.codec_device,
        upstream_root=args.upstream_root,
    )
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"repo_id: {result['repo_id']}")
        print(f"revision: {result['revision']}")
        print(f"checkpoint_path: {result['checkpoint_path']}")
        print(f"checkpoint_family: {result['report']['checkpoint_family']}")
        print(f"weights_path: {result['weights_path']}")
        print(f"output_wav: {result['output_wav']}")
        print(f"stdout_path: {result['stdout_path']}")
        print(f"stderr_path: {result['stderr_path']}")
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
