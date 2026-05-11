#!/usr/bin/env python3
"""Collect environment metadata for Irodori-TTS Apple Silicon baseline reports.

The script is intentionally lightweight: it does not download model weights, run
inference, or require PyTorch. If optional packages are installed, it reports
what it can discover.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def optional_torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"available": False, "import_error": repr(exc)}

    mps = getattr(torch.backends, "mps", None)
    return {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "mps_built": bool(mps.is_built()) if mps is not None else False,
        "mps_available": bool(mps.is_available()) if mps is not None else False,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def optional_package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:  # pragma: no cover - depends on local env
        return None
    return getattr(module, "__version__", None)


def main() -> None:
    cwd = Path.cwd()
    env_keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    ]

    data: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mac_ver": platform.mac_ver()[0],
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "commands": {
            "git": shutil.which("git"),
            "uv": shutil.which("uv"),
        },
        "git": {
            "cwd": str(cwd),
            "top_level": run(["git", "rev-parse", "--show-toplevel"], cwd=cwd),
            "commit": run(["git", "rev-parse", "HEAD"], cwd=cwd),
            "branch": run(["git", "branch", "--show-current"], cwd=cwd),
            "status_short": run(["git", "status", "--short"], cwd=cwd),
        },
        "packages": {
            "torch": optional_torch_info(),
            "torchaudio": optional_package_version("torchaudio"),
            "huggingface_hub": optional_package_version("huggingface_hub"),
            "safetensors": optional_package_version("safetensors"),
            "soundfile": optional_package_version("soundfile"),
        },
        "environment": {key: os.environ.get(key) for key in env_keys if os.environ.get(key)},
    }

    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
