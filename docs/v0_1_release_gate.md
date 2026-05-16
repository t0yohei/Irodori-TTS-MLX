# v0.1 release gate

Parent: [#68 v0.1 release scope](https://github.com/t0yohei/Irodori-TTS-MLX/issues/68)

The v0.1 release gate proves that a fresh Apple Silicon environment can reproduce WAV generation through the documented path without checked-in model weights or audio artifacts.

## Required gate

The required release gate is **v3 no-reference predicted-duration generation**:

1. download `Aratako/Irodori-TTS-500M-v3` from Hugging Face,
2. inspect the safetensors metadata and tensor layout,
3. validate and convert the checkpoint to MLX `.npz`,
4. run `scripts/generate_wav.py --no-reference` without `--seconds`, and
5. validate that the JSON metadata reports `duration_mode="predicted"` and that non-empty WAV / metadata / converted-weight artifacts were produced.

This path is the shortest v0.1-supported fresh-environment check because it does not require a committed reference WAV and exercises the duration predictor semantics that distinguish v3 support.

## Optional heavier gate

`Aratako/Irodori-TTS-500M-v2-VoiceDesign` caption-conditioned generation remains available as an optional companion gate. It exercises the same download → inspect → convert → generate → metadata/artifact path with `--caption`, but it is not required for every v0.1 release decision because the required v3 gate already proves the documented no-reference path to WAV generation.

## Local/manual command

From a fresh checkout on Apple Silicon macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
python -m pip install safetensors huggingface_hub

git clone --depth=1 https://github.com/Aratako/Irodori-TTS.git /tmp/Irodori-TTS
python -m pip install -e /tmp/Irodori-TTS

python scripts/run_v0_1_release_gate.py \
  --output-dir artifacts/v0.1-release-gate \
  --upstream-root /tmp/Irodori-TTS \
  --json
```

To include the optional VoiceDesign check:

```bash
python scripts/run_v0_1_release_gate.py \
  --output-dir artifacts/v0.1-release-gate \
  --upstream-root /tmp/Irodori-TTS \
  --include-optional-voicedesign \
  --json
```

The summary is written to `v0.1-release-gate-summary.json` inside the output directory. Per-check subdirectories preserve the generated WAV, generation metadata JSON, converted weights, and captured stdout/stderr.

## CI entry point

`.github/workflows/v0.1-release-gate.yml` runs the same command on the GitHub-hosted Apple Silicon `macos-14` runner. It can be started manually with `workflow_dispatch`; the optional VoiceDesign check is controlled by the workflow input. A monthly scheduled run keeps the gate exercised even outside release windows.

## Required vs optional checks

- **Required for v0.1 release readiness:** `scripts/run_v0_1_release_gate.py` default mode / `.github/workflows/v0.1-release-gate.yml` with `include-optional-voicedesign=false`.
- **Optional confidence check:** the same gate with `--include-optional-voicedesign` / workflow input `true`.
- **Lower-level helpers:** `scripts/run_v3_generation_ci.py` and `scripts/run_voicedesign_generation_ci.py` remain useful when debugging a single family, but the release decision should cite the v0.1 gate summary.

Failures are stage-localized by the helper scripts: download/import failures point to Hugging Face or dependency setup, inspection/conversion failures point to checkpoint layout validation, generation failures include captured subprocess stdout/stderr paths, and metadata failures identify the expected v0.1 runtime contract that was not met.
