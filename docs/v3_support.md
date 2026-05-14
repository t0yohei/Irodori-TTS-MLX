# Irodori-TTS-500M-v3 support

Issue #53 asks for a user-facing support statement plus a reproducible validation path for `Aratako/Irodori-TTS-500M-v3`. This document describes what the repository now supports, how to validate it, and which caveats still matter.

## Executive summary

The repository now treats **`Aratako/Irodori-TTS-500M-v3` as a first-class supported checkpoint family** for:

- checkpoint inspection
- checkpoint conversion to MLX `.npz`
- MLX bridge runtime generation
- predicted-duration runtime semantics when `--seconds` is omitted
- reproducible hosted Apple Silicon validation through a checked-in helper + workflow

In practice, that means the public v3 checkpoint can move through:

1. `scripts/inspect_checkpoint.py`
2. `scripts/convert_weights.py`
3. `scripts/generate_wav.py`
4. `scripts/run_v3_generation_ci.py` / `.github/workflows/v3-hosted-generation.yml`

without needing repo-local one-off patches.

## What is supported

| Surface | Status | Notes |
| --- | --- | --- |
| `scripts/inspect_checkpoint.py` | supported | Reads v3 metadata/config and tensor headers without loading the tensor payload. |
| `scripts/convert_weights.py` | supported | Detects the public v3 family, validates duration-predictor-specific keys/config, and exports MLX `.npz` weights. |
| `irodori_mlx.model.TextToLatentRFDiT` | supported | Includes the v3 duration predictor path and required weight loading hooks. |
| `irodori_mlx.runtime.MLXDACVAERuntime` | supported | Uses the duration predictor when `use_duration_predictor=true` and `--seconds` is omitted; manual `--seconds` still wins when provided. |
| `scripts/generate_wav.py` | supported | Exposes the runtime semantics directly, including `--duration-scale`, JSON metadata, and manual override behavior. |
| Hosted Apple Silicon validation | supported | `scripts/run_v3_generation_ci.py` downloads the public checkpoint, converts it, runs generation, and asserts `duration_mode="predicted"`. |

## Duration semantics

v3 uses the runtime semantics implemented under issue #52:

- `--seconds` is a manual override.
- If `--seconds` is omitted and the loaded config enables `use_duration_predictor`, the runtime predicts duration from the current text/reference conditions.
- `--duration-scale` only affects that predicted path.
- Older checkpoint families without the duration predictor keep the historical fixed-duration fallback when `--seconds` is omitted.

This matters for validation because a v3 smoke run should usually **omit `--seconds`** if you want to prove that the predictor path still works.

## Manual validation workflow

A reproducible local path looks like this:

```bash
python3 scripts/inspect_checkpoint.py /path/to/v3/model.safetensors --json
python3 scripts/convert_weights.py /path/to/v3/model.safetensors /tmp/irodori-v3.npz --dry-run --json
python3 scripts/convert_weights.py /path/to/v3/model.safetensors /tmp/irodori-v3.npz
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --weights /tmp/irodori-v3.npz \
  --model-config-json /path/to/v3-model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav \
  --num-steps 40 \
  --json > /tmp/irodori-v3-result.json
```

For this validation path, leave out `--seconds` on purpose. The resulting JSON should report:

- `result.duration_mode == "predicted"`
- `request.seconds == null`

If you want exact length control instead, add `--seconds N` and expect `duration_mode == "manual"`.

## Hosted CI-assisted validation

For repeatable Apple Silicon coverage, use:

- helper: `scripts/run_v3_generation_ci.py`
- workflow: `.github/workflows/v3-hosted-generation.yml`

That path:

1. downloads the public `Aratako/Irodori-TTS-500M-v3` checkpoint
2. validates and converts it
3. runs `generate_wav.py` without `--seconds`
4. uploads JSON/log/WAV artifacts
5. fails if the runtime does not report `duration_mode="predicted"`

This gives the repository a practical regression check that is stronger than docs-only claims.

## Known caveats

### 1. Hosted coverage depends on standard Apple Silicon runner availability

The checked-in hosted workflow uses the standard `macos-14` M1 runner, so public-repository coverage avoids larger-runner billing. Queueing and the smaller standard-runner resource envelope still matter.

### 2. The hosted smoke run uses `--no-reference`

That keeps the validation path self-contained and avoids committing reference audio assets. It proves the v3 conversion/runtime/duration semantics path, but it is not a speaker-fidelity benchmark. For real speaker-conditioned checks, run the manual workflow with `--reference-wav`.

### 3. Watermarking is still best-effort

`--enable-watermark` only matters when the upstream DACVAE runtime exposes watermark support. The repo should document it as optional behavior, not as a guaranteed property of every generated WAV.

### 4. Support is scoped to the inspected public v3 family

The repo does not claim broad support for every future or private v3-like variant. Validation assumes the checkpoint still matches the public family that the converter/runtime tests know how to recognize.

## Current user-facing support statement

> `Aratako/Irodori-TTS-500M-v3` is supported as a first-class checkpoint family for inspection, conversion, and MLX bridge generation. The repository includes both a manual validation recipe and a hosted Apple Silicon helper/workflow that exercises the predicted-duration path and records artifacts for regression tracking.
