# Upstream PyTorch Baseline on Apple Silicon

Issue: [#2 Reproduce upstream PyTorch inference on Apple Silicon](https://github.com/t0yohei/Irodori-TTS-MLX/issues/2)

Measured report: [2026-05-11 Apple Silicon PyTorch baseline](baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md)

This document defines the first reproducible baseline for comparing future MLX work against upstream Irodori-TTS PyTorch inference.

The baseline target is upstream `Aratako/Irodori-TTS` v2 inference with:

- model checkpoint: `Aratako/Irodori-TTS-500M-v2`
- codec checkpoint: `Aratako/Semantic-DACVAE-Japanese-32dim`
- host platform: macOS on Apple Silicon
- primary device target: PyTorch MPS
- precision: `fp32` for both model and codec unless a run explicitly records otherwise

Do not commit generated audio, downloaded checkpoints, Hugging Face cache contents, or upstream source checkouts to this repository.

## Prerequisites

Install the baseline tools outside this repository:

```bash
brew install git uv
```

Use a local workspace directory for upstream code and generated outputs. The examples below use `external/` and `baseline-runs/`; both are ignored by this repository.

## 1. Check out upstream Irodori-TTS

```bash
mkdir -p external baseline-runs

git clone https://github.com/Aratako/Irodori-TTS.git external/Irodori-TTS
cd external/Irodori-TTS

git rev-parse HEAD
uv sync
```

Record the upstream commit in the report template before running inference.

For macOS, upstream `uv sync` should install the default PyTorch build instead of the Linux/Windows CUDA index.

## 2. Collect environment information

From this repository root, run:

```bash
python3 scripts/collect_baseline_env.py > baseline-runs/env.json
```

If you want the helper to record the upstream checkout commit, run it from the upstream checkout or pass the upstream commit manually into the report.

Also record any memory observations from Activity Monitor, `powermetrics`, or another local monitoring tool. The helper intentionally does not require elevated privileges.

## 3. No-reference baseline command

Run this from `external/Irodori-TTS`:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav ../../baseline-runs/base-no-ref-seed-20260511.wav \
  --model-device mps \
  --codec-device mps \
  --model-precision fp32 \
  --codec-precision fp32 \
  --num-steps 40 \
  --seed 20260511 \
  --show-timings
```

Capture the full console output, especially:

- resolved checkpoint path
- seed
- per-stage `[timing]` lines
- total time to decode
- any MPS warnings or CPU fallback messages

## 4. Reference-audio baseline command

Use this only with a local reference WAV that can be safely used for development. Do not commit the reference audio unless its license explicitly allows redistribution.

Run this from `external/Irodori-TTS`:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav ../../baseline-runs/base-ref-seed-20260511.wav \
  --model-device mps \
  --codec-device mps \
  --model-precision fp32 \
  --codec-precision fp32 \
  --num-steps 40 \
  --seed 20260511 \
  --max-ref-seconds 30 \
  --ref-normalize-db -16 \
  --show-timings
```

Record the reference file source, duration, sample rate, and license status in the report. If the reference audio cannot be shared, record only metadata that is safe to publish.

## 5. Optional VoiceDesign baseline

This is not required for the base-model baseline, but it is useful if future MLX work targets the VoiceDesign checkpoint.

Run this from `external/Irodori-TTS`:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "今日はいい天気ですね。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav ../../baseline-runs/voice-design-no-ref-seed-20260511.wav \
  --model-device mps \
  --codec-device mps \
  --model-precision fp32 \
  --codec-precision fp32 \
  --num-steps 40 \
  --seed 20260511 \
  --show-timings
```

## 6. CPU fallback command

If MPS fails because of an unsupported operation, rerun the same command with CPU devices and record the failure as an MPS limitation:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav ../../baseline-runs/base-no-ref-cpu-seed-20260511.wav \
  --model-device cpu \
  --codec-device cpu \
  --model-precision fp32 \
  --codec-precision fp32 \
  --num-steps 40 \
  --seed 20260511 \
  --show-timings
```

## 7. Report the result

Create a dated Markdown report under `docs/baseline-reports/` only when the result is safe to publish and still useful as a current reference. Include:

- environment details from `baseline-runs/env.json`
- upstream commit and checkpoint identifiers
- exact commands
- sampling parameters
- latency/timing lines
- memory notes
- generated output paths and whether artifacts are shared externally
- known upstream or MPS limitations

Commit only documentation or small text reports that are safe to publish. Keep generated WAV files and caches outside git. One-off local notes belong in ignored `baseline-runs/`, not in the public docs tree.

## Known limitations to watch for

- First-run latency includes Hugging Face downloads and should not be compared against warm-cache MLX runs.
- PyTorch MPS may fall back to CPU or fail for unsupported operations. Record the exact warning or traceback.
- Reference-audio results are not comparable unless the same reference file and preprocessing parameters are used.
- The upstream command currently synthesizes a fixed 30-second latent window and may trim tail silence afterward.
