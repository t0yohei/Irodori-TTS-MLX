# Apple Silicon PyTorch Baseline Report

Date:
Reporter:
Related issue: [#2](https://github.com/t0yohei/Irodori-TTS-MLX/issues/2)

## Summary

- Baseline status: `not run` / `passed` / `passed with limitations` / `failed`
- Host:
- Main result:
- Known blocker, if any:

## Environment

Paste or summarize `scripts/collect_baseline_env.py` output.

```json
{}
```

Additional local notes:

- Power mode:
- Thermal state:
- Memory pressure:
- Hugging Face cache state: cold / warm / unknown

## Upstream source

- Repository: https://github.com/Aratako/Irodori-TTS
- Commit:
- Install command:

```bash
uv sync
```

## Checkpoints

| Component | Identifier | Revision / file | Notes |
| --- | --- | --- | --- |
| Irodori-TTS base model | `Aratako/Irodori-TTS-500M-v2` | `model.safetensors` | |
| DACVAE codec | `Aratako/Semantic-DACVAE-Japanese-32dim` | | |
| VoiceDesign model, optional | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | `model.safetensors` | |

## Common sampling parameters

| Parameter | Value |
| --- | --- |
| `--model-device` | `mps` |
| `--codec-device` | `mps` |
| `--model-precision` | `fp32` |
| `--codec-precision` | `fp32` |
| `--num-steps` | `40` |
| `--seed` | `20260511` |
| `--show-timings` | enabled |

## Run 1: base model, no reference audio

Command:

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

Result:

- Status:
- Output path:
- Used seed:
- Audio duration:
- Subjective notes:

Timing:

| Stage | Time |
| --- | ---: |
| model load / checkpoint download | |
| reference encode | n/a |
| latent sampling | |
| DACVAE decode | |
| total to decode | |
| wall clock | |

Memory notes:

- Peak memory:
- Measurement method:

Warnings or limitations:

```text

```

## Run 2: base model, reference audio

Reference audio:

- Source:
- License / sharing status:
- Duration:
- Sample rate:

Command:

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

Result:

- Status:
- Output path:
- Used seed:
- Audio duration:
- Subjective notes:

Timing:

| Stage | Time |
| --- | ---: |
| reference encode | |
| latent sampling | |
| DACVAE decode | |
| total to decode | |
| wall clock | |

Memory notes:

- Peak memory:
- Measurement method:

Warnings or limitations:

```text

```

## Optional run: VoiceDesign no-reference

Command:

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

Result:

- Status:
- Output path:
- Used seed:
- Audio duration:
- Subjective notes:

## Artifacts

Generated audio should not be committed directly unless explicitly approved and licensed.

| Artifact | Location | Shared? | Notes |
| --- | --- | --- | --- |
| no-ref WAV | | no | |
| reference WAV | | no | |
| VoiceDesign WAV | | no | |
| console log | | yes / no | |

## Conclusions for MLX work

- Baseline command to compare against:
- Baseline latency target:
- Quality comparison notes:
- MPS/PyTorch limitation that MLX may avoid:
- Follow-up issue(s):
