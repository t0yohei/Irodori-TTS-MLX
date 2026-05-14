# Apple Silicon local `num_steps` presets for v3 and VoiceDesign

Related issue: [#64](https://github.com/t0yohei/irodori-tts-mlx/issues/64)

## Question

For local Apple Silicon usage on this Mac mini, how far can `generate_wav.py --num-steps` be reduced before the latency/quality tradeoff stops making practical sense?

The target cases were:

1. `Aratako/Irodori-TTS-500M-v3` text-only generation
2. `Aratako/Irodori-TTS-500M-v3` generation with reference audio
3. `Aratako/Irodori-TTS-500M-v2-VoiceDesign` caption-conditioned generation

## Setup

- repo worktree: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`
- benchmark Python: `/Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python`
- upstream checkout: `/Users/kouka/.openclaw/workspace/repos/Irodori-TTS`
- v3 weights: `/Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz`
- v3 model config: `/Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json`
- v3 reference WAV: `/Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav`
- VoiceDesign weights: `/Users/kouka/.openclaw/workspace/tmp/irodori-voicedesign/irodori-voicedesign.npz`
- VoiceDesign model config: `/Users/kouka/.openclaw/workspace/tmp/irodori-voicedesign/voicedesign-model-config.json`
- text prompt: `今日はいい天気ですね。`
- VoiceDesign caption: `落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。`
- seed: `20260512`
- step sweep: `8, 12, 16, 24, 40`
- run shape: `--warmup-runs 1 --repeat 1`
- codec device/runtime: default benchmark path (`--codec-device cpu --codec-runtime-mode persistent`)

## Repro commands

### v3 text-only (predicted duration)

```bash
/Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python \
  --case-label v3 \
  --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz \
  --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/Irodori-TTS \
  --omit-seconds \
  --num-steps-sweep 8,12,16,24,40 \
  --warmup-runs 1 \
  --repeat 1 \
  --output-dir benchmark-runs/issue-64-v3-text \
  --report docs/benchmark-reports/2026-05-14-apple-silicon-num-steps-v3-text.md
```

### v3 reference-audio (predicted duration)

```bash
/Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python \
  --case-label v3 \
  --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz \
  --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/Irodori-TTS \
  --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav \
  --omit-seconds \
  --num-steps-sweep 8,12,16,24,40 \
  --warmup-runs 1 \
  --repeat 1 \
  --output-dir benchmark-runs/issue-64-v3-reference \
  --report docs/benchmark-reports/2026-05-14-apple-silicon-num-steps-v3-reference.md
```

### VoiceDesign caption-conditioned

```bash
/Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python \
  --case-label voicedesign-caption \
  --weights /Users/kouka/.openclaw/workspace/tmp/irodori-voicedesign/irodori-voicedesign.npz \
  --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-voicedesign/voicedesign-model-config.json \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/Irodori-TTS \
  --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' \
  --seconds 2 \
  --num-steps-sweep 8,12,16,24,40 \
  --warmup-runs 1 \
  --repeat 1 \
  --output-dir benchmark-runs/issue-64-voicedesign \
  --report docs/benchmark-reports/2026-05-14-apple-silicon-num-steps-voicedesign.md
```

## Measured warm-run results

### v3 text-only

| Steps | `sample_rf` | `total_to_decode` | Wall | Output duration | Mel-L1 vs 40-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 678.6 ms | 1818.5 ms | 6.37 s | 4.56 s | 1.2958 |
| 12 | 1010.7 ms | 2243.4 ms | 6.93 s | 4.48 s | 1.5803 |
| 16 | 1316.0 ms | 2491.6 ms | 7.14 s | 4.56 s | 1.3081 |
| 24 | 1885.3 ms | 2901.6 ms | 7.36 s | 4.32 s | 0.8587 |
| 40 | 3156.2 ms | 4229.6 ms | 8.83 s | 4.44 s | 0.0000 |

### v3 reference-audio

| Steps | `sample_rf` | `total_to_decode` | Wall | Output duration | Mel-L1 vs 40-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 616.1 ms | 2223.4 ms | 6.92 s | 3.64 s | 1.2276 |
| 12 | 881.5 ms | 2436.1 ms | 7.01 s | 3.60 s | 1.3591 |
| 16 | 1096.3 ms | 2572.2 ms | 7.15 s | 3.56 s | 1.3444 |
| 24 | 1667.8 ms | 3194.4 ms | 7.65 s | 3.72 s | 0.1463 |
| 40 | 2717.1 ms | 4241.1 ms | 8.70 s | 3.72 s | 0.0000 |

### VoiceDesign caption-conditioned

| Steps | `sample_rf` | `total_to_decode` | Wall | Output duration | Mel-L1 vs 40-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 471.5 ms | 979.3 ms | 6.11 s | 2.00 s | 1.0092 |
| 12 | 582.8 ms | 1063.8 ms | 6.21 s | 2.00 s | 0.9031 |
| 16 | 811.1 ms | 1341.6 ms | 6.65 s | 2.00 s | 0.5165 |
| 24 | 1163.9 ms | 1703.1 ms | 7.06 s | 2.00 s | 0.1834 |
| 40 | 1898.2 ms | 2390.0 ms | 7.85 s | 2.00 s | 0.0000 |

## Notes on the quality proxy

A real listening test is still a human task. This report does **not** pretend to automate hearing.

To keep the recommendation reproducible, it uses two machine-checkable signals next to latency:

1. generated output duration / shape consistency
2. log-mel L1 distance against the same-case 40-step output

That proxy is enough to spot where lower-step runs stop behaving like the higher-step local baseline, without claiming more than the measurement can support.

## Practical interpretation

### 1. `sample_rf` scales almost linearly with step count

That part was expected, and the measurements confirm it. Across all three cases, most of the latency increase from `12 -> 24 -> 40` comes directly from RF sampling rather than decode.

### 2. `8` is the raw fastest setting, but it is not the best universal preset

`8` wins the latency race everywhere, but the 40-step proxy divergence stays large for all three cases, especially both v3 paths. It is usable for pure speed checks, but it is a fragile default if you want outputs that stay closer to the higher-step baseline.

### 3. `16` does not earn a clear slot

`16` is faster than `24`, but the proxy gap still stays noticeably wider than `24`, while the wall-clock difference is already much smaller than the `40 -> 24` drop. In this sweep it did not emerge as the clearest recommendation point.

### 4. `24` is the first setting that consistently gets close to the 40-step baseline

This is most obvious on:

- v3 reference: Mel-L1 drops from `~1.23-1.36` at `8/12/16` down to `0.1463` at `24`
- VoiceDesign caption: Mel-L1 drops from `1.0092` at `8` to `0.1834` at `24`
- v3 text-only: `24` is also the closest lower-step result to `40` in this sweep

That makes `24` the best balanced preset from this measurement set.

## Recommended local presets

### Fastest acceptable iteration

- **`--num-steps 12`**

Why:

- it keeps `total_to_decode` around **2.24 s** (v3 text), **2.44 s** (v3 reference), and **1.06 s** (VoiceDesign)
- it cuts `total_to_decode` by about **47%**, **43%**, and **55%** respectively versus `40`
- it is still materially safer than blindly dropping to `8` when using one shared preset across all local flows

### Balanced local usage

- **`--num-steps 24`**

Why:

- it is the first lower-step point that stays consistently close to the 40-step baseline in the proxy comparisons
- it still keeps `total_to_decode` well below `40`: about **2.90 s** (v3 text), **3.19 s** (v3 reference), **1.70 s** (VoiceDesign)
- wall time stays in the roughly **7.1-7.7 s** band instead of **7.9-8.8 s**

### Higher-quality local usage

- **`--num-steps 40`**

Why:

- it remains the quality anchor used for comparison in this repo
- lower-step settings improve latency, but `24` is the first one that approaches it consistently rather than matching it outright

## Conclusion

- the current default `--num-steps 40` is still a reasonable **higher-quality** preset
- it is **not** the most practical default for interactive local Apple Silicon iteration
- for local iteration, this sweep supports moving the day-to-day recommendation to:
  - **`12`** when latency matters most
  - **`24`** when you want a better quality/speed balance

## Raw per-case reports

- [v3 text-only detailed report](2026-05-14-apple-silicon-num-steps-v3-text.md)
- [v3 reference detailed report](2026-05-14-apple-silicon-num-steps-v3-reference.md)
- [VoiceDesign detailed report](2026-05-14-apple-silicon-num-steps-voicedesign.md)
