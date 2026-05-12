# Apple Silicon benchmark workflow and initial report

Issue: [#13 Add benchmark script and baseline report](https://github.com/t0yohei/irodori-tts-mlx/issues/13)

This document has two jobs:

1. define the reproducible benchmark command surface for upstream PyTorch and the MLX bridge prototype
2. record the current decision baseline for whether a full MLX DACVAE port is worth prioritizing

## Current conclusion

Based on the measured upstream baseline from [docs/baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md](baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md):

- RF sampling dominates end-to-end latency on Apple Silicon MPS
- DACVAE decode is meaningful, but clearly secondary
- the next performance question is whether **MLX RF-DiT + PyTorch DACVAE bridge** materially reduces `sample_rf` and `total_to_decode`
- therefore, a full MLX DACVAE port is **not yet justified** by the current data alone

In short: if the MLX bridge cannot beat PyTorch mainly on the sampler/model path, porting DACVAE first is the wrong optimization target.

## Existing upstream baseline numbers

### Run 1: base model, no reference audio

| Stage | Time |
| --- | ---: |
| `tokenize_text` | 1.8 ms |
| `prepare_reference` | 0.7 ms |
| `sample_rf` | 23,713.9 ms |
| `decode_latent` | 5,648.5 ms |
| `total_to_decode` | 29,367.0 ms |
| wall clock | 122.86 s |

Derived share of post-load inference time:

- `sample_rf`: about **80.8%** of `total_to_decode`
- `decode_latent`: about **19.2%** of `total_to_decode`

### Run 2: base model, reference audio

| Stage | Time |
| --- | ---: |
| `tokenize_text` | 2.8 ms |
| `prepare_reference` | 1,399.6 ms |
| `sample_rf` | 24,285.4 ms |
| `decode_latent` | 5,637.5 ms |
| `total_to_decode` | 31,327.0 ms |
| wall clock | 40.77 s |

Derived share of post-load inference time:

- `sample_rf`: about **77.5%** of `total_to_decode`
- `decode_latent`: about **18.0%** of `total_to_decode`
- `prepare_reference`: about **4.5%** of `total_to_decode`

These numbers are enough to set the optimization priority: sampler/model work first, DACVAE later.

## Benchmark script

Use `scripts/benchmark.py` to run a reproducible benchmark harness.

### Self-test

```bash
python3 scripts/benchmark.py --self-test
```

This validates timing parsing and report generation without any model dependencies.

### Upstream PyTorch benchmark

```bash
python3 scripts/benchmark.py \
  --mode upstream \
  --upstream-root /path/to/Irodori-TTS \
  --upstream-python /path/to/Irodori-TTS/.venv/bin/python \
  --output-dir benchmark-runs \
  --report docs/benchmark-latest.md
```

Optional reference-audio run:

```bash
python3 scripts/benchmark.py \
  --mode upstream \
  --upstream-root /path/to/Irodori-TTS \
  --upstream-python /path/to/Irodori-TTS/.venv/bin/python \
  --reference-wav /path/to/reference.wav \
  --output-dir benchmark-runs \
  --report docs/benchmark-latest.md
```

### MLX bridge benchmark

The MLX benchmark expects a converted `.npz` checkpoint from `scripts/convert_weights.py` and an environment that can import:

- `mlx`
- `torch`
- `transformers`
- `sentencepiece`
- upstream `irodori_tts`

Example:

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --upstream-root /path/to/Irodori-TTS \
  --reference-wav /path/to/reference.wav \
  --codec-device cpu \
  --output-dir benchmark-runs \
  --report docs/benchmark-latest.md
```

For no-reference benchmarking, omit `--reference-wav`.

### Combined run

```bash
python3 scripts/benchmark.py \
  --mode both \
  --upstream-root /path/to/Irodori-TTS \
  --upstream-python /path/to/Irodori-TTS/.venv/bin/python \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --reference-wav /path/to/reference.wav \
  --output-dir benchmark-runs \
  --report docs/benchmark-latest.md
```

## What the script records

For each run, the benchmark harness stores:

- exact command line
- working directory
- stdout/stderr logs
- parsed `[timing]` stage breakdowns
- `/usr/bin/time -l` wall-clock and max RSS when available
- a JSON summary in `benchmark-runs/benchmark-summary.json`

The MLX bridge runtime now emits these benchmark-friendly timing keys:

- `prepare_text_condition`
- `prepare_reference_condition`
- `sample_rf`
- `decode_dacvae`
- `total_to_decode`

This keeps the bridge report aligned with the upstream `--show-timings` convention while still reflecting the MLX/PyTorch split.

## Next measurement to collect

The next meaningful report should answer this table with real MLX numbers:

| Question | Why it matters |
| --- | --- |
| Does MLX reduce `sample_rf` materially vs upstream MPS? | This is the main expected win. |
| Does bridge overhead erase the model-side gain? | If yes, DACVAE porting may become more important. |
| Is max RSS better or worse than upstream MPS? | Memory pressure can decide practical usability on smaller Apple Silicon machines. |
| Does reference-audio conditioning change the win profile? | Bridge overhead is higher in the reference path. |

## Decision rule for DACVAE port priority

Treat a full MLX DACVAE port as worth prioritizing only if at least one of these becomes true:

1. MLX RF-DiT bridge already wins on `sample_rf`, and DACVAE decode becomes the dominant remaining bottleneck
2. bridge conversion / CPU boundary overhead is large enough to cancel the RF-DiT gain
3. memory observations show the mixed MLX/PyTorch runtime is impractical without a single-framework DACVAE path

Until then, the project should keep optimizing and validating the bridge architecture first.
