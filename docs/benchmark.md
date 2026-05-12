# Apple Silicon benchmark workflow and report

Issue: [#13 Add benchmark script and baseline report](https://github.com/t0yohei/irodori-tts-mlx/issues/13)

This document has two jobs:

1. define the reproducible benchmark command surface for upstream PyTorch and the MLX bridge prototype
2. record the current decision baseline for whether a full MLX DACVAE port is worth prioritizing

## Current conclusion

We now have both:

- the measured upstream PyTorch/MPS baseline from [docs/baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md](baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md)
- the measured MLX bridge report from [docs/benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md](benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md)

Current read:

- MLX RF-DiT + PyTorch DACVAE bridge already reduces `sample_rf` dramatically on Apple Silicon
- end-to-end `total_to_decode` also improves materially even before a full DACVAE port
- therefore, a full MLX DACVAE port is still **not** the first latency optimization priority
- however, reference-path peak RSS increased enough that memory pressure remains a real follow-up question

In short: the bridge architecture is already good enough to justify continued optimization on the MLX model/sampler path first, while keeping DACVAE porting as a later optimization if memory or remaining decode cost becomes dominant.

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

## Measured MLX bridge numbers

Full report: [2026-05-12 Apple Silicon MLX bridge benchmark](benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md)

### Run 1: base model, no reference audio

| Stage | Time |
| --- | ---: |
| `prepare_text_condition` | 1.4 ms |
| `prepare_reference_condition` | 0.1 ms |
| `sample_rf` | 4,227.3 ms |
| `decode_dacvae` | 1,300.7 ms |
| `total_to_decode` | 5,529.5 ms |
| wall clock | 35.72 s |

Compared with the upstream no-reference baseline:

- `sample_rf`: **5.61x faster**
- `decode`: **4.34x faster**
- `total_to_decode`: **5.31x faster**
- max RSS: **1.91 GiB**, about **0.31 GiB higher** than the earlier upstream no-ref baseline

### Run 2: base model, reference audio

| Stage | Time |
| --- | ---: |
| `prepare_text_condition` | 3.2 ms |
| `prepare_reference_condition` | 1,310.7 ms |
| `sample_rf` | 4,095.8 ms |
| `decode_dacvae` | 988.4 ms |
| `total_to_decode` | 6,398.1 ms |
| wall clock | 15.15 s |

Compared with the upstream reference-audio baseline:

- `prepare_reference`: slightly faster
- `sample_rf`: **5.93x faster**
- `decode`: **5.70x faster**
- `total_to_decode`: **4.90x faster**
- max RSS: **3.36 GiB**, about **1.30 GiB higher** than the earlier upstream reference-audio baseline

So the latency question is largely answered: the MLX bridge is already a clear win. The open question that remains is memory behavior, especially on the reference path.

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

The next meaningful report should deepen or validate these measured MLX results:

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
