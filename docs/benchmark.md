# Apple Silicon benchmark workflow and report

Issue: [#13 Add benchmark script and baseline report](https://github.com/t0yohei/irodori-tts-mlx/issues/13)

This document has two jobs:

1. define the reproducible benchmark command surface for upstream PyTorch and the MLX bridge prototype
2. record the current decision baseline for whether a full MLX DACVAE port is worth prioritizing

## Current conclusion

We now have both:

- the measured upstream PyTorch/MPS baseline from [docs/baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md](baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md)
- the measured MLX bridge report from [docs/benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md](benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md)
- the warm-cache / codec-device / memory follow-up from [docs/benchmark-reports/2026-05-12-apple-silicon-mlx-followup.md](benchmark-reports/2026-05-12-apple-silicon-mlx-followup.md)
- the reference-path memory mitigation follow-up from [docs/benchmark-reports/2026-05-12-apple-silicon-memory-residency-mitigation.md](benchmark-reports/2026-05-12-apple-silicon-memory-residency-mitigation.md)

Current read:

- MLX RF-DiT + PyTorch DACVAE bridge already reduces `sample_rf` dramatically on Apple Silicon
- end-to-end `total_to_decode` also improves materially even before a full DACVAE port
- therefore, a full MLX DACVAE port is still **not** the first latency optimization priority
- warm-cache reruns are even faster than the first MLX report suggested
- switching `codec-device` from `mps` to `cpu` did not reduce reference-path RSS in this setup, so the likely issue was never just the codec backend alone
- explicit post-stage PyTorch cleanup reduced reference-path peak RSS from about **3.36 GiB** to about **2.10 GiB** in the measured setup
- an experimental helper-process DACVAE boundary did **not** materially improve memory beyond that cleanup, and it made latency much worse

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

## Warm-cache and codec-device follow-up

Follow-up report: [2026-05-12 Apple Silicon MLX benchmark follow-up](benchmark-reports/2026-05-12-apple-silicon-mlx-followup.md)

Key follow-up results:

- no-reference warm-cache `total_to_decode`: **4,612.2 ms**
- reference-audio warm-cache `total_to_decode`: **4,945.2 ms**
- reference-audio `codec-device=cpu` `total_to_decode`: **5,817.9 ms**
- reference-audio warm-cache max RSS with `codec-device=mps`: **3.36 GiB**
- reference-audio max RSS with `codec-device=cpu`: **3.92 GiB**

This sharpens the interpretation:

- warm-cache behavior strengthens the case that the MLX bridge is already a strong steady-state latency win
- the reference-path memory issue survives warm-cache reruns
- CPU codec fallback is not a simple memory fix here; it was slower and used more peak memory in this measurement

## Reference-path memory mitigation follow-up

Mitigation report: [2026-05-12 Apple Silicon MLX bridge memory residency follow-up](benchmark-reports/2026-05-12-apple-silicon-memory-residency-mitigation.md)

Two hypotheses were tested after the earlier follow-up:

1. would explicit lifecycle cleanup after PyTorch DACVAE encode/decode reduce the persistent bridge RSS?
2. would a helper-process DACVAE boundary reduce memory more effectively than staying in-process?

Key measured results on the reference path (`codec-device=mps`):

| Variant | `total_to_decode` | max RSS |
| --- | ---: | ---: |
| earlier warm-cache persistent bridge | 4,945.2 ms | 3.36 GiB |
| new persistent bridge + explicit cleanup | 5,162.9 ms | 2.10 GiB |
| experimental subprocess bridge | 11,854.0 ms | 2.14 GiB |

This changes the interpretation in an important way:

- the largest practical memory win came from **eagerly releasing PyTorch-stage tensors and backend cache state** after encode/decode
- the helper-process boundary did **not** buy additional memory relief worth its cost
- the subprocess experiment is still useful as a diagnostic switch, but not as the default runtime shape

So the current best mitigation is a simpler one than the original architectural hypothesis: keep the normal bridge in-process, but end each DACVAE stage with explicit cleanup.

## Benchmark script

Use `scripts/benchmark.py` to run a reproducible benchmark harness with repeated-run support, warmup labeling, and simple scaling sweeps.
For environment setup, use the packaged benchmark flow in [docs/packaging.md](packaging.md): Python 3.11 and 3.12 are supported for packaging, while the benchmark examples continue to use Python 3.11 as the reference environment. Create a venv, install `-e ".[bench]"`, and make upstream `irodori_tts` importable from the same environment.

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

The MLX benchmark expects a converted `.npz` checkpoint from `scripts/convert_weights.py` and an environment that can import the `.[bench]` dependency group plus upstream `irodori_tts`.
That means the benchmark venv should contain this repository with `pip install -e ".[bench]"`, while upstream Irodori-TTS is either installed into the same venv or provided on `PYTHONPATH`.

Example:

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --upstream-root /path/to/Irodori-TTS \
  --reference-wav /path/to/reference.wav \
  --codec-device cpu \
  --codec-runtime-mode persistent \
  --output-dir benchmark-runs \
  --report docs/benchmark-latest.md
```

For no-reference benchmarking, omit `--reference-wav`.

### Repeated runs and warm-cache tracking

Use `--repeat` to collect multiple measured runs for the same case. Add `--warmup-runs` when you want the harness to intentionally push the runtime toward steady state before recording measured runs.

Example:

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --upstream-root /path/to/Irodori-TTS \
  --reference-wav /path/to/reference.wav \
  --repeat 3 \
  --warmup-runs 1 \
  --output-dir benchmark-runs \
  --report docs/benchmark-latest.md
```

Cache labeling options:

- `--cache-state auto` (default): heuristically labels the first run in an invocation as cold and later steady-state runs as warm when that distinction is meaningful
- `--cache-state cold|warm|unknown`: override the label when you know the environment state better than the harness does

Warmup runs are recorded separately from measured runs in both the Markdown report and the JSON summary so one-shot startup effects do not get mixed into the steady-state aggregate.

### Scaling sweeps

Use `--num-steps-sweep` to compare multiple diffusion-step counts in one invocation.

```bash
python3 scripts/benchmark.py \
  --mode both \
  --upstream-root /path/to/Irodori-TTS \
  --upstream-python /path/to/Irodori-TTS/.venv/bin/python \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --reference-wav /path/to/reference.wav \
  --num-steps-sweep 20,40,60 \
  --repeat 2
```

Use `--seconds-sweep` for MLX-only output-length scaling runs:

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --upstream-root /path/to/Irodori-TTS \
  --seconds-sweep 3,5,8 \
  --num-steps 40 \
  --repeat 2
```

`--seconds-sweep` is MLX-only because the upstream CLI does not currently expose an output-length flag.

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
- run metadata such as `phase` (`warmup` / `measured`), `cache_state`, `seconds`, and `num_steps`
- a JSON summary in `benchmark-runs/benchmark-summary.json`

The summary JSON now uses a structured schema with:

- `results`: raw per-run entries
- `aggregates`: grouped min / median / max summaries by case, phase, and cache state
- `invocation`: the CLI parameters used to produce the run set

This keeps repeated measurements diffable over time without throwing away the raw run-by-run evidence.

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
| Does bridge overhead erase the model-side gain? | Already looks unlikely, but longer / repeated runs should confirm. |
| Is max RSS better or worse than upstream MPS? | Memory pressure can decide practical usability on smaller Apple Silicon machines. |
| Does reference-audio conditioning change the win profile? | Yes; memory remains the main unresolved concern. |
| Can mixed-runtime residency be reduced without a full DACVAE port? | This is now the most important follow-up optimization question. |

## Decision rule for DACVAE port priority

Treat a full MLX DACVAE port as worth prioritizing only if at least one of these becomes true:

1. MLX RF-DiT bridge already wins on `sample_rf`, and DACVAE decode becomes the dominant remaining bottleneck
2. bridge conversion / CPU boundary overhead is large enough to cancel the RF-DiT gain
3. memory observations show the mixed MLX/PyTorch runtime is impractical without a single-framework DACVAE path

Until then, the project should keep optimizing and validating the bridge architecture first.
