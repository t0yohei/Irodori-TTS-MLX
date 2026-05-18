# Apple Silicon benchmark workflow and report

Issue: [#13 Add benchmark script and baseline report](https://github.com/t0yohei/Irodori-TTS-MLX/issues/13)

This document has two jobs:

1. define the reproducible benchmark command surface for upstream PyTorch and the MLX bridge prototype
2. record the current decision baseline for whether a full MLX DACVAE port is worth prioritizing

## Current conclusion

We now have both:

- the measured upstream PyTorch/MPS baseline from [docs/baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md](baseline-reports/2026-05-11-apple-silicon-pytorch-baseline.md)
- the measured MLX bridge report from [docs/benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md](benchmark-reports/2026-05-12-apple-silicon-mlx-bridge.md)
- the warm-cache / codec-device / memory follow-up from [docs/benchmark-reports/2026-05-12-apple-silicon-mlx-followup.md](benchmark-reports/2026-05-12-apple-silicon-mlx-followup.md)
- the reference-path memory mitigation follow-up from [docs/benchmark-reports/2026-05-12-apple-silicon-memory-residency-mitigation.md](benchmark-reports/2026-05-12-apple-silicon-memory-residency-mitigation.md)
- the local Apple Silicon `num_steps` preset sweep for v3 + VoiceDesign from [docs/benchmark-reports/2026-05-14-apple-silicon-num-steps-presets.md](benchmark-reports/2026-05-14-apple-silicon-num-steps-presets.md)
- the real hosted/pre-converted weights loading measurement from [docs/benchmark-reports/2026-05-16-apple-silicon-hosted-weights.md](benchmark-reports/2026-05-16-apple-silicon-hosted-weights.md)
- the codec runtime mode comparison from [docs/benchmark-reports/2026-05-18-apple-silicon-codec-runtime-modes.md](benchmark-reports/2026-05-18-apple-silicon-codec-runtime-modes.md)
- the persistent batch generation measurement from [docs/benchmark-reports/2026-05-18-apple-silicon-persistent-batch.md](benchmark-reports/2026-05-18-apple-silicon-persistent-batch.md)
- the persistent batch cleanup comparison from [docs/benchmark-reports/2026-05-18-apple-silicon-persistent-batch-cleanup-comparison.md](benchmark-reports/2026-05-18-apple-silicon-persistent-batch-cleanup-comparison.md)
- the persistent batch runtime decode cleanup comparison from [docs/benchmark-reports/2026-05-18-apple-silicon-persistent-batch-runtime-cleanup-comparison.md](benchmark-reports/2026-05-18-apple-silicon-persistent-batch-runtime-cleanup-comparison.md)
- persistent batch generation is now documented in [dacvae_bridge.md](dacvae_bridge.md); the old one-off persistent-batch report was removed because no current summary or test referenced it

Current read:

- MLX RF-DiT + PyTorch DACVAE bridge already reduces `sample_rf` dramatically on Apple Silicon
- end-to-end `total_to_decode` also improves materially even before a full DACVAE port
- therefore, a full MLX DACVAE port is still **not** the first latency optimization priority
- warm-cache reruns are even faster than the first MLX report suggested
- switching `codec-device` from `mps` to `cpu` did not reduce reference-path RSS in this setup, so the likely issue was never just the codec backend alone
- explicit post-stage PyTorch cleanup reduced reference-path peak RSS from about **3.36 GiB** to about **2.10 GiB** in the measured setup
- an experimental helper-process DACVAE boundary did **not** materially improve memory beyond that cleanup, and it made latency much worse

In short: the bridge architecture is already good enough to justify continued optimization on the MLX model/sampler path first, while keeping DACVAE porting as a later optimization if memory or remaining decode cost becomes dominant.

For day-to-day local generation defaults, the later `num_steps` sweep now suggests a more specific operating point:

- `--num-steps 12` for fastest acceptable local iteration
- `--num-steps 24` for balanced local usage
- keep `--num-steps 40` as the higher-quality comparison/default anchor when latency matters less

The v0.2 hosted weights measurement shows that hosted loading is a setup/UX improvement rather than a generation-latency optimization: the first hosted run is dominated by the artifact download, while warm hosted repo, local hosted-layout directory, and direct local `.npz` fallback all produce similar `sample_rf` and `total_to_decode` timings.

The v0.2 codec runtime mode measurement shows a similar split: MLX codec artifacts do not change RF sampling time, but they can reduce PyTorch dependency surface and peak RSS. In the measured v3 reference-audio run, full `mlx` encode/decode reduced warm max RSS from about 4.33 GiB to 3.08 GiB versus the PyTorch bridge and lowered codec encode/decode timings. Treat this as a deployment and memory-pressure improvement first; larger repeated runs are still needed before claiming a broad codec-speed win.

The persistent batch measurement shows that simply reusing one runtime is not enough by itself to claim a server/worker speed win. A five-request `mlx-decode` batch amortized setup modestly at the process level, but request-level `decode_dacvae` drifted upward after the first request. A follow-up cleanup comparison found that `--cleanup-between-requests` stabilized measured `decode_dacvae` from a 717.4-1231.2 ms range to a 712.8-745.2 ms range and improved measured generation throughput from 0.522 req/s to 0.624 req/s in that run. The runtime decode cleanup fix then scoped that request-boundary cleanup to `MLXDACVAEBridge.decode_to_wav`: the #213 rerun stabilized measured `decode_dacvae` at 782.8-854.0 ms without enabling the benchmark-level cleanup switch and kept max RSS at 2.92 GiB. Keep `--cleanup-between-requests` as a diagnostic switch for future request-local residency issues outside codec decode.

The first-audio investigation is documented in [first_audio_latency.md](first_audio_latency.md). For the current short Japanese no-reference v3 path, first playable audio is not earlier than complete-WAV availability: RF-DiT sampling produces a full latent sequence, the DACVAE boundary decodes the full sequence, and the current public `decode_dacvae` timing is an inclusive decode-to-WAV measurement. Until a future segment or chunk path has explicit first-audio evidence, the sub-second target should be complete WAV available from a warmed persistent request.

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
For environment setup, use the packaged benchmark flow in [docs/packaging.md](packaging.md): Python 3.11 through 3.14 are supported for packaging, while the benchmark examples continue to use Python 3.11 as the reference environment. Create a venv, install `-e ".[bench]"`, and make upstream `irodori_tts` importable from the same environment.

Use scripts/benchmark_persistent_batch.py when the question is repeated generation through one initialized runtime. It wraps scripts/generate_wav.py --requests-json, records one process-level wall clock and max RSS, and summarizes first-request, warmup, measured steady-state, and throughput metrics. This is the right harness for estimating the benefit of a future persistent server or worker because model, tokenizer, hosted artifact, and codec setup are paid once for the whole batch instead of once per output.

Add `--cleanup-between-requests` to the persistent batch benchmark to forward an explicit MLX request boundary into `scripts/generate_wav.py`. The boundary synchronizes MLX, runs Python garbage collection, and clears reusable MLX cache memory after each generated request. Use it to test whether repeated-request latency drift is caused by request-local MLX residency outside the normal runtime cleanup boundaries.

### Self-test

```bash
python3 scripts/benchmark.py --self-test
```

This validates timing parsing and report generation without any model dependencies.

Persistent batch self-test:

    python3 scripts/benchmark_persistent_batch.py --self-test

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

The MLX benchmark accepts a converted `.npz` checkpoint from `scripts/convert_weights.py`, a local hosted/pre-converted weights layout directory, or a Hugging Face repo id with the hosted weights layout. It also expects an environment that can import the `.[bench]` dependency group plus upstream `irodori_tts`.
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

Hosted/pre-converted repo example:

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --seconds 2 \
  --num-steps 24 \
  --repeat 2 \
  --cache-state auto \
  --case-label hosted-repo-voicedesign \
  --output-dir benchmark-runs/hosted-repo \
  --report docs/benchmark-latest.md
```

Local hosted-layout directory example:

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights-dir /path/to/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --seconds 2 \
  --num-steps 24 \
  --case-label local-hosted-layout-voicedesign
```

Persistent hosted batch example:

    PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH python3 scripts/benchmark_persistent_batch.py --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-runtime-mode mlx-decode --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 --text '今日はいい天気ですね。' --omit-seconds --num-steps 12 --warmup-requests 1 --requests 4 --case-label v3-mlx-decode-persistent-batch --output-dir benchmark-runs/persistent-batch-v3 --report docs/benchmark-latest-persistent-batch.md

The persistent batch report includes process throughput, measured generation throughput, setup/load overhead for the whole process, and request-level total_to_decode, sample_rf, encode_dacvae, and decode_dacvae aggregates. The process wall/RSS values are intentionally not copied onto each request; they belong to the whole batch.

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

### Predicted-duration and caption-conditioned MLX cases

Two extra options are useful when the benchmark target is not the base v2 speaker-conditioned path:

- `--omit-seconds`: do not pass `--seconds` to `scripts/generate_wav.py`, so v3 checkpoints can use predicted duration
- `--caption "..."`: add caption/style conditioning for VoiceDesign runs
- `--case-label NAME`: prefix case names/log slugs so separate runs like `v3` vs `voicedesign-caption` stay distinguishable in reports

Example: v3 predicted-duration sweep

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights /path/to/irodori-v3.npz \
  --model-config-json /path/to/v3-model-config.json \
  --upstream-root /path/to/Irodori-TTS \
  --case-label v3 \
  --omit-seconds \
  --num-steps-sweep 8,12,16,24,40
```

Example: VoiceDesign caption-conditioned sweep

```bash
python3 scripts/benchmark.py \
  --mode mlx \
  --weights /path/to/irodori-voicedesign.npz \
  --model-config-json /path/to/voicedesign-model-config.json \
  --upstream-root /path/to/Irodori-TTS \
  --case-label voicedesign-caption \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --seconds 2 \
  --num-steps-sweep 8,12,16,24,40
```

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
