# Apple Silicon MLX bridge memory residency follow-up

Issue: [#29 Investigate and reduce reference-path memory residency in the MLX bridge](https://github.com/t0yohei/irodori-tts-mlx/issues/29)

This follow-up focused on one question:

> can we materially reduce reference-path peak RSS without doing a full DACVAE MLX port?

## Change under test

Two changes were evaluated on top of the existing MLX bridge runtime:

1. explicit post-stage PyTorch runtime cleanup after reference encode and DACVAE decode
2. an experimental `--codec-runtime-mode subprocess` path that runs PyTorch DACVAE encode/decode in short-lived helper processes instead of keeping the codec in the main MLX process

The second change is intentionally experimental. It exists to test the "helper-process boundary" hypothesis from issue #29, not because it is assumed to be the final architecture.

## Benchmark setup

- machine: same Apple Silicon benchmark host used for the earlier bridge reports
- model path: converted `Aratako/Irodori-TTS-500M-v2` MLX weights
- codec repo: `Aratako/Semantic-DACVAE-Japanese-32dim`
- codec device: `mps`
- prompt: `今日はいい天気ですね。`
- reference WAV: upstream baseline no-ref output reused as the speaker sample
- sampler steps: `40`
- seed: `20260512`

## Results

### Earlier warm-cache reference-path baseline

From the earlier report in [`2026-05-12-apple-silicon-mlx-followup.md`](2026-05-12-apple-silicon-mlx-followup.md):

| Metric | Earlier warm-cache persistent bridge |
| --- | ---: |
| `prepare_reference_condition` | 459.3 ms |
| `sample_rf` | 3,460.9 ms |
| `decode_dacvae` | 1,024.2 ms |
| `total_to_decode` | 4,945.2 ms |
| max RSS | 3.36 GiB |

### New persistent bridge with explicit cleanup

Command:

```bash
/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/.venv-bench311/bin/python \
  scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/.venv-bench311/bin/python \
  --weights /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/benchmark-runs/irodori-tts-500m-v2.npz \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/external/Irodori-TTS \
  --reference-wav /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/baseline-runs/base-no-ref-seed-20260511.wav \
  --codec-device mps \
  --codec-runtime-mode persistent \
  --output-dir benchmark-runs/issue-29-persistent
```

Observed result:

| Metric | New persistent bridge |
| --- | ---: |
| `prepare_reference_condition` | 510.6 ms |
| `sample_rf` | 3,532.4 ms |
| `decode_dacvae` | 1,117.7 ms |
| `total_to_decode` | 5,162.9 ms |
| max RSS | 2.10 GiB |

Delta vs the earlier warm-cache persistent run:

- `total_to_decode`: **+217.7 ms**
- max RSS: **-1.27 GiB**

### Experimental subprocess bridge

Command:

```bash
/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/.venv-bench311/bin/python \
  scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/.venv-bench311/bin/python \
  --weights /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/benchmark-runs/irodori-tts-500m-v2.npz \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/external/Irodori-TTS \
  --reference-wav /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/baseline-runs/base-no-ref-seed-20260511.wav \
  --codec-device mps \
  --codec-runtime-mode subprocess \
  --output-dir benchmark-runs/issue-29-subprocess
```

Observed result:

| Metric | Subprocess bridge |
| --- | ---: |
| `prepare_reference_condition` | 3,039.9 ms |
| `sample_rf` | 3,519.3 ms |
| `decode_dacvae` | 5,292.2 ms |
| `total_to_decode` | 11,854.0 ms |
| max RSS | 2.14 GiB |

Delta vs the new persistent-cleanup run:

- `total_to_decode`: **+6.69 s**
- max RSS: **+0.05 GiB**

## Interpretation

The main memory win did **not** come from splitting encode/decode into helper processes.

Instead, the evidence points more strongly at this simpler explanation:

- the earlier high RSS was inflated by PyTorch-side tensors / runtime allocations remaining resident longer than necessary after encode/decode
- explicitly dropping those tensors and flushing backend caches after each bridge stage removes most of that extra residency
- a helper-process boundary does not materially improve the measured max RSS beyond that cleanup, while it imposes a large serialization and model-reload penalty

That means issue #29 now has a concrete mitigation with evidence:

- **keep the default bridge in-process**
- **release PyTorch-stage intermediates eagerly after encode/decode**
- treat helper-process isolation as a useful experiment, but not the preferred default path

## Current recommendation

For the MLX bridge prototype:

1. keep `codec-runtime-mode=persistent` as the default
2. keep the explicit cleanup path in place
3. retain `codec-runtime-mode=subprocess` only as an investigation / benchmarking switch
4. continue tracking whether longer or larger reference runs reintroduce memory pressure

## What this rules out

This result does not prove that a future full MLX DACVAE port is never useful. It only narrows the current priority:

- a full DACVAE port is **no longer the first memory fix to try**
- same-process lifecycle cleanup already recovers most of the observed reference-path RSS spike in this benchmark setup
- helper-process isolation is not an attractive default tradeoff right now
