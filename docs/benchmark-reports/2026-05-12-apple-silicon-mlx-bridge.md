# Apple Silicon MLX bridge benchmark report: 2026-05-12

Related issue: [#13](https://github.com/t0yohei/Irodori-TTS-MLX/issues/13)

## Summary

- Status: passed
- Host: Apple Silicon Mac, macOS 26.x, arm64
- Runtime shape: MLX RF-DiT + PyTorch DACVAE bridge
- Python environment used for the MLX run: dedicated local benchmark venv, Python 3.11
- Main result:
  - no-reference MLX bridge `sample_rf` was **4,227.3 ms** vs upstream PyTorch/MPS **23,713.9 ms**
  - no-reference MLX bridge `total_to_decode` was **5,529.5 ms** vs upstream PyTorch/MPS **29,367.0 ms**
  - reference-audio MLX bridge `sample_rf` was **4,095.8 ms** vs upstream PyTorch/MPS **24,285.4 ms**
  - reference-audio MLX bridge `total_to_decode` was **6,398.1 ms** vs upstream PyTorch/MPS **31,327.0 ms**
- Current conclusion:
  - the MLX bridge already delivers a large model/sampler-side speedup on Apple Silicon
  - a full MLX DACVAE port is still **not** the first optimization priority for latency alone
  - reference-path memory usage is worth watching because the mixed runtime peaked higher than the earlier upstream reference-audio baseline

## Inputs

### Upstream checkout

- Path: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/external/Irodori-TTS`

### Model checkpoint

- Source checkpoint: `Aratako/Irodori-TTS-500M-v2`
- Local source path: `/Users/kouka/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2/snapshots/8fd631cafb911dde466bc30dd558a0dc55e8ccae/model.safetensors`
- Converted MLX weights: `benchmark-runs/irodori-tts-500m-v2.npz`

### Codec checkpoint

- `Aratako/Semantic-DACVAE-Japanese-32dim`

### Reference audio

- Path: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/baseline-runs/base-no-ref-seed-20260511.wav`
- Origin: local no-reference upstream baseline artifact from the previous measured baseline report
- Sharing status: local-only, not committed

## Commands used

### Convert weights

```bash
. .venv-bench311/bin/activate
python scripts/convert_weights.py \
  /Users/kouka/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2/snapshots/8fd631cafb911dde466bc30dd558a0dc55e8ccae/model.safetensors \
  benchmark-runs/irodori-tts-500m-v2.npz
```

### MLX bridge, no reference audio

```bash
. .venv-bench311/bin/activate
python scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/.venv-bench311/bin/python \
  --weights /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/benchmark-runs/irodori-tts-500m-v2.npz \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/external/Irodori-TTS \
  --codec-device mps \
  --output-dir benchmark-runs/mlx-no-ref
```

### MLX bridge, reference audio

```bash
. .venv-bench311/bin/activate
python scripts/benchmark.py \
  --mode mlx \
  --mlx-python /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/.venv-bench311/bin/python \
  --weights /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-13-benchmark/benchmark-runs/irodori-tts-500m-v2.npz \
  --upstream-root /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/external/Irodori-TTS \
  --reference-wav /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-2-measured-baseline/baseline-runs/base-no-ref-seed-20260511.wav \
  --codec-device mps \
  --output-dir benchmark-runs/mlx-ref
```

## Results

### No-reference comparison

| Metric | Upstream PyTorch/MPS | MLX bridge | Delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `sample_rf` | 23,713.9 ms | 4,227.3 ms | -19,486.6 ms | 5.61x faster |
| `decode` | 5,648.5 ms | 1,300.7 ms | -4,347.8 ms | 4.34x faster |
| `total_to_decode` | 29,367.0 ms | 5,529.5 ms | -23,837.5 ms | 5.31x faster |
| wall clock | 122.86 s | 35.72 s | -87.14 s | 3.44x faster |
| max RSS | 1.60 GiB | 1.91 GiB | +0.31 GiB | 1.19x higher |

MLX bridge stage breakdown:

| Stage | Time |
| --- | ---: |
| `prepare_text_condition` | 1.4 ms |
| `prepare_reference_condition` | 0.1 ms |
| `sample_rf` | 4,227.3 ms |
| `decode_dacvae` | 1,300.7 ms |
| `total_to_decode` | 5,529.5 ms |

### Reference-audio comparison

| Metric | Upstream PyTorch/MPS | MLX bridge | Delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `prepare_reference` / `prepare_reference_condition` | 1,399.6 ms | 1,310.7 ms | -88.9 ms | 1.07x faster |
| `sample_rf` | 24,285.4 ms | 4,095.8 ms | -20,189.6 ms | 5.93x faster |
| `decode` | 5,637.5 ms | 988.4 ms | -4,649.1 ms | 5.70x faster |
| `total_to_decode` | 31,327.0 ms | 6,398.1 ms | -24,928.9 ms | 4.90x faster |
| wall clock | 40.77 s | 15.15 s | -25.62 s | 2.69x faster |
| max RSS | 2.06 GiB | 3.36 GiB | +1.30 GiB | 1.63x higher |

MLX bridge stage breakdown:

| Stage | Time |
| --- | ---: |
| `prepare_text_condition` | 3.2 ms |
| `prepare_reference_condition` | 1,310.7 ms |
| `sample_rf` | 4,095.8 ms |
| `decode_dacvae` | 988.4 ms |
| `total_to_decode` | 6,398.1 ms |

## Interpretation

### What improved clearly

- The expected win showed up exactly where it should: `sample_rf`
- The bridge is already much faster than the earlier PyTorch/MPS baseline on both no-reference and reference-audio paths
- Even with the PyTorch DACVAE bridge still in place, end-to-end `total_to_decode` improved by about **5.3x** (no-ref) and **4.9x** (ref)

### What still needs attention

- Reference-path peak RSS increased substantially versus the earlier upstream baseline
- This means latency alone does **not** justify prioritizing a full DACVAE MLX port, but memory pressure might become a stronger argument later
- The no-reference wall-clock result is still much larger than `total_to_decode`, so startup/model-load overhead remains visible in cold-ish runs and should stay separate from steady-state latency comparisons

## Recommendation

- Keep the current priority on the MLX bridge / RF-DiT path rather than jumping straight to a full DACVAE port
- Next benchmark focus should be:
  1. repeat warm-cache MLX runs to separate one-time load overhead from steady-state latency
  2. investigate reference-path memory growth
  3. compare `codec-device mps` versus `codec-device cpu` for mixed-runtime behavior
  4. benchmark multiple output lengths to see how the gain scales with sequence length
