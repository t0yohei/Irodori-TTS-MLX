# Apple Silicon MLX benchmark follow-up: warm-cache, codec-device comparison, and reference-path memory notes

Related issue: [#13](https://github.com/t0yohei/Irodori-TTS-MLX/issues/13)

This report follows up on the first measured MLX bridge benchmark in [2026-05-12 Apple Silicon MLX bridge benchmark](2026-05-12-apple-silicon-mlx-bridge.md).

## Questions

1. How much better does the MLX bridge look once caches are warm?
2. Does `codec-device=cpu` reduce memory pressure versus `codec-device=mps`?
3. What does that imply about the likely source of reference-path memory growth?

## Setup

Common inputs:

- repo worktree: `/path/to/Irodori-TTS-MLX`
- MLX weights: `benchmark-runs/irodori-tts-500m-v2.npz`
- upstream checkout: `/path/to/Irodori-TTS-MLX/external/Irodori-TTS`
- reference audio: `/path/to/Irodori-TTS-MLX/baseline-runs/base-no-ref-seed-20260511.wav`
- benchmark Python: `.venv-bench311/bin/python`
- prompt: `今日はいい天気ですね。`
- seed: `20260512`
- steps: `40`

## Warm-cache rerun results

### No-reference, `codec-device=mps`

| Metric | First measured run | Warm-cache rerun | Delta |
| --- | ---: | ---: | ---: |
| `sample_rf` | 4,227.3 ms | 3,402.6 ms | -824.7 ms |
| `decode_dacvae` | 1,300.7 ms | 1,206.6 ms | -94.1 ms |
| `total_to_decode` | 5,529.5 ms | 4,612.2 ms | -917.3 ms |
| wall clock | 35.72 s | 13.93 s | -21.79 s |
| max RSS | 1.91 GiB | 2.47 GiB | +0.56 GiB |

Warm-cache takeaway:

- steady-state latency is better than the first MLX run suggested
- no-reference `total_to_decode` settles around **4.6 s** for this 5-second sample
- the large wall-clock drop indicates one-time startup / load cost is still significant in first-run measurements

### Reference-audio, `codec-device=mps`

| Metric | First measured run | Warm-cache rerun | Delta |
| --- | ---: | ---: | ---: |
| `prepare_reference_condition` | 1,310.7 ms | 459.3 ms | -851.4 ms |
| `sample_rf` | 4,095.8 ms | 3,460.9 ms | -634.9 ms |
| `decode_dacvae` | 988.4 ms | 1,024.2 ms | +35.8 ms |
| `total_to_decode` | 6,398.1 ms | 4,945.2 ms | -1,452.9 ms |
| wall clock | 15.15 s | 12.81 s | -2.34 s |
| max RSS | 3.36 GiB | 3.36 GiB | essentially unchanged |

Warm-cache takeaway:

- reference-audio `total_to_decode` improves to about **4.95 s**
- the biggest warm-cache gain on the reference path came from `prepare_reference_condition`
- peak memory stayed high even after caches were warm, so the reference-path RSS issue is not just a cold-start effect

## `codec-device=cpu` vs `codec-device=mps` on the reference path

### Reference-audio comparison

| Metric | `codec-device=mps` warm | `codec-device=cpu` | Delta (cpu - mps) |
| --- | ---: | ---: | ---: |
| `prepare_reference_condition` | 459.3 ms | 1,142.4 ms | +683.1 ms |
| `sample_rf` | 3,460.9 ms | 3,448.2 ms | -12.7 ms |
| `decode_dacvae` | 1,024.2 ms | 1,226.2 ms | +202.0 ms |
| `total_to_decode` | 4,945.2 ms | 5,817.9 ms | +872.7 ms |
| wall clock | 12.81 s | 12.53 s | -0.28 s |
| max RSS | 3.36 GiB | 3.92 GiB | +0.56 GiB |

Interpretation:

- moving the DACVAE codec to CPU did **not** reduce peak RSS in this setup
- it also made `prepare_reference_condition` and `decode_dacvae` slower
- `sample_rf` stayed essentially unchanged, which is expected because the RF-DiT path remains on MLX

## Likely explanation for reference-path memory growth

The new comparison rules out the simplest hypothesis that "the MPS codec backend is the main reason reference-path RSS is high."

A more plausible explanation is the combination of:

1. MLX model weights + MLX activations remaining resident for RF-DiT sampling
2. PyTorch DACVAE model state being resident in the same process
3. extra reference-audio encode intermediates and tensors on the reference path
4. framework-specific caching / allocator behavior when MLX and PyTorch coexist in one process

In other words, the memory pressure looks more like a **mixed-runtime residency problem** than a pure `codec-device=mps` problem.

## Practical conclusions

- For latency, the MLX bridge is already clearly successful
- For memory, switching the codec to CPU is **not** a free fix
- The next memory-focused work should probably target architecture or lifecycle decisions rather than just toggling codec backend, for example:
  - releasing temporary reference tensors more aggressively after conditioning
  - checking whether the DACVAE instance can be loaded lazily or moved into a shorter-lived worker process
  - measuring whether longer outputs amplify the same peak-RSS pattern
  - testing whether explicit cache-clearing / synchronization points change peak memory behavior

## Recommendation

- Keep `codec-device=mps` as the default benchmark path for now
- Treat reference-path memory investigation as a separate optimization track from raw latency optimization
- Do not prioritize a full MLX DACVAE port solely based on these results yet; first verify whether mixed-runtime residency can be reduced with smaller architectural changes
