# Persistent Batch MLX Cleanup Comparison

Issue: [#209 Investigate persistent batch MLX codec decode slowdown](https://github.com/t0yohei/Irodori-TTS-MLX/issues/209)

## Scenario

- RF-DiT weights: `t0yohei/Irodori-TTS-MLX-500M-v3@078ffb11ffad92e6dde237a6abef730f4341b359`
- DACVAE codec artifact: `t0yohei/Irodori-TTS-MLX-DACVAE-Codec@bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78`
- mode: `mlx-decode`
- requests: 1 warmup + 4 measured
- duration: predicted
- RF steps: 12

This compares the default persistent batch path against `--cleanup-between-requests`, which synchronizes MLX and clears reusable MLX cache memory after each generated request.

## Results

| Variant | Wall | Setup/load overhead | Max RSS | Measured total median | Measured sample median | Measured decode median | Decode measured min/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cleanup off | 19.72 s | 10192.0 ms | 0.85 GiB | 1918.2 ms | 873.8 ms | 1005.2 ms | 717.4 / 1231.2 ms |
| cleanup on | 14.35 s | 6289.4 ms | 2.92 GiB | 1602.0 ms | 831.1 ms | 735.7 ms | 712.8 / 745.2 ms |

Raw generated reports:

- [cleanup off](2026-05-18-apple-silicon-persistent-batch-cleanup-off.md)
- [cleanup on](2026-05-18-apple-silicon-persistent-batch-cleanup-on.md)

## Interpretation

The original slowdown is reproducible in the default persistent batch path: the later measured requests spend about 1.22 to 1.23 seconds in `decode_dacvae`, while earlier measured requests stay around 0.72 to 0.79 seconds.

Adding an explicit request boundary removes the observed decode drift in this run. With cleanup enabled, all measured `decode_dacvae` values stay in a narrow 0.71 to 0.75 second band, and measured generation throughput improves from 0.522 req/s to 0.624 req/s.

The likely source is request-to-request MLX execution/cache residency rather than RF-DiT sampling itself. `sample_rf` was already stable in the default run, and the largest change appears in `decode_dacvae`.

The cleanup switch should remain opt-in for now. This run reported higher max RSS with cleanup enabled, so the evidence supports it as a persistent worker mitigation and diagnostic mode, not as a universal default.

## Recommendation

Use `--cleanup-between-requests` when benchmarking or operating long-lived persistent workers where stable repeated-request latency matters more than preserving the default MLX cache behavior. Keep the default unchanged until repeated measurements clarify the RSS tradeoff.
