# Persistent Batch Runtime Decode Cleanup Comparison

Issue: [#213 Fix repeated MLX DACVAE decode residency in persistent batch](https://github.com/t0yohei/Irodori-TTS-MLX/issues/213)

## Scenario

- RF-DiT weights: `t0yohei/Irodori-TTS-MLX-500M-v3@078ffb11ffad92e6dde237a6abef730f4341b359`
- DACVAE codec artifact: `t0yohei/Irodori-TTS-MLX-DACVAE-Codec@bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78`
- mode: `mlx-decode`
- requests: 1 warmup + 4 measured
- duration: predicted
- RF steps: 12

This compares the previous default persistent batch path against the runtime decode cleanup fix. The fix materializes MLX DACVAE decode outputs, writes the WAV, then releases decode-local intermediates and reusable MLX cache state before the next request.

## Results

| Variant | Wall | Setup/load overhead | Max RSS | Measured total median | Measured sample median | Measured decode median | Decode measured min/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous default cleanup off | 20.35 s | 5994.5 ms | 2.92 GiB | 2882.6 ms | 940.9 ms | 1869.7 ms | 1338.7 / 2300.8 ms |
| runtime decode cleanup | 13.78 s | 5406.1 ms | 2.92 GiB | 1670.2 ms | 834.1 ms | 801.9 ms | 782.8 / 854.0 ms |

The previous-default row is the #213 local reproduction captured before the runtime fix. The runtime-cleanup source report was removed after this comparison captured the decision-relevant medians, ranges, and recommendation.

## Interpretation

The repeated-request slowdown was runtime-local to MLX DACVAE decode residency. The RF sampling band stayed stable, while decode latency stopped drifting once decode-local MLX work was materialized and released inside `MLXDACVAEBridge.decode_to_wav`.

Unlike the broad benchmark-level `--cleanup-between-requests` experiment, this fix scopes the cleanup to the MLX DACVAE decode boundary. In this run it preserved the lower `mlx-decode` max RSS profile at 2.92 GiB while restoring stable repeated-request latency.

## Recommendation

Use the default `mlx-decode` persistent path for repeated no-reference v3 generation after this fix. Keep `--cleanup-between-requests` as a diagnostic harness switch for testing future request-local residency issues outside codec decode.
