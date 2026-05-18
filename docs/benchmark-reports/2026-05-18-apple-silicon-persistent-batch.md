# Apple Silicon persistent batch generation benchmark

Issue: follow-up from codec runtime mode benchmark / PR #206.

## Summary

This measured the existing persistent batch path: one scripts/generate_wav.py process, one initialized MLXDACVAERuntime, and multiple generation requests supplied through --requests-json.

- RF-DiT weights repo: t0yohei/Irodori-TTS-MLX-500M-v3
- RF-DiT weights revision: 078ffb11ffad92e6dde237a6abef730f4341b359
- DACVAE codec artifact repo: t0yohei/Irodori-TTS-MLX-DACVAE-Codec
- DACVAE codec artifact revision: bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78
- benchmark commit: f22ecf8b1b8000ed7bc25c6b16536978657de228 plus local benchmark harness changes

The result is useful but not a clean win yet. Persistent batch amortizes process setup across requests and keeps max RSS around the mlx-decode one-shot level, but measured request latency drifted upward after the first request, mostly in decode_dacvae. This means a future persistent server/worker should not assume runtime reuse is automatically faster; the next optimization target is per-request MLX codec cleanup or decode residency behavior inside the persistent process.

## Environment

- machine: Apple Silicon arm64
- benchmark Python: Python 3.11.15
- codec runtime mode: mlx-decode
- upstream runtime import path: /path/to/Irodori-TTS
- benchmark harness: /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark_persistent_batch.py

## Benchmark configuration

- text: 今日はいい天気ですね。
- checkpoint family: v3
- duration mode: predicted duration (--seconds omitted)
- num_steps: 12
- seed start: 20260512
- warmup requests: 1
- measured requests: 4
- reference mode: no-reference

## Results

| Scope | Value |
| --- | ---: |
| process wall | 30.64 s |
| process setup/load overhead | 16248.1 ms |
| max RSS | 2.89 GiB |
| process throughput | 0.163 req/s |
| measured generation throughput | 0.327 req/s |
| measured total_to_decode median | 3256.1 ms |
| measured sample_rf median | 1093.8 ms |
| measured decode_dacvae median | 2005.4 ms |

Raw request timings:

| # | Phase | Seed | total_to_decode | sample_rf | decode_dacvae |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | warmup | 20260512 | 2141.8 ms | 993.9 ms | 1108.6 ms |
| 2 | measured | 20260513 | 3228.2 ms | 884.3 ms | 2310.3 ms |
| 3 | measured | 20260514 | 2307.3 ms | 994.4 ms | 1224.8 ms |
| 4 | measured | 20260515 | 3284.1 ms | 1253.3 ms | 1923.4 ms |
| 5 | measured | 20260516 | 3430.7 ms | 1193.2 ms | 2087.4 ms |

## Read against the one-shot codec mode benchmark

The previous warm one-shot v3 no-reference mlx-decode run recorded total_to_decode 1769.4 ms, wall 6.93 s, and max RSS 2.93 GiB. This persistent batch run recorded process wall 30.64 s for five requests, or about 6.13 s per request at the process level, so setup amortization helped only modestly in this short batch. The request-level measured median was slower than the one-shot warm request because decode_dacvae rose to about 2.0 s median.

Interpretation:

- startup/runtime reuse is still the right server/worker direction, but the current batch path exposes a decode-time regression across repeated requests
- max RSS stayed close to the one-shot mlx-decode run, so the issue is not obvious process-level memory explosion
- the next benchmark should add an explicit post-request MLX cleanup experiment around codec decode, then rerun the same persistent batch harness

## Command

    PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark_persistent_batch.py --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-runtime-mode mlx-decode --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 --text '今日はいい天気ですね。' --omit-seconds --num-steps 12 --warmup-requests 1 --requests 4 --case-label v3-mlx-decode-persistent-batch --output-dir benchmark-runs/persistent-batch-v3-mlx-decode --report benchmark-runs/persistent-batch-v3-mlx-decode-report.md

