# Persistent local serving benchmark contract

Issue: [#218 Add a persistent local serving benchmark for request latency](https://github.com/t0yohei/Irodori-TTS-MLX/issues/218)

## Summary

This report records the benchmark contract added for the sub-second local-generation track. The new harness is `scripts/benchmark_persistent_serving.py`.

It starts one long-lived JSON-line worker, waits until the RF-DiT runtime, tokenizers, hosted weights, and DACVAE bridge are initialized, then sends repeated requests over stdin/stdout. The measured headline field is persistent request latency: the parent-side roundtrip from request write to response read. That excludes fresh CLI/process setup for each request.

## Report schema

The JSON summary uses `benchmark_kind: persistent-local-serving` and separates:

- `worker.startup_ms`
- `worker.max_rss_bytes`
- `aggregates.measured_persistent_request_latency_ms`
- `aggregates.measured_sample_rf_ms`
- `aggregates.measured_decode_dacvae_ms`
- `aggregates.measured_decode_dacvae_model_ms`
- `aggregates.measured_audio_write_ms`
- `aggregates.measured_total_to_decode_ms`
- per-request `persistent_request_latency_ms`
- per-request `timings_ms.sample_rf`
- per-request `timings_ms.decode_dacvae`
- per-request `timings_ms.decode_dacvae_model`
- per-request `timings_ms.audio_write`
- per-request `timings_ms.total_to_decode`

`decode_dacvae` remains the full codec decode-to-file boundary. For in-process PyTorch and MLX codec paths, `decode_dacvae_model` and `audio_write` split the waveform decode work from WAV serialization/write time. For subprocess codec modes, the worker still reports the combined `decode_dacvae` boundary.

## Dependency-light validation

The self-test exercises the report and schema logic without model assets:

    python3 scripts/benchmark_persistent_serving.py --self-test

The dry-run path writes the request plan and worker command without loading private or hosted artifacts:

    python3 scripts/benchmark_persistent_serving.py --dry-run --weights-repo owner/repo --codec-runtime-mode mlx-decode --codec-artifact-repo owner/codec --text '今日はいい天気ですね。' --omit-seconds --num-steps 12 --warmup-requests 1 --requests 2 --output-dir benchmark-runs/persistent-serving-dry-run

## Apple Silicon v3 command

Use this command to gather real no-reference v3 `mlx-decode` evidence on an Apple Silicon machine with access to the hosted RF-DiT and DACVAE artifacts:

    PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark_persistent_serving.py --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-runtime-mode mlx-decode --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 --text '今日はいい天気ですね。' --omit-seconds --num-steps 12 --warmup-requests 1 --requests 4 --case-label v3-mlx-decode-persistent-serving --output-dir benchmark-runs/persistent-serving-v3-mlx-decode --report benchmark-runs/persistent-serving-v3-mlx-decode-report.md

## Current recommendation

For #220, use persistent request latency as the headline metric for local serving optimization. One-shot wall clock remains useful only as a setup/loading diagnostic because it includes process startup, hosted artifact resolution, tokenizer/model construction, and CLI setup overhead that a persistent request path does not pay per request.
