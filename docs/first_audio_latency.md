# First-Audio Latency Versus Complete-WAV Latency

Issue: [#222](https://github.com/t0yohei/Irodori-TTS-MLX/issues/222)
Parent epic: [#220](https://github.com/t0yohei/Irodori-TTS-MLX/issues/220)

## Decision

For the current short Japanese no-reference v3 architecture, treat **complete-WAV request latency** as the sub-second target. Do not claim a separate sub-second first-audio target yet.

The current RF-DiT + DACVAE path does not expose useful audio before a complete latent sequence has been sampled and decoded. A local assistant can start playback as soon as the WAV exists, but that first playable byte is currently after RF sampling, full DACVAE decode, and audio serialization.

## Current Architecture

The current generation boundary is:

1. Prepare text and optional speaker/caption conditions.
2. Resolve output duration into a full latent sequence length.
3. Run RF-DiT sampling for the full patched latent sequence.
4. Unpatch the full latent sequence.
5. Decode the full latent sequence through DACVAE.
6. Serialize the decoded audio to a WAV path.

The key implementation point is that `MLXDACVAERuntime.generate()` calls `sample_euler_rf_cfg()` once for the full sequence, evaluates the full latent tensor, and then calls `bridge.decode_to_wav(...)`. The public timing field `decode_dacvae` currently measures the decode-to-WAV boundary as one inclusive stage, so it includes DACVAE decode plus WAV serialization/write time.

## Evidence Summary

Existing Apple Silicon reports already separate one-shot wall clock, persistent request latency, RF sampling, and the inclusive decode-to-WAV stage:

- One-shot v3 no-reference 12-step report: `sample_rf` 1010.7 ms, `decode_dacvae` 1194.5 ms, `total_to_decode` 2243.4 ms.
- One-shot v3 no-reference 8-step report: `sample_rf` 678.6 ms, `decode_dacvae` 1099.3 ms, `total_to_decode` 1818.5 ms.
- Persistent v3 no-reference `mlx-decode` report after runtime cleanup: measured `sample_rf` median 834.1 ms, `decode_dacvae` median 801.9 ms, `total_to_decode` median 1670.2 ms, process wall 25.56 s for five requests.

These numbers mean first playable audio is not currently earlier than complete-WAV availability for a single short prompt. Even if RF sampling drops below one second, the current full-latent DACVAE boundary still puts first audio behind decode and serialization.

## Timing Contract

Latency evidence for this epic should use these fields:

- `one_shot_wall_clock_ms`: process-level wall time from `/usr/bin/time -l` for a single CLI generation.
- `persistent_request_latency_ms`: per-request `total_to_decode` from `scripts/generate_wav.py --requests-json` metadata, excluding process setup.
- `rf_sampling_ms`: request `timings_ms.sample_rf`.
- `dacvae_decode_ms`: request DACVAE decode time when a future harness separates it from serialization.
- `audio_serialization_write_ms`: request WAV serialization/write time when separately measured.
- `decode_to_wav_ms`: current inclusive `timings_ms.decode_dacvae` field, used until `dacvae_decode_ms` and `audio_serialization_write_ms` are separately instrumented.
- `first_audio_available_ms`: earliest point at which local playback can start. For the current architecture this equals complete-WAV availability, so use `persistent_request_latency_ms` for persistent runs or `one_shot_wall_clock_ms` for one-shot user-visible CLI runs.
- `complete_wav_available_ms`: the time at which the output WAV exists and can be opened by playback code.

Use [first_audio_latency_report_schema.json](first_audio_latency_report_schema.json) for machine-readable reports. If a run cannot split DACVAE decode from WAV serialization, set `dacvae_decode_ms` and `audio_serialization_write_ms` to `null`, keep `decode_to_wav_ms`, and set `notes` to explain that `decode_to_wav_ms` is inclusive.

## Reproducible Evidence Command

Real Apple Silicon/model artifacts are required for runtime evidence. The focused command is:

```bash
python3 scripts/benchmark_persistent_batch.py \
  --case-label issue-222-first-audio-v3-no-reference \
  --text '今日はいい天気ですね。' \
  --requests 4 \
  --warmup-requests 1 \
  --omit-seconds \
  --num-steps 8 \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
  --codec-runtime-mode mlx-decode \
  --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
  --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
  --codec-device cpu \
  --report docs/benchmark-reports/issue-222-first-audio-v3-no-reference.md
```

Expected evidence fields:

- process summary: `wall_seconds`, `process_setup_overhead_ms`, `process_throughput_rps`
- request rows: `phase`, `total_to_decode`, `sample_rf`, `decode_dacvae`, `output`
- interpretation: `first_audio_available_ms == complete_wav_available_ms` for each measured request because the current path writes a complete WAV before playback can begin

## Feasibility

True first-audio streaming is **not feasible at low risk** in the current single-utterance path:

- RF-DiT sampling is full-sequence today; it does not emit stable latent chunks that can be decoded independently.
- The current DACVAE bridge API is `decode_to_wav(latents, output_path)`, which takes the full latent tensor and writes a complete WAV.
- The current artifact validation proves full decode parity, not chunk-boundary parity.
- Chunked latent decode would need overlap/crossfade validation and perceptual checks to avoid boundary artifacts.

The lowest-risk local assistant path is therefore a persistent request runtime with immediate playback of the completed WAV. For longer text, sentence-level segmentation could reduce perceived wait by generating and playing the first sentence while later sentences render, but that is a segmentation product feature rather than evidence that the current short-prompt v3 path has earlier first audio.

## Recommendation

For #220, define the near-term sub-second target as **complete WAV available for playback after a warmed persistent request**. Keep first-audio latency as a future UX metric only after the runtime can produce validated segment or chunk artifacts with explicit `first_audio_available_ms` evidence.
