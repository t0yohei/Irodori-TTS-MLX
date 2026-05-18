# Upstream dependency boundary

Issue: [#233 Prepare a standalone MLX runtime install path without upstream DACVAE bridge dependencies](https://github.com/t0yohei/Irodori-TTS-MLX/issues/233)

## Public runtime boundary

The v0.3 alpha public generation runtime is standalone by default. It does not require upstream `irodori_tts`, `irodori_tts.codec.DACVAECodec`, `torch`, or `torchaudio` for WAV generation.

The public codec path is:

- RF-DiT weights from `--weights-repo`, `--weights-dir`, or `--weights`;
- DACVAE codec tensors from the approved default hosted artifact, `--codec-artifact-dir`, or `--codec-path`;
- `--codec-runtime-mode mlx`.

The old bridge-backed generation modes, including `persistent`, `subprocess`, and `mlx-decode`, are removed from the public runtime. Missing hosted artifacts should be handled by local conversion or a local codec artifact, not by falling back to upstream.

In this split, this MLX repo owns the text/caption conditioning, RF-DiT forward pass, duration handling, sampler path, and artifact-backed DACVAE runtime.

## Responsibility split

| Area | Owner in v0.3 alpha public runtime | Notes |
| --- | --- | --- |
| Text tokenizer loading and prompt token preparation | this MLX repo | Uses Transformers tokenizers configured by `ModelConfig`; tokenizer assets are not vendored. |
| Caption tokenizer and VoiceDesign-style conditioning | this MLX repo | Supported for the inspected public VoiceDesign family only. |
| RF-DiT forward pass and MLX layers | this MLX repo | Runs converted `.npz` weights through MLX modules. |
| Duration predictor / explicit `--seconds` semantics | this MLX repo | v3 predicted-duration behavior is implemented here. |
| Reference WAV loading and conditioning handoff | this MLX repo | Full-MLX mode uses executable Semantic-DACVAE encoder tensors from the codec artifact. |
| DACVAE encode/decode | this MLX repo | Full-MLX mode uses hosted/local codec artifacts. |
| WAV writing | this MLX repo | Writes decoded audio with `soundfile` or the stdlib fallback. |

## What this does not claim

This repository provides standalone v0.3 alpha WAV generation without upstream `irodori_tts` when using approved hosted/local MLX codec artifacts. It still does not vendor upstream code, checkpoints, tokenizer assets, generated audio, or codec weights.
