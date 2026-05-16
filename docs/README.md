# Documentation Index

This directory keeps the public technical documentation for Irodori-TTS-MLX.
The current user-facing entry points are:

- [README.md](../README.md): quick project overview, install, hosted weights, local conversion, and mlx-audio adapter examples.
- [hosted_weights_usage.md](hosted_weights_usage.md): shortest current generation path with `--weights-repo` / `--weights-dir` plus local conversion fallback.
- [dacvae_bridge.md](dacvae_bridge.md): generation CLI, codec runtime modes, JSON configuration, and persistent batch generation.
- [checkpoint_support.md](checkpoint_support.md): supported checkpoint families and unsupported/local-only boundaries.
- [packaging.md](packaging.md): reproducible install and console script setup.

## Runtime And Model Path

- [architecture.md](architecture.md): high-level runtime architecture.
- [weight_mapping.md](weight_mapping.md): checkpoint tensor layout and conversion mapping.
- [text_preprocessing.md](text_preprocessing.md): prompt and caption preprocessing contract.
- [caption_condition_support.md](caption_condition_support.md): VoiceDesign support statement.
- [v3_support.md](v3_support.md): v3 duration-predictor support and validation path.
- [rf_sampler.md](rf_sampler.md): RF Euler sampler and CFG behavior.

## Hosted Artifacts And Interop

- [hosted_weights_layout.md](hosted_weights_layout.md): hosted/pre-converted RF-DiT weights repository contract.
- [hosted_weights_usage.md](hosted_weights_usage.md): hosted weights usage and local fallback.
- [preconverted_weights_redistribution_audit.md](preconverted_weights_redistribution_audit.md): engineering due-diligence notes for reviewed checkpoint families.
- [license_and_distribution.md](license_and_distribution.md): repository license and non-redistribution boundary.
- [mlx_audio_interop.md](mlx_audio_interop.md): mlx-audio artifact interop and adapter boundary.

## DACVAE Work

- [upstream_dependency.md](upstream_dependency.md): upstream PyTorch DACVAE bridge dependency boundary.
- [dacvae_bridge.md](dacvae_bridge.md): runtime bridge details and codec mode choices.
- [codec_artifact_layout.md](codec_artifact_layout.md): local/hosted DACVAE codec artifact contract.
- [dacvae_architecture.md](dacvae_architecture.md): DACVAE architecture and checkpoint research notes.
- [dacvae_decode_parity.md](dacvae_decode_parity.md): decode-only parity evidence.
- [dacvae_encode_parity.md](dacvae_encode_parity.md): encode parity evidence.
- [mlx_audio_dacvae_contract.md](mlx_audio_dacvae_contract.md): mlx-audio DACVAE layout comparison and selected compatibility path.

## Validation And Benchmarks

- [benchmark.md](benchmark.md): benchmark workflow and current decision baseline.
- [baseline.md](baseline.md): upstream PyTorch baseline workflow.
- [upstream_parity_harness.md](upstream_parity_harness.md): upstream-vs-MLX parity harness and report schema.
- [v0_1_release_gate.md](v0_1_release_gate.md): historical release-gate workflow retained for reproducibility.
- [v0_2_delivery_plan.md](v0_2_delivery_plan.md): public delivery plan and downstream consumer handoff boundary.

Raw measured reports live under [benchmark-reports/](benchmark-reports/) and
[baseline-reports/](baseline-reports/). Keep only reports that are linked from
summary docs or tests; one-off reports with no current reference should be
removed instead of left as stale navigation clutter.
