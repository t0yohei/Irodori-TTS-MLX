# v0.2 delivery plan

Linear rollup: [TOY-5](https://linear.app/toyontech/issue/TOY-5/irodori-tts-mlx-v02-cross-repo-delivery)

This document is the implementation plan for delivering Irodori-TTS-MLX v0.2 across the model/runtime repository and downstream local-assistant/OpenClaw consumers. GitHub issues and pull requests remain the implementation source of truth; the Linear issue is the project-level status rollup.

## Goal

Deliver Irodori-TTS-MLX as a usable local TTS path with:

- clear checkpoint-family UX for base v2, VoiceDesign v2, and v3;
- a practical VoiceDesign/v3 generation path that avoids known duration and artifact traps;
- MLX-native DACVAE encode/decode work planned behind parity evidence and a PyTorch bridge fallback;
- reproducible upstream-vs-MLX parity reports for supported families;
- a documented downstream smoke path for local-assistant/OpenClaw integration.

## Completion criteria

v0.2 is ready when all of the following are true:

- Hosted or local artifacts for supported checkpoint families load with user-facing family/capability messages.
- VoiceDesign v2 no-reference generation can choose a fallback duration without truncating ordinary prompts or over-extending tails.
- DACVAE decode and encode have either MLX parity evidence or an explicit bridge-only fallback statement for each generation path.
- Upstream-vs-MLX parity harness reports cover at least VoiceDesign and v3 scenarios, including partial reports when optional dependencies are missing.
- Local-assistant/OpenClaw smoke docs show the exact command path, expected metadata, and fallback behavior.
- Release/runbook docs state the remaining bridge boundary, artifact provenance rules, unsupported families, and known limitations.

## Workstreams

### 1. Runtime UX and duration handling

Primary issues: [#105](https://github.com/t0yohei/Irodori-TTS-MLX/issues/105), [#107](https://github.com/t0yohei/Irodori-TTS-MLX/issues/107)

Deliver this first because it improves the current usable path and gives downstream integrations stable metadata to assert against.

Required work:

- Add a text-length-based fallback duration heuristic for checkpoints without `use_duration_predictor`.
- Preserve explicit `--seconds` as the highest-priority manual override.
- Report checkpoint family, supported capabilities, duration mode, and resolved seconds in CLI/JSON metadata.
- Fail unsupported caption/reference/no-reference combinations with direct error messages.
- Document examples for base v2, VoiceDesign v2, and v3.

Validation:

- Unit tests cover short, smoke-length, and longer VoiceDesign fallback estimates.
- CLI/runtime tests prove explicit `--seconds` still wins.
- Documentation includes when users should override `--seconds` to avoid tail artifacts.

### 2. DACVAE MLX port and artifact policy

Primary issues: [#106](https://github.com/t0yohei/Irodori-TTS-MLX/issues/106), [#110](https://github.com/t0yohei/Irodori-TTS-MLX/issues/110), [#111](https://github.com/t0yohei/Irodori-TTS-MLX/issues/111), [#112](https://github.com/t0yohei/Irodori-TTS-MLX/issues/112), [#113](https://github.com/t0yohei/Irodori-TTS-MLX/issues/113), [#114](https://github.com/t0yohei/Irodori-TTS-MLX/issues/114), [#115](https://github.com/t0yohei/Irodori-TTS-MLX/issues/115), [#116](https://github.com/t0yohei/Irodori-TTS-MLX/issues/116)

Research handoff: [dacvae_architecture.md](dacvae_architecture.md) and
[dacvae_codec_contract.json](dacvae_codec_contract.json) pin the upstream codec
constants, logical tensor groups, shared family assumptions, and open blockers
that #112-#115 must consume.

Sequence decode before encode. Generation can use MLX decode without reference-audio encode, while reference-conditioned paths need encode later.

Required work:

- Research upstream `irodori_tts.codec.DACVAECodec`, Semantic-DACVAE architecture, tensor names, shapes, sampling rate, hop length, and preprocessing.
- Evaluate mlx-audio artifacts/runtime as a reference for codec layout and possible interoperability.
- Define codec artifact layout, manifest fields, provenance requirements, and local/hosted resolution behavior.
- Implement MLX decode path with PyTorch bridge fallback.
- Add decode parity fixtures and report tolerances before making MLX decode a recommended path.
- Implement MLX encode path for reference audio only after decode is validated.
- Add encode parity fixtures and report tolerances.

Validation:

- Decode parity compares fixed latents through upstream PyTorch and MLX.
- Encode parity compares fixed audio through upstream PyTorch and MLX.
- Runtime metadata records `codec_decode_backend` and `codec_encode_backend`.
- Docs state which families and flows still require the PyTorch bridge.

### 3. Upstream-vs-MLX parity harness

Primary issues: [#108](https://github.com/t0yohei/Irodori-TTS-MLX/issues/108), [#109](https://github.com/t0yohei/Irodori-TTS-MLX/issues/109), [#117](https://github.com/t0yohei/Irodori-TTS-MLX/issues/117), [#118](https://github.com/t0yohei/Irodori-TTS-MLX/issues/118), [#119](https://github.com/t0yohei/Irodori-TTS-MLX/issues/119), [#120](https://github.com/t0yohei/Irodori-TTS-MLX/issues/120), [#121](https://github.com/t0yohei/Irodori-TTS-MLX/issues/121)

Build the report schema before adding scenario-specific checks, so VoiceDesign and v3 evidence share the same structure.

Required work:

- Define a compact JSON report schema for inputs, environment, checkpoint family, seed, duration mode, runtime metadata, output WAV properties, timings, and dependency availability.
- Add a runner skeleton that can run upstream-only, MLX-only, or both sides and mark partial results clearly.
- Add VoiceDesign caption/CFG scenarios with contrasting captions.
- Add v3 duration-predictor scenarios.
- Add lightweight audio metrics: duration, sample rate, peak/RMS, tail/silence indicators, and optional spectral distance.
- Add intermediate comparisons where practical: token IDs, masks, caption conditioning shapes, predicted duration, latent shape/statistics.
- Document current baseline results and known non-parity limits.

Validation:

- The harness can produce a report without heavyweight assets by recording missing optional dependencies as partial status.
- Real-checkpoint local or hosted runs attach reports under ignored runtime directories or committed text summaries only.
- Docs explain what the metrics can and cannot prove.

### 4. Downstream local-assistant/OpenClaw integration

Primary issue: [#146](https://github.com/t0yohei/Irodori-TTS-MLX/issues/146).

Start this only after the runtime metadata from Workstream 1 is stable enough for a smoke assertion.

Required work:

- Pick the downstream consumer repository and entry point for the local TTS smoke path.
- Document the environment variables, model/artifact location, command, expected metadata fields, and output WAV check.
- Define fallback behavior when hosted weights, upstream `irodori_tts`, DACVAE codec assets, or MLX runtime dependencies are missing.
- Link the downstream PR and smoke report back from TOY-5.

Validation:

- Smoke command runs from a clean local environment or produces an actionable missing-dependency message.
- OpenClaw/local-assistant docs identify the exact Irodori-TTS-MLX version or PR head used.
- No generated audio, checkpoint cache, codec weights, or secrets are committed.

Smoke procedure for the local-assistant/OpenClaw smoke path: [docs/downstream_openclaw_smoke.md](downstream_openclaw_smoke.md).

### 5. Release and runbook cleanup

Primary issue: tracked by TOY-5 after the preceding workstreams have current evidence.

Required work:

- Update README and focused docs to describe the v0.2 support boundary.
- Keep hosted/pre-converted artifact provenance and license language aligned with `docs/hosted_weights_layout.md`, `docs/hosted_weights_usage.md`, and `docs/license_and_distribution.md`.
- Add a release checklist that links runtime UX, DACVAE parity, parity harness reports, downstream smoke results, and known limitations.
- Close or defer child issues explicitly rather than letting the Linear rollup become the only state record.

Validation:

- Fresh-reader docs have one supported quick path and one fallback path.
- Release notes name bridge-only paths and unsupported families directly.
- GitHub issue/PR links from this plan all resolve to completed, deferred, or follow-up state.

## Dependency order

1. Implement runtime UX and duration handling (#105, #107).
2. Define parity report schema (#117), then add VoiceDesign/v3 scenarios (#118, #119).
3. Evaluate mlx-audio interoperability (#110), research DACVAE layout (#111), and define codec artifact policy (#116).
4. Implement MLX DACVAE decode (#112) and decode parity (#113).
5. Add audio/intermediate metrics (#120) and publish current baseline docs (#121).
6. Implement MLX DACVAE encode (#114) and encode parity (#115).
7. Run downstream local-assistant/OpenClaw smoke once metadata and artifact behavior are stable.
8. Finish release/runbook cleanup and update TOY-5 with final issue/PR links.

## Current risks

- Full waveform parity between PyTorch and MLX may be unrealistic; the harness must emphasize diagnosable metrics and intermediate comparisons instead of bitwise equality.
- MLX DACVAE decode may require artifact format decisions before useful runtime code can land.
- Hosted converted weights cannot be treated as a shortcut around provenance and license review.
- Downstream smoke tests can become flaky if they assert on audio quality instead of metadata, output properties, and actionable failure modes.

## Tracking policy

- Every implementation PR should link the GitHub issue it closes or advances.
- TOY-5 should be updated with milestone status, not low-level implementation details.
- Heavy artifacts stay out of git; reports can commit compact Markdown/JSON summaries only when they contain no generated audio, checkpoint payloads, cache paths with secrets, or private machine details.
