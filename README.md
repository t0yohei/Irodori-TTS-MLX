# irodori-tts-mlx

[日本語 README](README.ja.md)

An unofficial MLX inference port of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) for Apple Silicon.

> [!IMPORTANT]
> This project is in the planning and early prototype stage. It does not provide usable inference code yet.

## Project goal

The first practical target is a v0 inference prototype with this boundary:

> MLX RF-DiT inference + PyTorch DACVAE encode/decode bridge

In other words, the initial implementation should port the Irodori-TTS text/condition encoders, RF-DiT model, and rectified-flow sampler to MLX, while continuing to use the upstream PyTorch DACVAE path for reference audio encoding and waveform decoding.

This keeps the first milestone focused on the part most likely to benefit from MLX, without taking on a full DACVAE port before the core model path is validated.

## Intended v0 architecture

```text
text prompt ───────────────┐
reference audio ── PyTorch DACVAE encode ──┐
caption/style text ────────┐               │
                            ▼               ▼
                     MLX encoders + RF-DiT sampler
                                      │
                                      ▼
                          generated DACVAE latents
                                      │
                                      ▼
                         PyTorch DACVAE decode
                                      │
                                      ▼
                                  output WAV
```

For more detail, see [docs/architecture.md](docs/architecture.md).

For the upstream PyTorch baseline workflow used to compare future MLX work, see [docs/baseline.md](docs/baseline.md).

For checkpoint metadata and state_dict layout notes that will guide weight conversion, see [docs/weight_mapping.md](docs/weight_mapping.md).

For optional upstream PyTorch vs MLX deterministic component parity tests, see [docs/pytorch_parity.md](docs/pytorch_parity.md).

For the initial MLX rectified-flow Euler sampler and CFG behavior, see [docs/rf_sampler.md](docs/rf_sampler.md).

For Apple Silicon benchmark workflow, current baseline conclusions, and the benchmark harness for upstream vs MLX bridge comparison, see [docs/benchmark.md](docs/benchmark.md).

For the first end-to-end MLX RF-DiT + PyTorch DACVAE bridge and WAV-generation CLI, see [docs/dacvae_bridge.md](docs/dacvae_bridge.md).

For the current `Aratako/Irodori-TTS-500M-v3` support statement, manual validation recipe, and hosted Apple Silicon coverage, see [docs/v3_support.md](docs/v3_support.md).

For the packaged install story, supported Python versions, and reproducible runtime / benchmark environment setup, see [docs/packaging.md](docs/packaging.md).

## Supported Python and install targets

The current packaged environment supports **Python 3.11 through 3.14**.
Python 3.11 remains the reference environment for the published benchmark notes and examples in this repository.

Install this repo in editable mode depending on your use case:

```bash
python3.11 -m venv .venv  # or: python3.12/3.13/3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[runtime]"  # WAV generation / bridge runtime
python -m pip install -e ".[bench]"    # benchmark + conversion workflow
python -m pip install -e ".[dev]"      # local contributor environment
```

The bridge runtime still depends on upstream `irodori_tts` for `DACVAECodec`, so either install the upstream checkout into the same venv or expose it on `PYTHONPATH`. The full setup guide lives in [docs/packaging.md](docs/packaging.md).

## README split

To keep both READMEs discoverable without letting them drift too far:

- [README.md](README.md) is the canonical source for exact technical scope, compatibility, and milestone status.
- [README.ja.md](README.ja.md) provides a Japanese overview, setup entry points, current limitations, and links into the detailed docs.
- Detailed procedures and validation notes should continue to live under `docs/*.md` so both READMEs can stay concise.

## Checkpoint inspection

Use the installed `irodori-tts-inspect` command to inspect local or Hugging Face `model.safetensors` checkpoints without loading tensor payloads:

```bash
irodori-tts-inspect Aratako/Irodori-TTS-500M-v2
irodori-tts-inspect Aratako/Irodori-TTS-500M-v2 --json > checkpoint.json
irodori-tts-inspect /path/to/model.safetensors --all-tensors
```

The legacy `python3 scripts/inspect_checkpoint.py ...` path remains supported for repository checkouts.

The script prints metadata/config, tensor names, shapes, dtypes, and parameter totals for weight-converter planning.

## Weight conversion

Use the installed `irodori-tts-convert` command to convert a local base v2, v3, or VoiceDesign checkpoint into an MLX-friendly `.npz` archive:

```bash
irodori-tts-convert /path/to/model.safetensors /path/to/irodori-tts-500m-v2.npz
irodori-tts-convert /path/to/model.safetensors --dry-run
irodori-tts-convert /path/to/model.safetensors --dry-run --json
```

The legacy `python3 scripts/convert_weights.py ...` path remains supported for repository checkouts.

The converter now supports the base `Aratako/Irodori-TTS-500M-v2` layout, the `Aratako/Irodori-TTS-500M-v2-VoiceDesign` caption-conditioned layout, and the `Aratako/Irodori-TTS-500M-v3` duration-predictor layout. It validates the documented key mapping, shape expectations, float32 dtypes, and family-specific config assumptions before writing output. Use `--dry-run --json` to confirm the detected `checkpoint_family` before exporting large checkpoints. V3 is now supported through conversion plus the MLX bridge runtime, with duration semantics documented in [docs/dacvae_bridge.md](docs/dacvae_bridge.md) and reproducible validation coverage documented in [docs/v3_support.md](docs/v3_support.md). See [docs/caption_condition_support.md](docs/caption_condition_support.md) for the separate VoiceDesign support matrix.

The initial converter accepts only local `.safetensors` checkpoints. Converting them requires the optional `safetensors` Python package. Header-only `--dry-run` validation works without loading the multi-GiB tensor payload.

For standing integration coverage against the real public VoiceDesign checkpoint, use `scripts/run_voicedesign_integration.py` or the scheduled/manual GitHub Actions workflow in `.github/workflows/voicedesign-real-checkpoint.yml`. That lightweight automation validates inspect + converter family detection without forcing full `.npz` export on every run.

For full end-to-end hosted coverage of `scripts/generate_wav.py --caption ...`, use `scripts/run_voicedesign_generation_ci.py` or `.github/workflows/voicedesign-hosted-generation.yml`. For equivalent v3 coverage on the predicted-duration path, use `scripts/run_v3_generation_ci.py` or `.github/workflows/v3-hosted-generation.yml`. These workflows now target the standard GitHub-hosted Apple Silicon M1 runner (`macos-14`), so public-repository runs stay on the free hosted macOS tier without needing self-hosted infrastructure.

## Benchmarking

Use `scripts/benchmark.py` to orchestrate reproducible upstream PyTorch and MLX bridge timing runs, collect `/usr/bin/time -l` memory observations, repeat runs with warm/cold labeling, and emit Markdown + JSON summaries. Python 3.11 through 3.14 are supported for packaging, while Python 3.11 remains the recommended benchmark reference environment described in [docs/packaging.md](docs/packaging.md).


```bash
python3 scripts/benchmark.py --self-test
python3 scripts/benchmark.py --mode upstream --upstream-root /path/to/Irodori-TTS
python3 scripts/benchmark.py --mode mlx --weights /path/to/irodori-tts-500m-v2.npz --upstream-root /path/to/Irodori-TTS
python3 scripts/benchmark.py --mode mlx --weights /path/to/irodori-tts-500m-v2.npz --upstream-root /path/to/Irodori-TTS --repeat 3 --warmup-runs 1 --reference-wav /path/to/reference.wav
python3 scripts/benchmark.py --mode mlx --weights /path/to/irodori-tts-500m-v2.npz --upstream-root /path/to/Irodori-TTS --seconds-sweep 3,5,8 --num-steps-sweep 20,40
```

The MLX bridge runtime emits benchmark-friendly `[timing]` lines for text/reference preparation, RF sampling, DACVAE decode, and total inference time.
For reference-path memory experiments, `--codec-runtime-mode persistent|subprocess` can compare the normal in-process bridge against a helper-process DACVAE boundary.
The benchmark summary JSON now records per-run metadata (`phase`, `cache_state`, sweep parameters) plus aggregated min/median/max statistics so future reports can diff repeated runs instead of relying on one-off measurements.

## Core MLX layers

The `irodori_mlx.layers` module contains the first reusable MLX primitives for model parity work:

- `RMSNorm`
- RoPE frequency generation and application helpers
- sinusoidal timestep embeddings
- `SwiGLU`
- low-rank AdaLN modulation
- latent sequence patch/unpatch helpers

These implementations follow the upstream PyTorch formulas used by Irodori-TTS and keep normalization/embedding math in `float32` where practical. Floating-point inputs are cast back to their original dtype after operations such as RMSNorm and RoPE application, so future bf16 inference paths can keep bf16 activations while still using fp32 statistics for numerically sensitive steps.

## Condition encoders

The `irodori_mlx.encoders` module contains the first MLX conditioning stack:

- token `TextEncoder` for prompt text
- `ReferenceLatentEncoder` for base-checkpoint speaker/reference latent conditioning
- optional caption encoder wiring for VoiceDesign-style checkpoints
- `ConditionEncoders` wrapper for text, speaker, and caption masks/dropout
- narrow `.npz` weight assignment helpers for converted upstream encoder weights

Masked positions are hard-zeroed after embedding and after each residual block so fully masked conditioning becomes an unconditional path. Speaker/reference conditioning also patches the latent sequence when configured and prepends the upstream-style masked-mean summary token.

The first `irodori_mlx.model.TextToLatentRFDiT` forward path is now available for MLX model-parity work. It wires the condition encoders into joint RF-DiT attention, timestep-conditioned AdaLN blocks, static conditioning K/V projection caches, and final latent velocity projection. See [docs/rf_dit_forward.md](docs/rf_dit_forward.md) for implementation and numerical-comparison notes.

`irodori_mlx.sampling.sample_euler_rf_cfg` adds the first RF Euler sampling loop on top of the MLX model path. It can generate patched latent sequences with fixed-seed noise, upstream-style timesteps, optional context K/V cache, and text/speaker/caption CFG modes.

`irodori-tts-generate` (also available as the legacy `python scripts/generate_wav.py` path) and `irodori_mlx.runtime.MLXDACVAERuntime` provide the first prototype WAV-generation path: tokenize text, encode reference audio with upstream/PyTorch DACVAE, sample generated latents with MLX RF-DiT, decode them back to waveform with PyTorch DACVAE, and save a WAV. The CLI now supports repeatable `--config-json` presets, user-facing `--preset fast|balanced|quality` step-count shortcuts, `--requests-json` persistent batch mode for repeated local generations that reuse one initialized runtime, plus `--json` / `--metadata-json` output for automation-friendly metadata and timings. Caption-conditioned checkpoints can now use the documented conversion + runtime path as long as their metadata and tensor layout match the inspected VoiceDesign family, and the public `Aratako/Irodori-TTS-500M-v3` path is supported with predicted-duration semantics when `--seconds` is omitted. See [docs/dacvae_bridge.md](docs/dacvae_bridge.md) for dependencies, invocation patterns, preset mappings, persistent batch examples, and boundary notes, [docs/caption_condition_support.md](docs/caption_condition_support.md) for the VoiceDesign support statement, and [docs/v3_support.md](docs/v3_support.md) for the v3 validation story.

## Public API direction

The first user-facing interface should be CLI-first, with a small Python API underneath it.

Planned shape:

- CLI: simple generation commands for local experimentation
- Python API: reusable loading and generation functions used by the CLI
- No stable API guarantee until the first end-to-end inference prototype works

## Non-goals for v0

The initial prototype should not include:

- training or fine-tuning support
- a full MLX DACVAE port
- Gradio or web UI support
- model distribution or checkpoint redistribution
- broad compatibility with every historical Irodori-TTS checkpoint

## Related resources

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2 model card: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign model card: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3 model card: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
- DACVAE: <https://github.com/facebookresearch/dacvae>

## Roadmap

The current project milestones are organized as follows:

1. **M0 Baseline**: define scope and reproduce upstream PyTorch inference on Apple Silicon.
2. **M1 Weight conversion**: inspect checkpoints and implement PyTorch/safetensors to MLX weight conversion.
3. **M2 MLX model parity**: port model components and compare against PyTorch outputs.
4. **M3 MLX inference prototype**: generate audio with MLX RF-DiT and a PyTorch DACVAE bridge.
5. **M4 Performance and packaging**: benchmark, optimize, document, and package the prototype.

Current status by milestone:

- **M0 Baseline**: completed.
- **M1 Weight conversion**: completed, including VoiceDesign / caption-conditioned checkpoint conversion support from [#41 Add VoiceDesign / caption-conditioned checkpoint conversion support](https://github.com/t0yohei/irodori-tts-mlx/issues/41).
- **M2 MLX model parity**: completed for the currently supported checkpoint families, with VoiceDesign follow-up work captured and closed in [#33 Audit and expand VoiceDesign / caption-conditioned checkpoint support](https://github.com/t0yohei/irodori-tts-mlx/issues/33).
- **M3 MLX inference prototype**: completed for the current CLI/runtime scope, including generation UX follow-up from [#32 Improve the generation CLI and runtime UX](https://github.com/t0yohei/irodori-tts-mlx/issues/32), real-checkpoint integration coverage from [#44 Add real-checkpoint VoiceDesign integration coverage](https://github.com/t0yohei/irodori-tts-mlx/issues/44), and hosted Apple Silicon full-generation coverage from [#46 Add hosted Apple Silicon CI coverage for full VoiceDesign generation](https://github.com/t0yohei/irodori-tts-mlx/issues/46).
- **M4 Performance and packaging**: completed for the current prototype scope, including memory residency mitigation from [#29 Investigate and reduce reference-path memory residency in the MLX bridge](https://github.com/t0yohei/irodori-tts-mlx/issues/29), repeated benchmark automation from [#30 Extend benchmark automation for warm-cache, repeated runs, and scaling sweeps](https://github.com/t0yohei/irodori-tts-mlx/issues/30), and reproducible packaging from [#31 Package the project for reproducible runtime and benchmark environments](https://github.com/t0yohei/irodori-tts-mlx/issues/31).

## License notes

The project license has not been finalized yet. Upstream code and model weights may have different license terms. Check the upstream repository and model cards before reusing or redistributing any derived artifacts.
