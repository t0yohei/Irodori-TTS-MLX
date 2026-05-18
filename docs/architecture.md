# v0 Architecture

This document defines the initial architecture target for `irodori-tts-mlx`.

## Goal

The v0 target is:

> MLX RF-DiT inference + PyTorch DACVAE encode/decode bridge

The goal is to validate whether porting the core Irodori-TTS latent generation path to MLX improves practical Apple Silicon inference before attempting a full codec port.

## Scope

v0 should include:

- MLX implementations of the Irodori-TTS inference-time model components.
- A weight conversion path from upstream PyTorch/safetensors checkpoints into MLX-loadable weights.
- Rectified-flow Euler sampling in MLX.
- A bridge that uses the upstream PyTorch DACVAE path for:
  - reference audio encoding
  - generated latent decoding
- A CLI-first generation flow backed by internal reusable Python modules.
- Baseline and benchmark documentation against upstream PyTorch/MPS behavior.

## Non-goals

v0 should not include:

- training support
- LoRA fine-tuning support
- a full MLX DACVAE implementation
- a Gradio app or hosted demo
- checkpoint redistribution
- broad support for older incompatible Irodori-TTS versions
- Swift, Core ML, or app integration

These may be revisited after the first end-to-end prototype is working and benchmarked.

## High-level data flow

```text
text / caption input ──► MLX text/caption path ───────┐
                                                       │
reference WAV ────────► PyTorch DACVAE encode ────────┤
                                                       ▼
                                             MLX RF-DiT sampler
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

## Component boundary

| Component | v0 owner | Notes |
| --- | --- | --- |
| Tokenization and text normalization | TBD / likely Python compatibility layer first | Keep behavior close to upstream before optimizing. |
| Text encoder | MLX | Needed for core model parity. |
| Reference latent encoder | MLX | Consumes DACVAE latents produced by the PyTorch bridge. |
| Caption encoder / VoiceDesign path | MLX, if included in the first checkpoint target | Can be staged after base-model parity if necessary. |
| RF-DiT blocks | MLX | Main porting target. |
| Euler RF sampler and CFG | MLX | Should match upstream sampling parameters where practical. |
| DACVAE reference encoding | PyTorch bridge | Explicitly not ported in v0. |
| DACVAE waveform decoding | PyTorch bridge | Explicitly not ported in v0. |
| CLI | Project code | First user-facing interface. |
| Python modules | Project code | Internal reusable layer for CLI; not a stable public API during alpha. |

## API direction

The v0 interface is CLI-first while keeping the implementation reusable inside
the repository. The public support boundary is documented in
[public_api_stability.md](public_api_stability.md): downstream users should rely
on installed console scripts and documented artifact layouts, not Python module imports.

CLI shape:

```bash
irodori-tts-generate \
  --weights path/to/weights.npz \
  --text "今日はいい天気ですね。" \
  --reference-wav reference.wav \
  --output output.wav
```

No stable Python API should be promised until a separate design issue names the
supported modules, classes, functions, data contracts, versioning policy, and
deprecation expectations.

## Implementation sequence

1. Reproduce upstream PyTorch inference on Apple Silicon.
2. Inspect checkpoint metadata and tensor layout. See [weight_mapping.md](weight_mapping.md) for the current checkpoint overview.
3. Implement checkpoint inspection and conversion tooling.
4. Port reusable MLX layers and validate layer-level parity.
5. Port encoders and RF-DiT forward pass.
6. Add PyTorch vs MLX parity tests.
7. Implement MLX Euler sampling and CFG.
8. Add the PyTorch DACVAE bridge.
9. Generate the first end-to-end WAV.
10. Benchmark against the upstream PyTorch/MPS baseline.

## Assumptions

- The first target checkpoint is `Aratako/Irodori-TTS-500M-v2`.
- VoiceDesign support is desirable, but base-model inference parity is the first priority if the two paths need to be staged.
- The upstream PyTorch implementation remains the reference for correctness.
- The project should avoid vendoring upstream code unless a later issue explicitly decides otherwise.

## Risks and open design notes

- **Repository status**: the project starts from an empty repository, so the first PR establishes the documentation baseline.
- **Numerical parity**: MLX and PyTorch may differ slightly in attention, dtype behavior, or random sampling. Parity tolerances need to be documented.
- **DACVAE boundary cost**: moving tensors between MLX and PyTorch may reduce speed gains. This should be measured before deciding whether to port DACVAE.
- **Licensing**: upstream code and model weights may use different terms. This project should link to the upstream sources and model cards rather than making broad redistribution claims.
- **Scope creep**: training, UI, and full codec porting should stay out of v0 until the core inference path is proven.
