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
- A CLI-first generation flow backed by a small Python API.
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
| Python API | Project code | Internal reusable layer for CLI; not stable before prototype validation. |

## API direction

The v0 interface should be CLI-first while keeping the implementation reusable from Python.

Planned CLI shape, subject to change:

```bash
irodori-tts-mlx generate \
  --checkpoint path/or/hf-repo \
  --text "今日はいい天気ですね。" \
  --ref-wav reference.wav \
  --output-wav output.wav
```

Planned Python shape, subject to change:

```python
from irodori_mlx import IrodoriGenerator

generator = IrodoriGenerator.from_pretrained("Aratako/Irodori-TTS-500M-v2")
generator.generate(
    text="今日はいい天気ですね。",
    ref_wav="reference.wav",
    output_wav="output.wav",
)
```

No stable API should be promised until the first end-to-end prototype works and the model boundary has been benchmarked.

## Implementation sequence

1. Reproduce upstream PyTorch inference on Apple Silicon.
2. Inspect checkpoint metadata and tensor layout.
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
