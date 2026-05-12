# irodori-tts-mlx

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

## Checkpoint inspection

Use `scripts/inspect_checkpoint.py` to inspect local or Hugging Face `model.safetensors` checkpoints without loading tensor payloads:

```bash
python3 scripts/inspect_checkpoint.py Aratako/Irodori-TTS-500M-v2
python3 scripts/inspect_checkpoint.py Aratako/Irodori-TTS-500M-v2 --json > checkpoint.json
python3 scripts/inspect_checkpoint.py /path/to/model.safetensors --all-tensors
```

The script prints metadata/config, tensor names, shapes, dtypes, and parameter totals for weight-converter planning.

## Weight conversion

Use `scripts/convert_weights.py` to convert a local base v2 checkpoint into an MLX-friendly `.npz` archive:

```bash
python3 scripts/convert_weights.py /path/to/model.safetensors /path/to/irodori-tts-500m-v2.npz
python3 scripts/convert_weights.py /path/to/model.safetensors --dry-run
python3 scripts/convert_weights.py /path/to/model.safetensors --dry-run --json
```

The initial converter supports the base `Aratako/Irodori-TTS-500M-v2` layout only. It validates the documented key mapping, shape expectations, float32 dtypes, and base speaker-conditioning config before writing output. The VoiceDesign/caption checkpoint is rejected until caption conversion support is implemented.

The initial converter accepts only local `.safetensors` checkpoints. Converting them requires the optional `safetensors` Python package. Header-only `--dry-run` validation works without loading the multi-GiB tensor payload.

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

This is still a layer-level/model-component API, not a stable public generation interface. End-to-end RF-DiT forward, sampling, tokenization, and the PyTorch DACVAE bridge remain later milestones.

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
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
- DACVAE: <https://github.com/facebookresearch/dacvae>

## Roadmap

The current project milestones are organized as follows:

1. **M0 Baseline**: define scope and reproduce upstream PyTorch inference on Apple Silicon.
2. **M1 Weight conversion**: inspect checkpoints and implement PyTorch/safetensors to MLX weight conversion.
3. **M2 MLX model parity**: port model components and compare against PyTorch outputs.
4. **M3 MLX inference prototype**: generate audio with MLX RF-DiT and a PyTorch DACVAE bridge.
5. **M4 Performance and packaging**: benchmark, optimize, document, and package the prototype.

## License notes

The project license has not been finalized yet. Upstream code and model weights may have different license terms. Check the upstream repository and model cards before reusing or redistributing any derived artifacts.
