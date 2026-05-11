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

## Checkpoint inspection

Use `scripts/inspect_checkpoint.py` to inspect local or Hugging Face `model.safetensors` checkpoints without loading tensor payloads:

```bash
python3 scripts/inspect_checkpoint.py Aratako/Irodori-TTS-500M-v2
python3 scripts/inspect_checkpoint.py Aratako/Irodori-TTS-500M-v2 --json > checkpoint.json
python3 scripts/inspect_checkpoint.py /path/to/model.safetensors --all-tensors
```

The script prints metadata/config, tensor names, shapes, dtypes, and parameter totals for weight-converter planning.

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
