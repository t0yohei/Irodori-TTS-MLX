# PyTorch DACVAE bridge

Issue #12 adds the first end-to-end prototype boundary:

```text
text prompt + reference WAV
        │
        ├─ transformers tokenizer ───────────────┐
        │                                        │
        └─ PyTorch DACVAE encode ── CPU/NumPy ──▶ MLX RF-DiT sampler
                                                     │
                                                     ▼
                                      generated DACVAE latents in MLX
                                                     │
                                      CPU/NumPy ─────┘
                                                     ▼
                                      PyTorch DACVAE decode ── WAV
```

The DACVAE itself stays in PyTorch for v0. The MLX side owns condition encoding,
RF-DiT forward, and Euler RF sampling. This keeps the first practical prototype
focused on the model path that should benefit from MLX while avoiding a full
DACVAE port before the RF-DiT path is validated.

## CLI

```bash
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --reference-wav /path/to/reference.wav \
  --text "こんにちは。今日は良い天気です。" \
  --output /tmp/irodori.wav \
  --seconds 5 \
  --num-steps 40 \
  --codec-device cpu \
  --codec-runtime-mode persistent
```

Use `--model-config-json` when the converted weights do not match the default
base-v2 `ModelConfig`. The default tokenizer repo is
`sbintuitions/sarashina2.2-0.5b`, matching the base config.

For smoke tests without speaker conditioning, pass `--no-reference`; this builds
an unconditional speaker mask. Normal base-v2 generation should pass
`--reference-wav`.

`--codec-runtime-mode` controls how the PyTorch DACVAE boundary is hosted:

- `persistent` (default): keep the codec in-process and eagerly release PyTorch-side intermediates after encode/decode
- `subprocess`: run encode/decode in short-lived helper processes for investigation and benchmarking

The current recommendation is to keep `persistent` as the normal runtime mode.

## Python API

```python
from irodori_mlx.runtime import GenerationRequest, MLXDACVAERuntime, MLXRuntimeConfig
from irodori_mlx.config import ModelConfig

runtime = MLXDACVAERuntime(
    config=MLXRuntimeConfig(
        model_config=ModelConfig(),
        weights_path="/path/to/irodori-tts-500m-v2.npz",
    )
)
result = runtime.generate(
    GenerationRequest(
        text="こんにちは。",
        reference_wav="/path/to/ref.wav",
        output_wav="/tmp/out.wav",
        seconds=5.0,
        seed=0,
    )
)
print(result.output_wav)
```

## Boundary details

- PyTorch DACVAE encode returns `(batch, latent_steps, latent_dim)` tensors.
- The bridge converts PyTorch tensors through an explicit CPU/NumPy boundary into
  MLX arrays.
- The default bridge now also releases PyTorch-side tensors and backend cache
  state after reference encode and waveform decode so those allocations do not
  linger longer than necessary.
- Reference latents are patched on the MLX side with `latent_patch_size` before
  speaker conditioning.
- The sampler returns patched generated latents in MLX. The runtime unpatches
  them back to `(batch, latent_steps, latent_dim)` before decode.
- Decode converts MLX arrays back through CPU/NumPy into PyTorch tensors and
  calls upstream `DACVAECodec.decode_latent`.
- WAV writing uses `torchaudio` when available, then `soundfile`, then a stdlib
  PCM16 fallback.

## Runtime dependencies

The normal unit suite does not require the heavy runtime stack. Actual WAV
generation requires optional runtime packages:

- `mlx`
- `torch`
- upstream `irodori_tts` on `PYTHONPATH` or installed as a package
- DACVAE dependencies used by upstream `irodori_tts.codec.DACVAECodec`
- `transformers` / tokenizer dependencies
- `torchaudio` or `soundfile` for audio IO

## Current limitations

- The bridge is a prototype runtime surface, not a stable package API.
- The converted MLX `.npz` archive currently contains weights only; config is
  supplied separately or by the base-v2 defaults.
- DACVAE remains PyTorch-only in v0.
- The experimental subprocess codec mode is mainly for memory investigation; it
  is currently slower than the default persistent bridge.
- End-to-end audio quality still depends on full checkpoint conversion quality
  and the already documented RF sampler deviations.
