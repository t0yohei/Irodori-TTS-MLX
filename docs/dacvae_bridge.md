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

For repeatable local workflows, put common arguments in a JSON preset and only
override the per-run fields on the CLI:

```json
{
  "weights": "/path/to/irodori-tts-500m-v2.npz",
  "reference_wav": "/path/to/reference.wav",
  "output": "/tmp/irodori.wav",
  "seconds": 5.0,
  "num_steps": 40,
  "codec_device": "cpu"
}
```

```bash
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --config-json /path/to/generate-base.json \
  --text "こんにちは。今日は良い天気です。"
```

For automation, use `--json` to emit a machine-readable payload to stdout or
`--metadata-json /path/to/result.json` to save the same generation metadata,
timings, request fields, and runtime boundary description to disk.

Use `--model-config-json` when the converted weights do not match the default
base-v2 `ModelConfig`. The default tokenizer repo is
`sbintuitions/sarashina2.2-0.5b`, matching the base config.

For smoke tests without speaker conditioning, pass `--no-reference`; this builds
an unconditional speaker mask. Normal base-v2 generation should pass
`--reference-wav`. The CLI now reports clearer validation errors when
`--reference-wav` / `--no-reference` are misused.

Caption-conditioned / VoiceDesign-style configs already use a different runtime
path: they load a caption tokenizer, accept `--caption`, and can run without a
speaker reference because speaker conditioning is disabled in that config. The
checked-in weight converter now supports the inspected VoiceDesign checkpoint
family, and hosted Apple Silicon CI can exercise the full `generate_wav.py
--caption ...` path with a real public checkpoint. See
[caption_condition_support.md](caption_condition_support.md) for the current
support matrix and runner caveats.

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
generation should use the packaged Python 3.11 runtime environment from
[docs/packaging.md](packaging.md): install this repo with `pip install -e ".[runtime]"`
and make upstream `irodori_tts` importable from the same venv or `PYTHONPATH`.

In practical terms, the runtime needs:

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
- VoiceDesign / caption-conditioned runtime/model support is only partial until
  checkpoint conversion grows beyond the base v2 layout.
- DACVAE remains PyTorch-only in v0.
- The experimental subprocess codec mode is mainly for memory investigation; it
  is currently slower than the default persistent bridge.
- End-to-end audio quality still depends on full checkpoint conversion quality
  and the already documented RF sampler deviations.
