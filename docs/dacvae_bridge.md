# DACVAE bridge and MLX codec contract

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

## Upstream dependency boundary

The PyTorch side is provided by upstream `irodori_tts`, specifically
`irodori_tts.codec.DACVAECodec`. Install the upstream checkout into the same
venv with `python -m pip install -e /path/to/Irodori-TTS` (recommended) or make
it importable with `PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}`. The import
failure message in `PyTorchDACVAEBridge` points at the same two fixes.

This repo does not claim standalone DACVAE operation in v0.1. It owns the MLX
conditioning/model/sampling path and crosses into upstream PyTorch only for
DACVAE encode/decode. See [upstream_dependency.md](upstream_dependency.md) for
the full split.

## CLI

```bash
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --reference-wav /path/to/reference.wav \
  --text "こんにちは。今日は良い天気です。" \
  --output /tmp/irodori.wav \
  --seconds 5 \
  --preset balanced \
  --codec-device cpu \
  --codec-runtime-mode persistent
```

The user-facing preset surface is intentionally small:

- `--preset ultra-fast` → experimental latency-first `--num-steps 6 --cfg-guidance-mode joint --cfg-scale-text 1 --cfg-scale-caption 0 --cfg-scale-speaker 0`
- `--preset fast` → `--num-steps 12`
- `--preset balanced` → `--num-steps 24`
- `--preset quality` → `--num-steps 40`

These mappings come from the Apple Silicon local sweep in
[docs/benchmark-reports/2026-05-14-apple-silicon-num-steps-presets.md](benchmark-reports/2026-05-14-apple-silicon-num-steps-presets.md).
`ultra-fast` is the exception: it is deliberately experimental and follows the latency-first candidate from the #217/#220 sweep contract rather than a quality-validated default.
If you need exact control, pass `--num-steps` or the CFG flags explicitly and they will override the preset.

For repeatable local workflows, put common arguments in a JSON preset and only
override the per-run fields on the CLI:

```json
{
  "weights": "/path/to/irodori-tts-500m-v2.npz",
  "reference_wav": "/path/to/reference.wav",
  "output": "/tmp/irodori.wav",
  "seconds": 5.0,
  "preset": "balanced",
  "codec_device": "cpu"
}
```

```bash
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --config-json /path/to/generate-base.json \
  --text "こんにちは。今日は良い天気です。"
```

For repeated local experimentation, use `--requests-json` to keep one
`MLXDACVAERuntime` alive for multiple requests. Runtime-level fields such as
weights, model config, tokenizer repos, codec settings, and max token lengths are
loaded once from the CLI/config. Each request can override the generation fields
that change between outputs:

```json
[
  {
    "text": "最初のサンプルです。",
    "output": "/tmp/irodori-01.wav",
    "preset": "fast",
    "seed": 101
  },
  {
    "text": "少し落ち着いた読み方にします。",
    "caption": "落ち着いた自然な声",
    "output": "/tmp/irodori-02.wav",
    "num_steps": 24,
    "seed": 102
  }
]
```

```bash
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --config-json /path/to/generate-base.json \
  --requests-json /path/to/requests.json \
  --metadata-json /tmp/irodori-batch.json
```

The request objects support `text`, `output`, `reference_wav`,
`no_reference`, `caption`, `seconds`, `duration_scale`, `preset`, `num_steps`,
the CFG knobs, `seed`, `max_reference_seconds`, and `no_context_kv_cache`. Use
one-shot mode for isolated commands and shell pipelines; use batch mode when you
are iterating on prompts or seeds and want to avoid paying model/tokenizer/codec
startup cost for every output. `--json` and `--metadata-json` return a `results`
array in batch mode.

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

A minimal v3 validation run can reuse that unconditional path to exercise the
duration predictor without shipping a reference asset:

```bash
PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH \
python3 scripts/generate_wav.py \
  --weights /path/to/irodori-tts-500m-v3.npz \
  --model-config-json /path/to/v3-model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav \
  --preset balanced
```

That command intentionally omits `--seconds`, so checkpoints with
`use_duration_predictor=true` follow the predicted-duration path. Add
`--reference-wav` for real speaker-conditioned runs when you want the generated
voice to track a specific sample more closely.

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
- `mlx-decode`: decode generated latents with a local MLX codec artifact; reference-audio encode remains on the in-process PyTorch bridge
- `mlx-decode-subprocess`: decode generated latents with a local MLX codec artifact; reference-audio encode remains on the subprocess PyTorch bridge
- `mlx`: use the local MLX codec artifact for both reference-audio encode and generated-latent decode only when the artifact includes executable Semantic-DACVAE encoder and decoder tensors

The current recommendation is to keep `persistent` as the normal runtime mode
unless you are explicitly validating MLX codec artifacts.

For v0.2 codec-port work, `--codec-runtime-mode mlx-decode` selects the
decode-only MLX path. No-reference v3 and VoiceDesign generation can then write
the final WAV without importing the upstream PyTorch DACVAE bridge, because
reference encode is not required and generated-latent decode uses the local MLX
codec artifact. Reference-audio requests still route encode through the PyTorch
bridge and fail with a direct fallback message when that upstream bridge is not
installed:

```bash
python3 scripts/generate_wav.py \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --model-config-json /path/to/model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-mlx-codec.wav \
  --seconds 2 \
  --codec-runtime-mode mlx-decode \
  --codec-path /path/to/dacvae-codec.npz
```

The decode-only MLX codec artifact contract accepts either the legacy fixture
arrays or the executable Semantic-DACVAE decoder tensors. Full `mlx` mode no
longer accepts the legacy linear fixture arrays; it requires both executable
encoder and executable decoder tensors from the real converter.

The local `.npz` file contains:

- `sample_rate`, `hop_length`, `latent_dim` scalar arrays
- `decode_basis` shaped `(latent_dim, hop_length)`
- `decode_bias` shaped `(hop_length,)`
- optional legacy `encode_basis` / `encode_bias` fixture arrays, which are
  inspected for historical compatibility but are not sufficient for
  `--codec-runtime-mode mlx`
- `dacvae_decoder_exec/...` for executable MLX Semantic-DACVAE decode
- `dacvae_encoder_exec/...` for executable MLX Semantic-DACVAE reference encode
- optional `metadata_json` scalar string with the same metadata and provenance

The legacy fixture contract remains small enough for checked-in unit tests and
decode-only smoke coverage. The full MLX encode path is enabled only by the
executable encoder tensor layout: it downmixes to mono, truncates to
`--max-reference-seconds`, resamples to the artifact sample rate, applies the
configured `normalize_db` or peak safety, right-pads to the hop multiple, and
returns runtime latents as `(batch, latent_steps, latent_dim)` before the normal
speaker patching/mask logic runs.

It is not, by itself, a redistributed Semantic-DACVAE checkpoint. Real acoustic
parity requires a converted codec artifact produced from the supported upstream
codec weights and validated with fixed latent/audio fixtures.

Generation metadata distinguishes the request-specific boundary. On the
no-reference `mlx-decode` path, `codec_decode_backend` is `"mlx"` and
`codec_encode_backend` is `"not-required"`. On reference-audio
`mlx-decode` requests, `codec_decode_backend` remains `"mlx"`, while
`codec_encode_backend` reports the PyTorch fallback backend used for reference
encode. On reference-audio `mlx` requests with an executable encoder/decoder
artifact, both `codec_encode_backend` and `codec_decode_backend` are
`"mlx"`; that is the fully-MLX generation boundary for speaker-conditioned
requests.

The upstream architecture, runtime constants, logical tensor groups, and known
conversion blockers for that real codec artifact are tracked in
[dacvae_architecture.md](dacvae_architecture.md) and its machine-readable
[dacvae_codec_contract.json](dacvae_codec_contract.json).
The hosted/local artifact layout, manifest pointer shape, runtime capability
checks, and checkpoint-family fallback policy are defined in
[codec_artifact_layout.md](codec_artifact_layout.md).

Decode-only parity fixtures and the reproducible upstream PyTorch versus MLX
comparison command are documented in
[dacvae_decode_parity.md](dacvae_decode_parity.md). That check owns fixed-latent
decode evidence for #184. Encode parity fixtures, length/mask contract checks,
and the reproducible fixed-audio command for #174 are documented in
[dacvae_encode_parity.md](dacvae_encode_parity.md).

## Duration semantics

- `--seconds` is an explicit manual override.
- When `--seconds` is omitted and the loaded `ModelConfig` enables `use_duration_predictor`, the runtime predicts latent length from the current text/reference conditions.
- `--duration-scale` scales only that predicted length; it has no effect when `--seconds` is set.
- When `--seconds` is omitted for checkpoints without the duration predictor, the MLX runtime estimates a bounded fallback from normalized text length instead of using a fixed 5 seconds. Short prompts can resolve below 5 seconds; moderately longer VoiceDesign prompts get more room without allocating a very long tail.
- If a generated sample still sounds clipped or develops a buzzer-like tail, pass `--seconds` explicitly. Manual `--seconds` remains the highest-priority duration rule.
- Hosted v3 validation (`scripts/run_v3_generation_ci.py` / `.github/workflows/v3-hosted-generation.yml`) intentionally omits `--seconds` and asserts `duration_mode="predicted"` in the JSON payload so the first-class v3 semantics stay exercised.

JSON output now includes `duration_mode`, `requested_seconds`,
`resolved_seconds`, `codec_encode_backend`, `codec_decode_backend`, and
`boundaries.codec.capabilities` so automation can tell which duration rule,
codec path, MLX artifact availability, and PyTorch fallback policy were used.

## Internal runtime example

The following import path is useful for repository development and tests, but it
is not a stable public Python API. Downstream package users should call the
installed console scripts documented above and in
[public_api_stability.md](public_api_stability.md).

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
- The MLX codec mode loads a local artifact, keeps encode/decode math in MLX,
  and uses NumPy only at the WAV/file boundary.
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
generation should use a supported packaged runtime environment from
[docs/packaging.md](packaging.md) (currently Python 3.11 through 3.14): install this
repo with `pip install -e ".[runtime]"` and make upstream `irodori_tts`
importable from the same venv or `PYTHONPATH`.

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
- VoiceDesign / caption-conditioned runtime/model support is scoped to the
  inspected `Aratako/Irodori-TTS-500M-v2-VoiceDesign` family rather than every
  historical caption-conditioned checkpoint.
- V3 runtime support depends on a config/weights pair that still matches the
  inspected public checkpoint family; manual `--seconds` remains the escape hatch
  when you want exact duration control instead of the predictor.
- Codec watermarking remains optional and only has an effect when the upstream
  DACVAE runtime exposes watermark support; `--enable-watermark` should be
  treated as best-effort rather than guaranteed output tagging.
- DACVAE remains PyTorch-backed by default for v0.1-style runs. The v0.2
  `--codec-runtime-mode mlx` path is available for local converted codec
  artifacts and parity fixtures, but this repository still does not bundle
  Semantic-DACVAE weights.
- The experimental subprocess codec mode is mainly for memory investigation; it
  is currently slower than the default persistent bridge.
- End-to-end audio quality still depends on full checkpoint conversion quality
  and the already documented RF sampler deviations.

## Parity fixture boundary

Focused local tests cover the MLX codec artifact contract without large model
downloads:

```bash
python -m pytest tests/test_runtime_bridge.py -k 'mlx_dacvae or mlx_codec'
```

Full Semantic-DACVAE parity should be run only when a converted codec artifact
and the upstream PyTorch baseline are available locally. Use fixed latent/audio
fixtures and compare both directions:

```bash
export IRODORI_MLX_DACVAE_CODEC_NPZ=/path/to/converted-dacvae-codec.npz
export IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY=/path/to/decode-latents.npy
export IRODORI_MLX_DACVAE_DECODE_AUDIO_NPY=/path/to/upstream-decoded-audio.npy
export IRODORI_MLX_DACVAE_ENCODE_AUDIO_WAV=/path/to/reference.wav
export IRODORI_MLX_DACVAE_ENCODE_LATENTS_NPY=/path/to/upstream-encoded-latents.npy
python -m pytest tests/test_dacvae_mlx_parity_fixtures.py -q
```

That command is a fixture boundary, not a network-downloading test; the artifact
and upstream-generated `.npy` fixture outputs must be prepared by the runner.
