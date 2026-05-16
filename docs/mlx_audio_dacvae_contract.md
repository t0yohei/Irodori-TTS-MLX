# mlx-audio DACVAE artifact comparison

Issue: [#130](https://github.com/t0yohei/Irodori-TTS-MLX/issues/130)  
Parent epic: [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

This page records the DACVAE-specific result of comparing mlx-audio's bundled
MLX codec artifacts against this repository's current PyTorch
`irodori_tts.codec.DACVAECodec` bridge and local MLX codec artifact contract.

## Compared artifact layouts

mlx-audio publishes Irodori v2 and VoiceDesign v2 repositories with a bundled
codec directory:

```text
dacvae/
+-- config.json
+-- model.safetensors
```

The observed mlx-audio DACVAE config shape is:

```json
{
  "sample_rate": 48000,
  "encoder_rates": [2, 8, 10, 12],
  "decoder_rates": [12, 10, 8, 2],
  "n_codebooks": 16,
  "codebook_size": 1024,
  "codebook_dim": 32
}
```

This repository's current local MLX codec contract is intentionally different:

```text
dacvae-codec.npz
|-- sample_rate
|-- hop_length
|-- latent_dim
|-- decode_basis
|-- decode_bias
|-- encode_basis
|-- encode_bias
`-- metadata_json
```

The `.npz` file is a small explicit runtime contract used by
`MLXDACVAEBridge`. It validates runtime selection, metadata inspection, shape
checks, decode-only routing, full-MLX encode routing, and PyTorch fallback
boundaries without committing or downloading large upstream weights.

## Shape and channel conventions

The PyTorch bridge remains the reference behavior until a real mlx-audio DACVAE
conversion path has parity evidence:

- audio input/output is mono, shaped as one channel at the bridge boundary;
- runtime sample rate is `48000` Hz;
- runtime latents use `(B, T, D)` where `D == 32`;
- DACVAE internals use channel-first latent layout `(B, D, T)`;
- generated WAV output is mono at the codec sample rate;
- reference audio may be resampled to the codec sample rate before encode;
- fixture tests must cover sample rate, hop/step length, latent dimension,
  batch/channel conventions, and finite float32 values.

mlx-audio's `encoder_rates` and `decoder_rates` imply its own native codec time
stride. This repository must not infer drop-in compatibility from matching
`sample_rate` and `codebook_dim` alone. A converter has to prove the actual
hop/step semantics with fixed latent and waveform fixtures.

## Selected compatibility path

The selected path for v0.2 is:

1. Keep the PyTorch `DACVAECodec` bridge as the production-like fallback.
2. Keep the local `.npz` codec artifact contract as the only Irodori-TTS-MLX
   runtime MLX DACVAE artifact format for now.
3. Use mlx-audio's `dacvae/config.json` and `dacvae/model.safetensors` as
   source inputs for a future converter or adapter, not as files loaded directly
   by `MLXDACVAEBridge`.
4. Require fixed decode and encode parity fixtures before treating converted
   mlx-audio DACVAE weights as compatible.

For #131, the recommendation is explicit: ingest unquantized mlx-audio RF-DiT
artifacts into the existing hosted Irodori-TTS-MLX layout, but do not add direct
runtime loading of `dacvae/config.json + dacvae/model.safetensors`. If #131
chooses to preserve DACVAE information, it should record it as provenance and,
only after a codec converter exists, emit or point to a separate
`dacvae-codec.npz` companion artifact following
[codec_artifact_layout.md](codec_artifact_layout.md).

## Large-weight handling

Do not commit mlx-audio `dacvae/model.safetensors`, converted codec `.npz`
files, upstream PyTorch codec weights, reference audio, generated WAV files, or
Hugging Face cache snapshots to this repository.

Full local parity should be run by pointing at local artifacts:

```bash
# Decode fixture parity.
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
python scripts/check_dacvae_decode_parity.py \
  --latents-npy /tmp/irodori-dacvae/decode-latents.npy \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae/decode-parity \
  --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
  --codec-device cpu

# Encode fixture parity.
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
python scripts/check_dacvae_encode_parity.py \
  --audio-wav /tmp/irodori-dacvae/reference.wav \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae/encode-parity \
  --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
  --codec-device cpu
```

Use `docs/dacvae_decode_parity.md` and `docs/dacvae_encode_parity.md` for the
fixture-generation details and report schema. The checked-in contract tests use
small synthetic fixtures so CI can validate the selected compatibility path
without downloading multi-GiB upstream artifacts.

## Runtime differences to preserve

- mlx-audio bundles the codec beside RF-DiT weights; this repository keeps codec
  artifacts separate from hosted RF-DiT weights unless a manifest points to a
  companion codec artifact.
- mlx-audio stores DACVAE weights as `model.safetensors`; the current local MLX
  runtime reads `.npz` projection/fixture tensors.
- mlx-audio's config names codebooks and rates; this repository's runtime
  boundary names `sample_rate`, `hop_length`, `latent_dim`, and explicit encode
  and decode tensors.
- Direct loader compatibility is not supported. Silent fallback from the
  mlx-audio DACVAE directory shape would hide licensing and parity decisions.

