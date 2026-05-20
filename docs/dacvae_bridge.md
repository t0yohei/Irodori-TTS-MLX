# DACVAE runtime

The public generation runtime uses one codec mode: `--codec-runtime-mode mlx`.

That mode resolves an approved hosted or local DACVAE codec artifact and performs encode/decode through MLX-native artifact tensors. It does not require upstream `irodori_tts.codec.DACVAECodec`, `torch`, or `torchaudio` for generation.

## CLI surface

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-runtime-mode mlx \
  --text "こんにちは。今日は良い天気です。" \
  --no-ref \
  --output-wav /tmp/irodori-v3.wav
```

The CLI defaults to the approved hosted codec artifact. Use `--codec-artifact-dir` for a local hosted-layout artifact or `--codec-path /path/to/dacvae-codec.npz` for a direct local artifact.

See [codec_artifact_layout.md](codec_artifact_layout.md) for the artifact manifest, tensor layout, provenance, and license-review contract.

## Metadata

No-reference generation reports:

- `codec_decode_backend: "mlx"`
- `codec_encode_backend: "not-required"`

Reference-audio generation with an encode-capable codec artifact reports:

- `codec_decode_backend: "mlx"`
- `codec_encode_backend: "mlx"`

The `irodori_mlx` runtime classes are internal implementation details for the installed console scripts, not a stable public Python API.
