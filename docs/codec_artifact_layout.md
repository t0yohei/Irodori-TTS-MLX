# Hosted/local DACVAE codec artifact layout

DACVAE codec artifacts are separate from RF-DiT weights repositories. They do not bundle Semantic-DACVAE weights by default. A RF-DiT weights repository can point to a codec companion, but the codec lives in a dedicated Hugging Face model repository such as `Irodori-TTS-MLX-DACVAE-Codec`, or in a user-managed local layout.

## Files

A local or hosted codec layout contains:

- `irodori_dacvae_codec_manifest.json`
- `dacvae-codec.npz`
- `codec_metadata.json`
- `checksums.sha256`

The `.npz` includes scalar metadata such as `sample_rate`, `hop_length`, `latent_dim`, `artifact_kind`, and `metadata_json`. Fixture artifacts may include `decode_basis`, `decode_bias`, `encode_basis`, and `encode_bias`; executable Semantic-DACVAE artifacts include `semantic_encoder_manifest_json`, `dacvae_encoder_exec/<module-key>`, `dacvae_decoder/<state-dict-key>`, and `dacvae_decoder_exec/<module-key>`.

`scripts/convert_dacvae_decoder.py` writes executable encoder/decoder tensor layouts from an upstream codec repo id, source file, and exact revision. Every public artifact must carry provenance and license-review status.

## Manifest pointer

RF-DiT hosted weights may point at a companion codec artifact, but the codec remains a separate contract:

```json
{
  "codec": {
    "artifact_format": "irodori-tts-mlx-dacvae-codec",
    "source_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
    "runtime_modes": ["mlx"]
  }
}
```

Use `--codec-artifact-repo` for an approved hosted codec layout, `--codec-artifact-dir` for a local hosted-layout artifact, or `--codec-path` for a direct private `dacvae-codec.npz`.

## Runtime capability

`describe_codec_capabilities()` exposes the same data under `boundaries.codec.capabilities`. The only public generation codec mode is `mlx`.

The capability report covers:

- `base_v2`, `voicedesign`, and `v3` checkpoint-family routing;
- `has_executable_mlx_decode=true` and `has_executable_mlx_encode=true`;
- `runtime_status.mlx_encoder_execution=available_unvalidated`;
- `available_unvalidated` status until local validation evidence is attached;
- `--no-ref` no-reference requests where encode is not required.

Encode parity remains owned by local validation reports such as `scripts/check_dacvae_decode_parity.py` and issue #184. Those checks are evidence collection tools; they are not runtime fallback modes.
