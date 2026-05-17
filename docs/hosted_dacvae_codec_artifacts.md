# Hosted DACVAE codec artifact publication status

Issue: [#188 Publish and pin hosted DACVAE codec artifact](https://github.com/t0yohei/Irodori-TTS-MLX/issues/188)

This page records the v0.2 public DACVAE codec hosted-artifact status. The machine-readable in-package contract is `irodori_mlx.hosted_artifacts.HOSTED_DACVAE_CODEC_ARTIFACT`.

## Current status

The hosted artifact is prepared as a Hugging Face pull request, not pushed directly to the model repository main branch:

- repo: `t0yohei/Irodori-TTS-MLX-DACVAE-Codec`
- Hugging Face PR: https://huggingface.co/t0yohei/Irodori-TTS-MLX-DACVAE-Codec/discussions/1
- PR commit: `16d64e0978afe79c46b971405bba4f464cc743f8`
- publication status: `hf-pr-open`
- license review: approved for personal OSS, research, and development publication
- source codec: `Aratako/Semantic-DACVAE-Japanese-32dim`
- source revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- DACVAE package revision used for conversion: `414c20785fc3a28373073ea8ef7a1316eeeaca6e`

Do not document the artifact as an approved public `--codec-artifact-repo` dependency until the Hugging Face PR is merged and the resulting repository revision is pinned.

## Prepared artifact contract

The HF PR adds this hosted codec layout:

- `irodori_dacvae_codec_manifest.json`
- `dacvae-codec.npz`
- `codec_metadata.json`
- `checksums.sha256`
- `README.md`
- `LICENSE.md`
- `validation/convert-dacvae-codec.json`
- `validation/dacvae-decode-parity.json`
- `validation/dacvae-encode-parity.json`

Required runtime facts in the prepared manifest:

- format: `irodori-tts-mlx-dacvae-codec`
- format version: `0.2`
- sample rate: `48000`
- hop length: `1920`
- latent dim: `32`
- `supports_mlx_decode: true`
- `supports_mlx_encode: true`
- `requires_pytorch_fallback: false`
- `license_review.status: "approved"`

## Validation evidence

The prepared artifact was converted locally from `weights.pth` with `scripts/convert_dacvae_decoder.py`.

Decode parity against the upstream PyTorch bridge passed:

- max_abs: `3.0517578125e-05`
- mean_abs: `1.4040205087439972e-07`
- rmse: `2.0699590095318854e-06`
- cosine: `0.9999999403953552`

Encode parity against the upstream PyTorch bridge passed:

- max_abs: `1.33514404296875e-05`
- mean_abs: `2.287564711878076e-06`
- rmse: `3.305609425297007e-06`
- cosine: `1.0`

## Post-merge update

After the Hugging Face PR is merged:

1. Record the merged HF repository revision here.
2. Set `HOSTED_DACVAE_CODEC_ARTIFACT.publication_status` to `approved-public`.
3. Set `HOSTED_DACVAE_CODEC_ARTIFACT.revision` to the merged 40-character HF commit.
4. Replace examples that use `<approved-hf-commit>` with that pinned revision.
5. Run hosted codec resolution against the pinned revision.

