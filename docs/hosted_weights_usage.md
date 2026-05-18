# Hosted converted weights usage

Issue: [#85 Document hosted converted weights usage and fallback local conversion](https://github.com/t0yohei/Irodori-TTS-MLX/issues/85)  
Parent: [#78 v0.2: Support pre-converted MLX weights from Hugging Face](https://github.com/t0yohei/Irodori-TTS-MLX/issues/78)

This page explains the user-facing v0.2 flow for **hosted pre-converted MLX RF-DiT weights** and the supported fallback when no approved hosted artifact is available. The current family-by-family publication state is tracked in [hosted_rf_dit_artifacts.md](hosted_rf_dit_artifacts.md).

Hosted weights are a convenience path, not a new model source or redistribution waiver. The runtime boundary is:

> MLX RF-DiT inference + MLX DACVAE codec artifact encode/decode

That means users need the runtime dependencies and an approved hosted or local DACVAE codec artifact. Hosted converted weights do **not** bundle upstream source code, Semantic-DACVAE codec weights, reference audio, generated samples, Hugging Face cache snapshots, or unaudited tokenizer/model artifacts.
When an approved DACVAE codec artifact exists, resolve it separately with
`--codec-artifact-repo` or `--codec-artifact-dir`; RF-DiT `--weights-repo` and
DACVAE codec artifacts intentionally remain separate contracts.

## When to use hosted weights

Use a hosted converted weights repo when all of these are true:

1. the repo follows the [hosted weights layout contract](hosted_weights_layout.md);
2. `irodori_mlx_manifest.json` reports `license_review.status: "approved"`;
3. the hosted README/model card links to the provenance and license review for the exact upstream checkpoint revision;
4. your installed `irodori-tts-mlx` version meets `runtime.minimum_irodori_tts_mlx_version` in the manifest;
5. you accept the upstream model-card terms and ethical-use restrictions for your prompts, reference audio, and generated audio.

If any of those checks fail, use the [local conversion fallback](#fallback-local-conversion) instead.

## Hosted quick path

The current CLI accepts a Hugging Face repo id with `--weights-repo`, or the
same hosted-layout contract from disk with `--weights-dir`. The concrete repo
ids below are examples of the expected shape; use only repositories that have
actually been published and approved by the v0.2 publication checklist.

```bash
python -m pip install -e ".[runtime]"
python -m pip install huggingface_hub  # required for Hugging Face repo resolution if not already installed

irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-hosted.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-hosted-metadata.json \
  --json
```

Expected behavior:

- the CLI resolves the repository snapshot;
- validates `irodori_mlx_manifest.json`, required files, checksum coverage, runtime flags, and approved license review metadata;
- loads `model_config.json`, `tokenizer_config.json`, and `weights.npz` through the same internal runtime path used by local hosted-layout directories;
- uses the approved hosted DACVAE codec artifact for codec encode/decode unless `--codec-artifact-dir` or `--codec-path` is provided.

An approved public v3 hosted artifact is now recorded in [hosted_rf_dit_artifacts.md](hosted_rf_dit_artifacts.md). For v3, use the approved repo id, omit `--caption`, and omit `--seconds` to exercise predicted-duration generation:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted.wav \
  --preset balanced
```

To use an approved hosted DACVAE codec artifact for the MLX decode path, add
the codec artifact repo explicitly. This is a hosted artifact contract only; do
not use unpublished or unapproved repos as public examples.

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-runtime-mode mlx \
  --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
  --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted-codec.wav
```

The CLI validates `irodori_dacvae_codec_manifest.json`, the declared
`dacvae-codec.npz`, `codec_metadata.json`, checksum coverage, and approved
license review before passing the resolved codec file to the existing
`--codec-path` runtime.

For VoiceDesign v2 hosted artifacts, `--seconds` can also be omitted for the normal path. Because those checkpoints do not have the v3 learned duration predictor, the runtime reports `duration_mode: "estimated"` and estimates a bounded duration from normalized text length plus conservative caption style hints. Caption text is treated as voice/style guidance rather than spoken content, so long captions do not directly lengthen the transcript. Keep `--seconds` as the manual override when a specific prompt still clips or when the generated tail becomes audibly over-extended.

## Local hosted-layout directory

Use `--weights-dir` when you have the same v0.2 layout on disk, for example during private staging, CI fixtures, or local-only conversions that cannot be redistributed publicly:

```bash
irodori-tts-generate \
  --weights-dir /models/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-local-layout.wav
```

A local directory may be useful even when `license_review.status` is `pending`, but do not publish it or document it as a public model until the review is approved. Local-only use does not remove your obligation to follow the upstream terms.
The same staging rule applies to `--codec-artifact-dir` for DACVAE codec
artifacts: `pending` is acceptable locally, while public hosted codec repos must
be approved.

## Fallback: local conversion

Local conversion remains the supported fallback for:

- unaudited, third-party, fine-tuned, quantized, LoRA, renamed, or otherwise modified checkpoints;
- unquantized mlx-audio Irodori v2/VoiceDesign artifacts that first need the supported `irodori-tts-adapt-mlx-audio` interop path;
- hosted repos with missing files, incompatible manifest/runtime metadata, or unapproved redistribution status;
- hosted DACVAE codec repos with missing `irodori_dacvae_codec_manifest.json`, checksum mismatches, or unapproved license review status;
- offline or private workflows where Hugging Face resolution is unavailable;
- users who prefer to keep all model artifacts local.

For mlx-audio artifacts such as `mlx-community/Irodori-TTS-500M-v2-fp16`, do not pass the mlx-community repo directly to `--weights-repo`; those repos contain `config.json` and `model.safetensors`, not this project's `irodori_mlx_manifest.json` contract. Download or point at the snapshot directory, adapt it, then use the emitted hosted layout:

```bash
irodori-tts-adapt-mlx-audio /path/to/mlx-audio-snapshot /tmp/irodori-mlx-audio-hosted --source-repo mlx-community/Irodori-TTS-500M-v2-fp16 --source-revision <commit-sha>
irodori-tts-generate --weights-dir /tmp/irodori-mlx-audio-hosted --text "こんにちは。今日は良い天気です。" --reference-wav /path/to/reference.wav --output /tmp/irodori-local.wav
```

The adapter rejects 4-bit/8-bit mlx-audio repos with an explicit quantization error. Quantized artifacts remain local-conversion-only or unsupported until this runtime has a designed quantized MLX loading path.

```bash
python -m pip install -e ".[runtime]" safetensors

CHECKPOINT=/path/to/model.safetensors
WORK=/tmp/irodori-local-conversion
mkdir -p "$WORK"

python scripts/inspect_checkpoint.py "$CHECKPOINT" --json > "$WORK/checkpoint-inspect.json"
python - "$WORK/checkpoint-inspect.json" > "$WORK/model_config.json" <<'PY'
import json
import sys
from dataclasses import fields
from irodori_mlx.config import ModelConfig
payload = json.load(open(sys.argv[1]))
allowed = {field.name for field in fields(ModelConfig)}
config = {key: value for key, value in payload["config"].items() if key in allowed}
print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
PY
python scripts/convert_weights.py "$CHECKPOINT" "$WORK/weights.npz" --dry-run --json \
  > "$WORK/convert-dry-run.json"
python scripts/convert_weights.py "$CHECKPOINT" "$WORK/weights.npz"

python scripts/generate_wav.py \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --reference-wav /path/to/reference.wav \
  --output "$WORK/irodori-local.wav" \
  --preset balanced
```

For v3, verify that `$WORK/model_config.json` includes `"use_duration_predictor": true`, then use `--no-reference` and omit `--seconds` when you want predicted duration. For VoiceDesign, verify that it includes `"use_caption_condition": true`, add the `--caption` argument described in [caption_condition_support.md](caption_condition_support.md), and omit `--seconds` to use the bounded text-length fallback unless you need a manual duration override.

## Provenance and licensing checklist

Every hosted converted weights README/model card should make these points clear:

- the upstream checkpoint repo id and exact revision used for conversion;
- that the artifact is a converted MLX `.npz`, not the original upstream checkpoint;
- the `t0yohei/Irodori-TTS-MLX` converter version or commit SHA and conversion command;
- the upstream model-card license and ethical-use restrictions, including no impersonation or misleading synthetic speech where applicable;
- that Semantic-DACVAE codec weights, upstream source code, reference audio, generated audio, and Hugging Face cache snapshots are not bundled;
- the runtime uses the approved hosted/local MLX DACVAE codec artifact instead of an upstream PyTorch bridge;
- a link to the license audit / publication decision, currently [preconverted_weights_redistribution_audit.md](preconverted_weights_redistribution_audit.md) for the reviewed v0.2 candidate families.

The current audit allows only the explicitly reviewed Irodori-TTS checkpoint families to proceed with conditions. Anything outside that list is **local-conversion-only** until separately audited.
