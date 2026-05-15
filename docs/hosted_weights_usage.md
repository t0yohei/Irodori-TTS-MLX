# Hosted converted weights usage

Issue: [#85 Document hosted converted weights usage and fallback local conversion](https://github.com/t0yohei/irodori-tts-mlx/issues/85)  
Parent: [#78 v0.2: Support pre-converted MLX weights from Hugging Face](https://github.com/t0yohei/irodori-tts-mlx/issues/78)

This page explains the user-facing v0.2 flow for **hosted pre-converted MLX RF-DiT weights** and the supported fallback when no approved hosted artifact is available.

Hosted weights are a convenience path, not a new model source or redistribution waiver. The runtime boundary remains:

> MLX RF-DiT inference + upstream PyTorch DACVAE encode/decode bridge

That means users still need the runtime dependencies and upstream `irodori_tts` / `DACVAECodec` import path described in [dacvae_bridge.md](dacvae_bridge.md) and [upstream_dependency.md](upstream_dependency.md). Hosted converted weights do **not** bundle upstream source code, Semantic-DACVAE codec weights, reference audio, generated samples, Hugging Face cache snapshots, or unaudited tokenizer/model artifacts.

## When to use hosted weights

Use a hosted converted weights repo when all of these are true:

1. the repo follows the [hosted weights layout contract](hosted_weights_layout.md);
2. `irodori_mlx_manifest.json` reports `license_review.status: "approved"`;
3. the hosted README/model card links to the provenance and license review for the exact upstream checkpoint revision;
4. your installed `irodori-tts-mlx` version meets `runtime.minimum_irodori_tts_mlx_version` in the manifest;
5. you accept the upstream model-card terms and ethical-use restrictions for your prompts, reference audio, and generated audio.

If any of those checks fail, use the [local conversion fallback](#fallback-local-conversion) instead.

## Hosted quick path

The v0.2 CLI path introduced by the hosted-loader work in #82 accepts a Hugging Face repo id with `--weights-repo`. These flags are a v0.2 contract and are not executable on the current `main` CLI until that implementation lands. The concrete repo id below is an example of the expected shape; use only a repository that has actually been published and approved by the v0.2 publication checklist.

```bash
python -m pip install -e ".[runtime]"
python -m pip install huggingface_hub  # required for Hugging Face repo resolution if not already installed

PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-repo t0yohei/irodori-tts-mlx-voicedesign-v2-500m \
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
- still uses the upstream PyTorch DACVAE bridge for codec encode/decode.

For a v3 hosted artifact, use a v3-approved repo id and omit `--caption`. If the v3 manifest says `supports_predicted_duration: true`, omit `--seconds` to exercise predicted-duration generation:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-repo t0yohei/irodori-tts-mlx-v3-500m \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted.wav \
  --preset balanced
```

## Local hosted-layout directory

Use `--weights-dir` when you have the same v0.2 layout on disk, for example during private staging, CI fixtures, or local-only conversions that cannot be redistributed publicly:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-dir /models/irodori-tts-mlx-voicedesign-v2-500m \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-local-layout.wav
```

A local directory may be useful even when `license_review.status` is `pending`, but do not publish it or document it as a public model until the review is approved. Local-only use does not remove your obligation to follow the upstream terms.

## Fallback: local conversion

Local conversion remains the supported fallback for:

- unaudited, third-party, fine-tuned, quantized, LoRA, renamed, or otherwise modified checkpoints;
- hosted repos with missing files, incompatible manifest/runtime metadata, or unapproved redistribution status;
- offline or private workflows where Hugging Face resolution is unavailable;
- users who prefer to keep all model artifacts local.

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

PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
python scripts/generate_wav.py \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --reference-wav /path/to/reference.wav \
  --output "$WORK/irodori-local.wav" \
  --preset balanced
```

For v3, verify that `$WORK/model_config.json` includes `"use_duration_predictor": true`, then use `--no-reference` and omit `--seconds` when you want predicted duration. For VoiceDesign, verify that it includes `"use_caption_condition": true` and add the `--caption` argument described in [caption_condition_support.md](caption_condition_support.md).

## Provenance and licensing checklist

Every hosted converted weights README/model card should make these points clear:

- the upstream checkpoint repo id and exact revision used for conversion;
- that the artifact is a converted MLX `.npz`, not the original upstream checkpoint;
- the `t0yohei/irodori-tts-mlx` converter version or commit SHA and conversion command;
- the upstream model-card license and ethical-use restrictions, including no impersonation or misleading synthetic speech where applicable;
- that Semantic-DACVAE codec weights, upstream source code, reference audio, generated audio, and Hugging Face cache snapshots are not bundled;
- the runtime still requires the upstream PyTorch DACVAE bridge unless a future full-MLX codec is implemented;
- a link to the license audit / publication decision, currently [preconverted_weights_redistribution_audit.md](preconverted_weights_redistribution_audit.md) for the reviewed v0.2 candidate families.

The current audit allows only the explicitly reviewed Irodori-TTS checkpoint families to proceed with conditions. Anything outside that list is **local-conversion-only** until separately audited.
