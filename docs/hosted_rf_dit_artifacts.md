# Hosted RF-DiT artifact publication status

Issue: [#187 Publish approved hosted RF-DiT artifact for v3](https://github.com/t0yohei/Irodori-TTS-MLX/issues/187)
Parent: [#160 v0.2 follow-up: DACVAE codec, packaging, and hosted artifacts](https://github.com/t0yohei/Irodori-TTS-MLX/issues/160)

This page records the v0.2 public RF-DiT hosted-artifact status that runtime users may rely on. It does not upload weights, create model repositories, or approve any private/local artifact path. The machine-readable in-package contract is irodori_mlx.hosted_artifacts.HOSTED_RF_DIT_ARTIFACTS.

## Publication status

| Family | Upstream checkpoint | Public hosted repo | Status | Runtime use |
| --- | --- | --- | --- | --- |
| VoiceDesign v2 caption-conditioned | Aratako/Irodori-TTS-500M-v2-VoiceDesign | t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign at bf877a3beb7d921dc6bfb2b6812d02be07f39f2a | Approved public artifact | Use --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign after installing huggingface_hub. |
| v3 speaker-conditioned / duration predictor | Aratako/Irodori-TTS-500M-v3 | t0yohei/Irodori-TTS-MLX-500M-v3 at 078ffb11ffad92e6dde237a6abef730f4341b359 | Approved public artifact | Use --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 after installing huggingface_hub. |

Both families remain covered by the conditional redistribution audit in [preconverted_weights_redistribution_audit.md](preconverted_weights_redistribution_audit.md). VoiceDesign and v3 now have concrete public repos whose manifests report license_review.status: "approved".

## VoiceDesign contract

The approved VoiceDesign repo must keep the [hosted weights layout](hosted_weights_layout.md) at the repository root:

- irodori_mlx_manifest.json
- model_config.json
- tokenizer_config.json
- conversion_metadata.json
- weights.npz
- checksums.sha256
- README.md
- LICENSE.md

Required manifest/runtime facts:

- format: "irodori-tts-mlx-weights"
- format_version: "0.2"
- family: "voicedesign"
- upstream_checkpoint: "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
- runtime.supports_caption: true
- runtime.supports_no_reference: true
- runtime.supports_predicted_duration: false
- license_review.status: "approved"

Smoke command:

    PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
    irodori-tts-generate \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
      --weights-revision bf877a3beb7d921dc6bfb2b6812d02be07f39f2a \
      --text "こんにちは。今日は良い天気です。" \
      --caption "落ち着いた女性の声" \
      --no-reference \
      --output /tmp/irodori-hosted.wav \
      --preset balanced \
      --json

## v3 contract

The approved v3 repo must keep the [hosted weights layout](hosted_weights_layout.md) at the repository root:

- irodori_mlx_manifest.json
- model_config.json
- tokenizer_config.json
- conversion_metadata.json
- weights.npz
- checksums.sha256
- README.md
- LICENSE.md

Required manifest/runtime facts:

- format: "irodori-tts-mlx-weights"
- format_version: "0.2"
- family: "v3"
- upstream_checkpoint: "Aratako/Irodori-TTS-500M-v3"
- runtime.supports_caption: false
- runtime.supports_no_reference: true
- runtime.supports_predicted_duration: true
- license_review.status: "approved"

Smoke command:

    PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
    irodori-tts-generate \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
      --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
      --text "こんにちは。今日は良い天気です。" \
      --no-reference \
      --output /tmp/irodori-v3-hosted.wav \
      --preset balanced \
      --json

The v3 hosted smoke intentionally omits --caption and --seconds. The manifest sets supports_predicted_duration: true, so omitting --seconds exercises the v3 duration predictor.

## Local fallback

The documented local conversion fallback remains conversion from the upstream checkpoint when hosted resolution is unavailable or a user is working with an unapproved/private checkpoint:

```bash
CHECKPOINT=/path/to/Irodori-TTS-500M-v3/model.safetensors
WORK=/tmp/irodori-v3-local
mkdir -p "$WORK"

irodori-tts-inspect "$CHECKPOINT" --json > "$WORK/checkpoint-inspect.json"
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
irodori-tts-convert "$CHECKPOINT" "$WORK/weights.npz"
irodori-tts-generate \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output "$WORK/irodori-v3.wav" \
  --preset balanced
```

Do not replace an approved hosted status with a local filesystem path, a private Hugging Face repo, or a mutable staging branch. Publication requires an approved public repo id, immutable revision, matching checksums, README/model-card provenance, and license_review.status: "approved" in the hosted manifest.
