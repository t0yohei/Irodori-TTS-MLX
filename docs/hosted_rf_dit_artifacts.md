# Hosted RF-DiT artifact publication status

Issue: [#157 Publish approved hosted RF-DiT weights for v3 and VoiceDesign](https://github.com/t0yohei/Irodori-TTS-MLX/issues/157)  
Parent: [#160 v0.2 follow-up: DACVAE codec, packaging, and hosted artifacts](https://github.com/t0yohei/Irodori-TTS-MLX/issues/160)

This page records the v0.2 public RF-DiT hosted-artifact status that runtime users may rely on. It does not upload weights, create model repositories, or approve any private/local artifact path. The machine-readable in-package contract is irodori_mlx.hosted_artifacts.HOSTED_RF_DIT_ARTIFACTS.

## Publication status

| Family | Upstream checkpoint | Public hosted repo | Status | Runtime use |
| --- | --- | --- | --- | --- |
| VoiceDesign v2 caption-conditioned | Aratako/Irodori-TTS-500M-v2-VoiceDesign | t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign at bf877a3beb7d921dc6bfb2b6812d02be07f39f2a | Approved public artifact | Use --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign after installing huggingface_hub. |
| v3 speaker-conditioned / duration predictor | Aratako/Irodori-TTS-500M-v3 | Not published in an approved public location by #157 | Blocked | Use local conversion fallback until an approved public repo id and immutable revision are published. |

Both families remain covered by the conditional redistribution audit in [preconverted_weights_redistribution_audit.md](preconverted_weights_redistribution_audit.md). VoiceDesign has a concrete public repo whose manifest reports license_review.status: "approved". v3 has audit eligibility, but this issue did not provide an approved public artifact location, so the public repo must not document a private cache path or pretend that --weights-repo can resolve v3 today.

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
      --text "こんにちは。今日は良い天気です。" \
      --caption "落ち着いた女性の声" \
      --no-reference \
      --output /tmp/irodori-hosted.wav \
      --preset balanced \
      --json

## v3 blocker and fallback

The v3 hosted artifact is intentionally marked blocked until there is an approved public repository and immutable revision. The documented fallback remains local conversion from the upstream checkpoint:

    CHECKPOINT=/path/to/Irodori-TTS-500M-v3/model.safetensors
    WORK=/tmp/irodori-v3-local

    irodori-tts-inspect "$CHECKPOINT" --json > "$WORK/checkpoint-inspect.json"
    irodori-tts-convert "$CHECKPOINT" "$WORK/weights.npz"
    irodori-tts-generate \
      --weights "$WORK/weights.npz" \
      --model-config-json "$WORK/model_config.json" \
      --text "こんにちは。今日は良い天気です。" \
      --no-reference \
      --output "$WORK/irodori-v3.wav" \
      --preset balanced

Do not replace the blocked status with a local filesystem path, a private Hugging Face repo, or a mutable staging branch. Publication requires an approved public repo id, immutable revision, matching checksums, README/model-card provenance, and license_review.status: "approved" in the hosted manifest.
