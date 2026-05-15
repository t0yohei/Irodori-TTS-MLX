# License and distribution policy

Issue: [#77 Finalize license and distribution policy for v0.1](https://github.com/t0yohei/irodori-tts-mlx/issues/77)  
Parent: [#68 v0.1 documentation and release readiness](https://github.com/t0yohei/irodori-tts-mlx/issues/68)

## Repository code license

The source code and documentation in this repository are licensed under the [MIT License](../LICENSE), unless a file explicitly states otherwise.

This license covers this repository's own MLX implementation, scripts, tests, and documentation. It does not grant rights to upstream projects, model checkpoints, tokenizer assets, DACVAE weights, reference audio, generated audio, or local cache contents that are not committed here.

## Upstream dependency licenses

v0.1 intentionally depends on upstream projects and artifacts that keep their own licenses and terms:

| Dependency or artifact | Where to check terms | Notes |
| --- | --- | --- |
| Upstream Irodori-TTS code | [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) | Required for the PyTorch `irodori_tts.codec.DACVAECodec` bridge. |
| Irodori-TTS 500M v2 checkpoint | [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) | Users download this themselves when they need base v2 conversion or generation. |
| Irodori-TTS 500M v2 VoiceDesign checkpoint | [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) | Users download this themselves for the supported VoiceDesign v0.1 path. |
| Irodori-TTS 500M v3 checkpoint | [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | Users download this themselves for the supported v3 v0.1 path. |
| Semantic-DACVAE Japanese codec weights | [model card](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) | Needed by upstream DACVAE encode/decode; not redistributed here. |
| DACVAE upstream project | [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae) | Check separately if you use DACVAE code or weights directly. |

Always follow the upstream repository, model-card, dataset, and audio-rights terms for artifacts you download, convert, or use to generate outputs.

## Non-redistribution policy

This repository does **not** redistribute:

- Irodori-TTS checkpoint files such as `model.safetensors`;
- Semantic-DACVAE or other codec weights;
- Hugging Face cache directories or downloaded model snapshots;
- converted `.npz` archives derived from upstream checkpoints;
- reference audio supplied by users;
- generated WAV files or other audio artifacts.

Converted `.npz` files and generated audio can be derived artifacts from upstream models, DACVAE weights, prompts, and reference audio. Keep them local unless you have confirmed that every relevant upstream license and rights holder permits redistribution.

The v0.2 hosted/pre-converted MLX weights repository layout is documented in [hosted_weights_layout.md](hosted_weights_layout.md), but that layout is a packaging contract only. It does not approve publishing converted weights; any public hosted repository still needs an explicit license review and provenance record before upload.

## Contributor guidance

- Do not commit model weights, converted weights, cache directories, generated WAV files, or reference audio to this repository.
- Prefer documentation and validation scripts that tell users how to obtain artifacts from the original source instead of mirroring artifacts here.
- If a future release needs to distribute sample media or model-derived artifacts, add an explicit license review for that artifact before committing it.
