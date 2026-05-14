# v0.1 checkpoint support

This page is the v0.1 support contract for Irodori-TTS checkpoint families in this repository.

The support labels are intentionally conservative:

- **Supported**: implemented in checked-in code, documented, and covered by either lightweight tests or real-checkpoint manual/hosted validation.
- **Experimental**: implemented enough for local development, but not yet a v0.1 promise because hosted validation, representative manual evidence, or clear user workflow coverage is incomplete.
- **Unsupported**: not covered by the converter/runtime contract. A checkpoint may inspect as raw safetensors metadata, but users should not expect conversion or generation to work.

## v0.1 support matrix

| Checkpoint family | Example checkpoint | Inspection | Conversion | Generation | Validation status | v0.1 status |
| --- | --- | --- | --- | --- | --- | --- |
| Base v2 speaker-conditioned | `Aratako/Irodori-TTS-500M-v2` | Supported via `scripts/inspect_checkpoint.py` | Supported via `scripts/convert_weights.py`; detected as `base_v2` | Experimental via `scripts/generate_wav.py` with a reference WAV or explicit no-reference path | Manual Apple Silicon benchmark/generation notes exist, but there is no dedicated hosted v0.1 generation gate for this family yet | **Experimental** |
| VoiceDesign v2 caption-conditioned | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | Supported | Supported; detected as `voicedesign` | Supported for the inspected public VoiceDesign family through `--caption`, caption CFG, and optional `--no-reference` | Hosted Apple Silicon real-checkpoint inspect/conversion and full `generate_wav.py --caption ...` generation workflows exist | **Supported** |
| v3 speaker-conditioned / duration-predictor | `Aratako/Irodori-TTS-500M-v3` | Supported | Supported; detected as `v3` | Supported through the MLX bridge runtime. Omit `--seconds` to use predicted duration; manual `--seconds` remains an override | Hosted Apple Silicon generation workflow downloads, converts, runs generation, and asserts predicted-duration metadata | **Supported** |
| Other historical, fine-tuned, LoRA, architecture-modified, or renamed Irodori-TTS checkpoints | Any checkpoint whose tensor layout/config does not match one of the families above | Best-effort metadata inspection only if the file is a readable `.safetensors` checkpoint | Unsupported | Unsupported | No compatibility guarantee | **Unsupported** |

## Family boundaries

### Base v2 speaker-conditioned

Base v2 means the public `Aratako/Irodori-TTS-500M-v2` tensor layout: text encoder, speaker/reference encoder, RF-DiT blocks, and no v3 duration predictor. The converter validates the expected family-specific key set and float32 tensor assumptions before writing `.npz` output.

For v0.1, base v2 is **experimental** rather than fully supported. The code path is implemented and has manual benchmark/generation evidence, but v0.1 should not imply the same hosted release-gate coverage that exists for VoiceDesign and v3.

### VoiceDesign v2 caption-conditioned

VoiceDesign support is scoped to the inspected public `Aratako/Irodori-TTS-500M-v2-VoiceDesign` family. It replaces the speaker/reference encoder path with caption conditioning and is supported through inspection, conversion, runtime model loading, `generate_wav.py --caption ...`, and hosted Apple Silicon validation.

This does **not** promise compatibility with every caption-conditioned or VoiceDesign-like checkpoint. If a checkpoint changes tensor names, dimensions, tokenizer metadata, caption architecture, or DACVAE assumptions, treat it as unsupported until a separate inspection/conversion contract is added.

### v3 speaker-conditioned / duration-predictor

v3 support is scoped to the public `Aratako/Irodori-TTS-500M-v3` family. It keeps speaker/reference conditioning and adds the token-sum duration predictor. For generation, omitting `--seconds` exercises predicted-duration semantics; passing `--seconds` is an explicit manual override.

### Unsupported and best-effort families

For unsupported families, documentation should say exactly that: inspection may help users understand the checkpoint, but conversion and generation are outside the v0.1 contract. Do not describe unsupported families as "should work", "probably compatible", or "drop-in" without adding converter/runtime validation first.

## Redistribution and licensing caveats

This repository does **not** redistribute Irodori-TTS checkpoints, Semantic-DACVAE weights, Hugging Face cache contents, converted `.npz` archives, or generated audio artifacts. Users must obtain checkpoints from their original upstream sources and follow the upstream repository/model-card terms.

Converted weights are derived artifacts from upstream checkpoints. Do not commit them to this repository unless the project license and upstream model terms explicitly allow it.

## Related docs

- [`docs/weight_mapping.md`](weight_mapping.md): observed tensor layouts and family differences.
- [`docs/caption_condition_support.md`](caption_condition_support.md): VoiceDesign-specific support evidence.
- [`docs/v3_support.md`](v3_support.md): v3-specific support evidence and predicted-duration behavior.
- [`docs/v0_1_release_gate.md`](v0_1_release_gate.md): required v0.1 fresh-environment WAV-generation release gate and optional heavier validation.
- [`docs/dacvae_bridge.md`](dacvae_bridge.md): generation CLI and runtime boundary notes.
