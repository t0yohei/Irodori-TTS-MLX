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
| VoiceDesign v2 caption-conditioned | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | Supported | Supported; detected as `voicedesign` | Supported for the inspected public VoiceDesign family through `--caption`, caption CFG, and optional `--no-ref` | Hosted Apple Silicon real-checkpoint inspect/conversion and full `generate_wav.py --caption ...` generation workflows exist | **Supported** |
| v3 speaker-conditioned / duration-predictor | `Aratako/Irodori-TTS-500M-v3` | Supported | Supported; detected as `v3` | Supported through the MLX bridge runtime. Omit `--seconds` to use predicted duration; manual `--seconds` remains an override | Hosted Apple Silicon generation workflow downloads, converts, runs generation, and asserts predicted-duration metadata | **Supported** |
| Upstream-merged LoRA export, layout-compatible | A LoRA fine-tune that upstream Irodori-TTS has merged/exported back into a normal inference `.safetensors` checkpoint matching Base v2, VoiceDesign v2, or v3 | Supported via `scripts/inspect_checkpoint.py`; `metadata.config_json` must identify one of the supported families | Experimental via `scripts/convert_weights.py` because conversion is accepted only when the merged export has the same tensor names, shapes, dtype class, and config contract as the detected family | Experimental through the same generation path as the detected family | Lightweight docs/test coverage only; users should run the manual recipe below with their local merged export | **Experimental** |
| Other historical, unmerged LoRA adapters, dynamic LoRA adapter loading, fine-tuned, architecture-modified, quantized, or renamed Irodori-TTS checkpoints | Any checkpoint or adapter whose tensor layout/config does not match one of the families above | Best-effort metadata inspection only if the file is a readable `.safetensors` checkpoint | Unsupported | Unsupported | No compatibility guarantee | **Unsupported** |

## Family boundaries

The generation CLI reports `checkpoint_family` and `checkpoint_capabilities` in `--json` / `--metadata-json` output and in the human-readable summary. Treat those fields as the first check when a run behaves differently than expected:

- `base_v2`: speaker/reference family. Use `--ref-wav` for normal generation, or `--no-ref` for the unconditional speaker path. `--caption` is rejected. Omit `--seconds` only if the fixed fallback duration is acceptable.
- `voicedesign`: VoiceDesign v2 caption family. Use `--caption` with `--no-ref`. `--ref-wav` is rejected because this family does not use speaker/reference conditioning in this runtime.
- `v3`: speaker/reference family with duration predictor. Use `--ref-wav` or `--no-ref`. Omit `--seconds` to use predicted duration, or pass `--seconds` for a manual override. `--caption` is rejected.

### Base v2 speaker-conditioned

Base v2 means the public `Aratako/Irodori-TTS-500M-v2` tensor layout: text encoder, speaker/reference encoder, RF-DiT blocks, and no v3 duration predictor. The converter validates the expected family-specific key set and float32 tensor assumptions before writing `.npz` output.

For v0.1, base v2 is **experimental** rather than fully supported. The code path is implemented and has manual benchmark/generation evidence, but v0.1 should not imply the same hosted release-gate coverage that exists for VoiceDesign and v3.

```bash
python scripts/generate_wav.py \
  --weights /path/to/base-v2.npz \
  --model-config-json /path/to/base-v2-model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --ref-wav /path/to/reference.wav \
  --seconds 5.0 \
  --output-wav /tmp/irodori-base-v2.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-base-v2-metadata.json
```

### VoiceDesign v2 caption-conditioned

VoiceDesign support is scoped to the inspected public `Aratako/Irodori-TTS-500M-v2-VoiceDesign` family. It replaces the speaker/reference encoder path with caption conditioning and is supported through inspection, conversion, runtime model loading, `generate_wav.py --caption ...`, and hosted Apple Silicon validation.

This does **not** promise compatibility with every caption-conditioned or VoiceDesign-like checkpoint. If a checkpoint changes tensor names, dimensions, tokenizer metadata, caption architecture, or DACVAE assumptions, treat it as unsupported until a separate inspection/conversion contract is added.

```bash
python scripts/generate_wav.py \
  --weights /path/to/voicedesign-v2.npz \
  --model-config-json /path/to/voicedesign-v2-model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-ref \
  --seconds 5.0 \
  --output-wav /tmp/irodori-voicedesign-v2.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-voicedesign-v2-metadata.json
```

### v3 speaker-conditioned / duration-predictor

v3 support is scoped to the public `Aratako/Irodori-TTS-500M-v3` family. It keeps speaker/reference conditioning and adds the token-sum duration predictor. For generation, omitting `--seconds` exercises predicted-duration semantics; passing `--seconds` is an explicit manual override.

```bash
python scripts/generate_wav.py \
  --weights /path/to/v3.npz \
  --model-config-json /path/to/v3-model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-ref \
  --output-wav /tmp/irodori-v3.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-metadata.json \
  --json
```

### Upstream-merged LoRA exports

Merged LoRA checkpoints are **experimental** and layout-bound. The supported path is not MLX-side LoRA training and not dynamic `--lora-adapter` inference. It is only the optimized inference workflow where upstream Irodori-TTS fine-tunes and validates a LoRA, merges/exports the adapter into a normal inference `.safetensors` checkpoint, and the merged export still looks exactly like one of the existing MLX-supported checkpoint families.

The converter/runtime do not need special LoRA handling when all of these are true:

- `scripts/inspect_checkpoint.py` can read the merged `.safetensors` header.
- `metadata.config_json` identifies Base v2, VoiceDesign v2, or v3 using the same config fields as the public family.
- `scripts/convert_weights.py --dry-run` accepts the tensor names, shapes, and float dtype class without missing or unexpected keys.
- Generation uses the same CLI requirements as the detected family: Base v2 and v3 use `--ref-wav` or `--no-ref`; VoiceDesign v2 uses `--caption` with `--no-ref`.

Manual validation recipe:

```bash
MERGED=/path/to/upstream-merged-lora/model.safetensors
WORK=/tmp/irodori-merged-lora
mkdir -p "$WORK"

python scripts/inspect_checkpoint.py "$MERGED" --json > "$WORK/checkpoint-inspect.json"
python - "$WORK/checkpoint-inspect.json" > "$WORK/model_config.json" <<'PY'
import json
import sys
from dataclasses import fields
from irodori_mlx.config import ModelConfig

payload = json.load(open(sys.argv[1]))
allowed = {field.name for field in fields(ModelConfig)}
print(json.dumps({k: v for k, v in payload["config"].items() if k in allowed}, ensure_ascii=False, indent=2, sort_keys=True))
PY

python scripts/convert_weights.py "$MERGED" "$WORK/weights.npz" --dry-run --json
python scripts/convert_weights.py "$MERGED" "$WORK/weights.npz"

# Base v2 or v3 merged exports:
python scripts/generate_wav.py \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-ref \
  --output-wav "$WORK/irodori-merged-lora.wav" \
  --metadata-json "$WORK/irodori-merged-lora-metadata.json"

# VoiceDesign v2 merged exports:
python scripts/generate_wav.py \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-ref \
  --output-wav "$WORK/irodori-merged-lora-voicedesign.wav" \
  --metadata-json "$WORK/irodori-merged-lora-voicedesign-metadata.json"
```

If `convert_weights.py --dry-run` reports missing keys, unexpected keys, shape mismatches, dtype mismatches, or config errors, the merged export is outside this path. Record the concrete converter error and open follow-up scope for either a new checkpoint-family contract or dynamic LoRA adapter inference. Do not describe that checkpoint as supported until the converter/runtime contract is extended and tested.

### Unsupported and best-effort families

For unsupported families, documentation should say exactly that: inspection may help users understand the checkpoint, but conversion and generation are outside the v0.1 contract. Do not describe unsupported families as "should work", "probably compatible", or "drop-in" without adding converter/runtime validation first.

## Redistribution and licensing caveats

This repository does **not** redistribute Irodori-TTS checkpoints, Semantic-DACVAE weights, Hugging Face cache contents, converted `.npz` archives, reference audio, or generated audio artifacts. Users must obtain checkpoints from their original upstream sources and follow the upstream repository/model-card terms.

Converted weights are derived artifacts from upstream checkpoints. Do not commit them to this repository unless the project license and upstream model terms explicitly allow it.

For hosted converted weights, the eligibility decision framework and audited candidate list live in [docs/preconverted_weights_redistribution_audit.md](preconverted_weights_redistribution_audit.md). Anything outside that audited list remains local-conversion-only until separately reviewed.

The repository's own source code and documentation are MIT-licensed, but that license does not extend to upstream or user-supplied artifacts. See [docs/license_and_distribution.md](license_and_distribution.md) for the v0.1 license and distribution policy.

## Related docs

- [`docs/weight_mapping.md`](weight_mapping.md): observed tensor layouts and family differences.
- [`docs/caption_condition_support.md`](caption_condition_support.md): VoiceDesign-specific support evidence.
- [`docs/v3_support.md`](v3_support.md): v3-specific support evidence and predicted-duration behavior.
- [`docs/dacvae_bridge.md`](dacvae_bridge.md): generation CLI and runtime boundary notes.
- [`docs/license_and_distribution.md`](license_and_distribution.md): repository license, upstream terms links, and non-redistribution rules.
