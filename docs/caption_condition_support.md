# VoiceDesign / caption-conditioned support audit

Issue #33 asks for a clear support statement for `Aratako/Irodori-TTS-500M-v2-VoiceDesign` and related caption-conditioned checkpoints. This document records the current status of the codebase, the tested subset, and the next concrete expansion step.

## Executive summary

The repository now contains **working caption-conditioned conversion support** for the inspected VoiceDesign checkpoint family, alongside the existing MLX model/runtime support.

In practice, that means:

- metadata inspection for VoiceDesign checkpoints works
- `ModelConfig`, encoder wiring, RF-DiT attention, and runtime caption tokenization already understand caption-conditioned configs
- the WAV-generation runtime can run a caption-conditioned path **if compatible converted MLX weights already exist**
- the checked-in weight converter now detects the VoiceDesign family and can export caption-conditioned `.npz` archives when the checkpoint matches the documented layout

So the current support statement is:

> Caption-conditioned inference code paths are implemented through conversion, weight loading, and runtime usage for the inspected VoiceDesign family, with tests covering the main converter decision points.

## Support matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `scripts/inspect_checkpoint.py` | supported | Reads VoiceDesign metadata/config and tensor headers without loading payloads. |
| `irodori_mlx.config.ModelConfig` | supported | Resolves caption tokenizer/dim/layer defaults for `use_caption_condition=true`. |
| `irodori_mlx.encoders.ConditionEncoders` | supported | Builds caption encoder/norm path and optional caption masks. |
| `irodori_mlx.model.TextToLatentRFDiT` | supported | RF-DiT blocks include `wk_caption` / `wv_caption` projections when caption conditioning is enabled. |
| `irodori_mlx.runtime.MLXDACVAERuntime` | partially supported | Loads a caption tokenizer and can run without speaker reference audio because VoiceDesign disables the speaker path. |
| `scripts/generate_wav.py` | partially supported | Exposes `--caption`, caption tokenizer overrides, and caption guidance controls. |
| `scripts/convert_weights.py` | supported for the inspected family | Detects base-v2 vs VoiceDesign, validates family-specific tensor/config layouts, and exports caption-conditioned `.npz` archives. |
| Reproducible end-to-end VoiceDesign smoke test | partially supported | The manual inspect → convert → `generate_wav.py --caption ...` path is documented, and a scheduled/manual real-checkpoint workflow now exercises inspect + converter validation. Full MLX generation still remains outside the standing hosted CI path. |

## What is already covered in code

### Runtime and config

The runtime already does the key caption-conditioned setup work:

- loads a dedicated caption tokenizer when `use_caption_condition=true`
- encodes caption text separately from prompt text
- turns blank caption text into an unconditional caption mask
- skips speaker/reference preparation when the checkpoint is caption-conditioned

These behaviors are now covered by unit tests so they do not regress silently.

### Weight loading assumptions

The MLX weight loader already knows the caption-specific tensor names once a compatible `.npz` exists:

- `caption_encoder.*`
- `caption_norm.weight`
- `blocks.{i}.attention.wk_caption.weight`
- `blocks.{i}.attention.wv_caption.weight`

This means the RF-DiT module graph and converter path now line up for the inspected VoiceDesign family.

## Known gaps

### 1. Hosted CI still does not run full MLX generation

The repository now has a real checkpoint-backed automated fixture for inspect + converter validation, but hosted CI still does not execute the full MLX generation path.

### 2. Manual end-to-end VoiceDesign recipe still needs a real checkpoint

The docs can now describe a supported sequence for:

1. inspecting a VoiceDesign checkpoint
2. converting it to MLX `.npz`
3. generating audio from that converted archive
4. benchmarking or validating parity on that path

### 3. No full checkpoint-backed generation fixture

Current tests now cover both lightweight converter-family validation and a real-checkpoint automation path for inspect + conversion checks. They still do not prove a real converted VoiceDesign checkpoint can run all the way through MLX generation in hosted CI.

## Recommended next implementation step

The next concrete functional expansion should be:

1. decide whether to add a self-hosted Apple Silicon workflow for full `generate_wav.py --caption ...` execution
2. capture a reproducible full-conversion example from the new real-checkpoint workflow artifacts when running `--full-conversion`
3. decide whether broader caption-conditioned families beyond the inspected VoiceDesign layout should be supported explicitly

That next step is now about deeper runtime coverage rather than converter wiring, because the core conversion/model/runtime path exists.

## Manual VoiceDesign workflow

A reproducible manual path now looks like this:

```bash
python3 scripts/inspect_checkpoint.py /path/to/voice-design/model.safetensors --json
python3 scripts/convert_weights.py /path/to/voice-design/model.safetensors /tmp/irodori-voicedesign.npz --dry-run --json
python3 scripts/convert_weights.py /path/to/voice-design/model.safetensors /tmp/irodori-voicedesign.npz
python3 scripts/generate_wav.py \
  --weights /tmp/irodori-voicedesign.npz \
  --model-config-json '{"use_caption_condition": true, "caption_vocab_size": 99574, "caption_tokenizer_repo": "llm-jp/llm-jp-3-150m", "caption_add_bos": true, "caption_dim": 512, "caption_layers": 10, "caption_heads": 8, "caption_mlp_ratio": 2.6}' \
  --text "こんにちは。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --output /tmp/irodori-voicedesign.wav
```

Check the `checkpoint_family` field in the dry-run JSON/text report before running the full conversion. A valid inspected VoiceDesign checkpoint should report `checkpoint_family: voicedesign`.

For automated regression coverage, the repository now also includes `.github/workflows/voicedesign-real-checkpoint.yml`, which downloads the public VoiceDesign checkpoint on a schedule or manual dispatch and runs `scripts/run_voicedesign_integration.py`.

## Current user-facing support statement

> VoiceDesign / caption-conditioned checkpoints are supported for the inspected `Aratako/Irodori-TTS-500M-v2-VoiceDesign` family through conversion, MLX weight loading, and runtime generation. The repository now has real checkpoint-backed automation for inspect + converter validation; the main remaining caveat is that hosted CI still does not execute the full MLX generation path.
