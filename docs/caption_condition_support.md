# VoiceDesign / caption-conditioned support audit

Issue #33 asks for a clear support statement for `Aratako/Irodori-TTS-500M-v2-VoiceDesign` and related caption-conditioned checkpoints. This document records the current status of the codebase, the tested subset, and the next concrete expansion step.

## Executive summary

The repository already contains **partial caption-conditioned support** in the MLX model/runtime stack, but **does not yet support end-to-end VoiceDesign checkpoint conversion**.

In practice, that means:

- metadata inspection for VoiceDesign checkpoints works
- `ModelConfig`, encoder wiring, RF-DiT attention, and runtime caption tokenization already understand caption-conditioned configs
- the WAV-generation runtime can run a caption-conditioned path **if compatible converted MLX weights already exist**
- the checked-in weight converter still rejects VoiceDesign checkpoints on purpose

So the current support statement is:

> Caption-conditioned inference code paths are partially implemented and test-covered, but VoiceDesign is not yet a turnkey workflow because checkpoint conversion remains base-v2-only.

## Support matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `scripts/inspect_checkpoint.py` | supported | Reads VoiceDesign metadata/config and tensor headers without loading payloads. |
| `irodori_mlx.config.ModelConfig` | supported | Resolves caption tokenizer/dim/layer defaults for `use_caption_condition=true`. |
| `irodori_mlx.encoders.ConditionEncoders` | supported | Builds caption encoder/norm path and optional caption masks. |
| `irodori_mlx.model.TextToLatentRFDiT` | supported | RF-DiT blocks include `wk_caption` / `wv_caption` projections when caption conditioning is enabled. |
| `irodori_mlx.runtime.MLXDACVAERuntime` | partially supported | Loads a caption tokenizer and can run without speaker reference audio because VoiceDesign disables the speaker path. |
| `scripts/generate_wav.py` | partially supported | Exposes `--caption`, caption tokenizer overrides, and caption guidance controls. |
| `scripts/convert_weights.py` | not supported | Explicitly validates for the base checkpoint layout and rejects `use_caption_condition=true`. |
| Reproducible end-to-end VoiceDesign smoke test | not supported | No converted VoiceDesign fixture/recipe is checked in yet. |

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

This means the main missing piece is not the RF-DiT module graph itself, but the converter path that produces the `.npz` archive.

## Known gaps

### 1. Weight conversion is still base-v2-only

`scripts/convert_weights.py` intentionally validates against the base speaker-conditioned checkpoint layout.

Current blockers:

- `validate_base_config()` rejects `use_caption_condition=true`
- caption-only keys are treated as unsupported
- converter output assumptions still target the base speaker-conditioned tensor set

This is the main reason VoiceDesign is not yet an end-to-end supported workflow.

### 2. No repository-level end-to-end VoiceDesign recipe

The docs mention optional VoiceDesign baselines, but there is not yet a fully supported sequence for:

1. inspecting a VoiceDesign checkpoint
2. converting it to MLX `.npz`
3. generating audio from that converted archive
4. benchmarking or validating parity on that path

### 3. No real checkpoint-backed integration fixture

Current tests cover the code paths with small fakes/mocks. They do not yet prove a real converted VoiceDesign checkpoint can run end to end.

## Recommended next implementation step

The next concrete functional expansion should be:

1. extend `scripts/convert_weights.py` to accept VoiceDesign checkpoint metadata and caption tensor names
2. emit `.npz` archives that match the existing caption-aware MLX module tree
3. add one checkpoint-backed smoke/integration test (or reproducible manual validation recipe) for `scripts/generate_wav.py --caption ...`

That next step is higher leverage than more runtime refactoring, because most of the runtime/model wiring already exists.

## Current user-facing support statement

Until converter support lands, the project should describe VoiceDesign support like this:

> VoiceDesign / caption-conditioned checkpoints are partially supported internally. The MLX config, model, and runtime paths understand caption conditioning, but the checked-in weight conversion workflow only supports the base `Aratako/Irodori-TTS-500M-v2` checkpoint today.
