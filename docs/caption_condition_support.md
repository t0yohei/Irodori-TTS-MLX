# VoiceDesign / caption-conditioned support audit

Issue #33 asks for a clear support statement for `Aratako/Irodori-TTS-500M-v2-VoiceDesign` and related caption-conditioned checkpoints. This document records the current status of the codebase, the tested subset, and the next concrete expansion step.

## Executive summary

The repository now contains **working caption-conditioned conversion support** for the inspected VoiceDesign checkpoint family, alongside the existing MLX model/runtime support.

In practice, that means:

- metadata inspection for VoiceDesign checkpoints works
- `ModelConfig`, encoder wiring, RF-DiT attention, and runtime caption tokenization already understand caption-conditioned configs
- the WAV-generation runtime can run a caption-conditioned path **if compatible converted MLX weights already exist**
- the checked-in weight converter now detects the VoiceDesign family and can export caption-conditioned `.npz` archives when the checkpoint matches the documented layout
- the checked-in inspection/conversion tooling also recognizes the public `Aratako/Irodori-TTS-500M-v3` schema intentionally; that v3 support is tracked separately in [v3_support.md](v3_support.md)

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
| `scripts/convert_weights.py` | supported for the inspected family | Detects base-v2 vs VoiceDesign for caption-conditioned checkpoints, validates family-specific tensor/config layouts, exports caption-conditioned `.npz` archives, and also recognizes the public `Aratako/Irodori-TTS-500M-v3` schema intentionally. |
| Reproducible end-to-end VoiceDesign smoke test | supported for the inspected family | The manual inspect → convert → `generate_wav.py --caption ...` path is documented, the lightweight hosted workflow exercises inspect + converter validation, and a separate GitHub-hosted Apple Silicon workflow now runs the full generation path. |

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

### 1. Hosted full generation depends on standard Apple Silicon runner availability

The repository now has a GitHub-hosted Apple Silicon workflow for full VoiceDesign generation, and it uses the standard `macos-14` M1 runner. That means public-repository coverage no longer requires self-hosted infrastructure or paid larger runners, though it still depends on GitHub-hosted queueing and the smaller standard-runner resource envelope.

### 2. Manual end-to-end VoiceDesign recipe still needs a real checkpoint

The docs can now describe a supported sequence for:

1. inspecting a VoiceDesign checkpoint
2. converting it to MLX `.npz`
3. generating audio from that converted archive
4. benchmarking or validating parity on that path

### 3. Hosted coverage stays tightly scoped to the inspected family

Current tests and workflows now cover both lightweight converter-family validation and a real-checkpoint hosted generation path. They still do not claim broad support for caption-conditioned checkpoints beyond the inspected VoiceDesign layout.

### 4. V3 support is documented separately

This document is only about the inspected VoiceDesign / caption-conditioned family. The public `Aratako/Irodori-TTS-500M-v3` support statement, validation helper, and runtime caveats live in [v3_support.md](v3_support.md).

## Recommended next implementation step

The next concrete functional expansion should be:

1. monitor the hosted Apple Silicon workflow runtime and artifact usefulness over a few runs
2. decide whether broader caption-conditioned families beyond the inspected VoiceDesign layout should be supported explicitly
3. decide whether any of the hosted-generation setup should be shared with future benchmark or parity workflows

The remaining work is now mostly operational and scope-management oriented, because the core conversion/model/runtime path and hosted end-to-end coverage both exist.

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

For automated regression coverage, the repository now includes two workflows:

- `.github/workflows/voicedesign-real-checkpoint.yml` downloads the public VoiceDesign checkpoint on a schedule or manual dispatch and runs `scripts/run_voicedesign_integration.py` for inspect + converter validation.
- `.github/workflows/voicedesign-hosted-generation.yml` runs `scripts/run_voicedesign_generation_ci.py` on the standard GitHub-hosted Apple Silicon `macos-14` runner and executes the full `generate_wav.py --caption ...` path.

## Current user-facing support statement

> VoiceDesign / caption-conditioned checkpoints are supported for the inspected `Aratako/Irodori-TTS-500M-v2-VoiceDesign` family through conversion, MLX weight loading, and runtime generation. The repository now has real checkpoint-backed hosted automation for both inspect + converter validation and the full `generate_wav.py --caption ...` path on the standard GitHub-hosted `macos-14` Apple Silicon runner.
