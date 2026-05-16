# mlx-audio interoperability evaluation

Issue: [#110 Evaluate mlx-audio artifact and runtime interoperability](https://github.com/t0yohei/Irodori-TTS-MLX/issues/110)  
Parent: [#123 TOY-5: Irodori-TTS-MLX v0.2 cross-repo delivery](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

This report evaluates [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) as a reference implementation and interoperability target for Irodori-TTS-MLX v0.2 work. The checked source revision used for this evaluation was `Blaizzy/mlx-audio` main as of 2026-05-16, and the public hosted model metadata was read from Hugging Face on the same date.

## Relevant mlx-audio artifacts

mlx-audio now has a first-class `irodori_tts` model package under `mlx_audio/tts/models/irodori_tts`. Its README lists these Irodori artifacts:

| mlx-community repo | Conditioning | Artifact shape observed |
| --- | --- | --- |
| `mlx-community/Irodori-TTS-500M-v2-fp16` | reference audio | `config.json`, `model.safetensors`, `dacvae/config.json`, `dacvae/model.safetensors` |
| `mlx-community/Irodori-TTS-500M-v2-8bit` | reference audio | same files plus `quantization: {bits: 8, group_size: 64}` in `config.json` |
| `mlx-community/Irodori-TTS-500M-v2-4bit` | reference audio | same files plus 4-bit quantization metadata |
| `mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16` | caption / instruction text | same files; `dit.use_caption_condition: true` |
| `mlx-community/Irodori-TTS-500M-v2-VoiceDesign-8bit` | caption / instruction text | same files plus 8-bit quantization metadata |
| `mlx-community/Irodori-TTS-500M-v2-VoiceDesign-4bit` | caption / instruction text | same files plus 4-bit quantization metadata |

The mlx-audio model package README also mentions an older `mlx-community/Irodori-TTS-500M-fp16` v1 artifact. It is outside this repository's current supported checkpoint-family contract.

## Layout comparison

| Concern | Irodori-TTS-MLX v0.2 hosted layout | mlx-audio layout |
| --- | --- | --- |
| Main weights | `weights.npz` produced by `scripts/convert_weights.py` | `model.safetensors` loaded by `mlx_audio.tts.load(...)` |
| Loader source of truth | `irodori_mlx_manifest.json` | `config.json` with `model_type: "irodori_tts"` |
| Runtime config | flat `model_config.json` accepted by `irodori_mlx.config.ModelConfig` | nested `dit` and `sampler` objects accepted by mlx-audio's `ModelConfig` |
| Tokenizer metadata | explicit `tokenizer_config.json` contract | tokenizer repos are embedded under `dit.*_tokenizer_repo` |
| Provenance/license metadata | explicit `conversion_metadata.json` and `license_review` | model card metadata and repository files; no Irodori-TTS-MLX manifest |
| DACVAE | not bundled in hosted RF-DiT weights; runtime uses PyTorch bridge or future local MLX codec `.npz` | bundled under `dacvae/model.safetensors` and loaded as mlx-audio's MLX `DACVAE` |
| Quantized variants | not part of the current v0.2 hosted layout contract | 4-bit and 8-bit repos are published for v2 and VoiceDesign |

The layouts are intentionally not drop-in compatible. Passing a mlx-audio repo directly to Irodori-TTS-MLX `--weights-repo` would fail because there is no `irodori_mlx_manifest.json`, no `weights.npz`, and its nested `config.json` is not a flat `ModelConfig` payload. Conversely, passing an Irodori-TTS-MLX hosted layout to mlx-audio would not provide its expected `model.safetensors` + `config.json` pair.

## Runtime behavior comparison

mlx-audio is a useful runtime reference for these areas:

- It has an MLX DACVAE implementation and loads `Semantic-DACVAE-Japanese-32dim` either from a bundled `dacvae/` subdirectory or from `config.dacvae_repo`.
- Its Irodori runtime uses `sequence_length` rather than this repository's seconds/predicted-duration UX. This is an intentional UX difference, not a compatibility target for the current CLI.
- It supports `cfg_guidance_mode` values such as `independent` and `alternating`; this is worth tracking separately from the artifact format.
- It exposes VoiceDesign as `instruct`/`caption`, while this repository's CLI uses `--caption`.
- It currently covers v2 and VoiceDesign v2 artifacts; this repository also treats v3 predicted-duration checkpoints as first-class.

The supported interoperability target is not direct loader compatibility. Use the adapter/conversion path, then load the emitted hosted layout with `--weights-dir` or publish it as an approved hosted repo:

```text
mlx-audio model.safetensors + config.json
        |
        +-- compare/remap RF-DiT keys against irodori_mlx converter keys
        +-- translate nested config.dit fields into flat ModelConfig fields
        +-- reject quantization metadata until quantized MLX runtime support exists
        +-- emit Irodori-TTS-MLX hosted layout: manifest + model_config.json + tokenizer_config.json + conversion_metadata.json + weights.npz
```

The local adapter entry point is:

```bash
irodori-tts-adapt-mlx-audio /path/to/mlx-community/Irodori-TTS-500M-v2-fp16-snapshot /tmp/irodori-mlx-audio-hosted --source-repo mlx-community/Irodori-TTS-500M-v2-fp16 --source-revision <commit-sha>
```

The source directory must contain mlx-audio's `config.json` and `model.safetensors`. The adapter writes a normal Irodori-TTS-MLX hosted layout and validates it with `validate_weights_layout(...)`; use that output rather than passing `mlx-community/...` directly to `--weights-repo`.

The reverse direction is also possible for unquantized RF-DiT weights, but it is lower priority because mlx-audio already publishes its own artifacts and because Irodori-TTS-MLX's v0.2 delivery goal is downstream local-assistant/OpenClaw consumption, not acting as a general mlx-audio artifact publisher.

## DACVAE implications

mlx-audio's MLX DACVAE path is relevant to downstream DACVAE implementation work. The bundled codec layout is:

```text
dacvae/
+-- config.json
+-- model.safetensors
```

The observed v2 DACVAE config uses `sample_rate: 48000`, encoder rates `[2, 8, 10, 12]`, decoder rates `[12, 10, 8, 2]`, `n_codebooks: 16`, `codebook_size: 1024`, and `codebook_dim: 32`. That matches the Semantic-DACVAE Japanese 32-dim family this repository already names, but the runtime contract differs from the small test `.npz` codec contract documented in [dacvae_bridge.md](dacvae_bridge.md).

Recommendation: use mlx-audio's DACVAE implementation as an implementation reference for #106/#111-class work, but keep this repository's DACVAE artifact contract explicit. The DACVAE-specific comparison in [mlx_audio_dacvae_contract.md](mlx_audio_dacvae_contract.md) selects the current compatibility path: keep runtime loading on the local `dacvae-codec.npz` contract, treat `dacvae/config.json` + `dacvae/model.safetensors` as converter inputs only, and require fixed latent/audio parity fixtures before any converted mlx-audio codec artifact is treated as compatible.

## Recommended follow-up work

1. [#131](https://github.com/t0yohei/Irodori-TTS-MLX/issues/131): the adapter imports unquantized mlx-audio Irodori v2/VoiceDesign artifacts into the Irodori-TTS-MLX hosted layout; reject 4-bit/8-bit quantized repos until this runtime has an explicit quantization story. It does not add direct runtime loading for bundled `dacvae/config.json` + `dacvae/model.safetensors`; it preserves DACVAE provenance and leaves codec conversion to the existing hosted/local codec companion contract after the parity gate exists.
2. [#130](https://github.com/t0yohei/Irodori-TTS-MLX/issues/130): compare mlx-audio's MLX DACVAE output against the current PyTorch `DACVAECodec` bridge and the local `.npz` codec artifact contract.
3. Keep direct `--weights-repo mlx-community/...` loading out of scope for v0.2 unless a manifest sidecar or adapter layer is added. Silent fallback from mlx-audio's `config.json` shape would be too easy to misconfigure.

## Smoke evidence

The contract test `tests/test_mlx_audio_interop_doc.py` captures the evaluated compatibility path without downloading multi-GiB model payloads:

- it asserts the documented mlx-audio repo set and artifact file names;
- it proves a mlx-audio-like directory is rejected by `validate_weights_layout` because the Irodori-TTS-MLX manifest is missing;
- it proves mlx-audio's nested `config.json` shape is rejected when used as `model_config.json` in an otherwise valid hosted layout;
- `tests/test_mlx_audio_adapter.py` proves a mlx-audio-like unquantized layout can be adapted into a valid `validate_weights_layout(...)` output and that quantized metadata is rejected with a targeted error;
- it documents the adapter boundary rather than treating direct loader compatibility as supported.
