# Irodori-TTS-MLX

[日本語 README](README.ja.md)

An unofficial Apple Silicon-focused MLX inference prototype for [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

> [!IMPORTANT]
> This repository is still an alpha inference prototype. The current runtime can generate WAV files with MLX RF-DiT weights and a DACVAE codec boundary, but it is CLI-first and does not promise a stable public Python API yet. This repository does **not** redistribute upstream code, checkpoints, Semantic-DACVAE weights, tokenizer assets, converted `.npz` archives, reference audio, generated audio, or Hugging Face cache snapshots.

## Current Scope

The implemented path is:

> MLX text/caption conditioning + MLX RF-DiT sampling + upstream PyTorch DACVAE bridge by default

The project currently supports:

- inspecting local or Hugging Face Irodori-TTS `.safetensors` checkpoints
- converting supported checkpoints to MLX-friendly `.npz` RF-DiT weights
- loading direct local `.npz` weights, local hosted-layout directories/archives, or approved Hugging Face hosted-layout repositories
- adapting unquantized `mlx-audio` Irodori artifact directories into this project's hosted weights layout
- generating WAV files through `irodori-tts-generate` / `scripts/generate_wav.py`
- using `--config-json`, `--requests-json`, `--preset fast|balanced|quality`, JSON metadata output, and persistent runtime reuse for repeated local generation
- running local benchmarks, parity checks, and hosted Apple Silicon validation workflows

The default audio codec path still imports upstream `irodori_tts.codec.DACVAECodec`. Experimental local MLX codec artifact modes exist for v0.2 codec-port work, but this repository does not bundle codec weights or claim broad acoustic parity for arbitrary codec artifacts.

## Supported Inputs

Supported checkpoint families are limited to the layouts explicitly validated in this repository:

| Checkpoint family | Example | Inspect | Convert | Generate |
| --- | --- | --- | --- | --- |
| Base v2 speaker-conditioned | `Aratako/Irodori-TTS-500M-v2` | Supported | Supported | Experimental manual path with reference audio |
| VoiceDesign v2 caption-conditioned | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | Supported | Supported | Supported with `--caption` and `--no-reference` |
| v3 speaker-conditioned / duration predictor | `Aratako/Irodori-TTS-500M-v3` | Supported | Supported | Supported; omit `--seconds` for predicted duration |

Unsupported means outside the current conversion/runtime contract, not merely untested. Historical, third-party, fine-tuned, quantized, LoRA, renamed, or architecture-modified checkpoints remain local-conversion-only until separately audited.

Python **3.11 through 3.14** are packaging targets. Python 3.11 remains the benchmark reference environment used by the project docs.

## Install

```bash
git clone https://github.com/t0yohei/Irodori-TTS-MLX.git
cd Irodori-TTS-MLX

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime,bench]"
```

Install upstream Irodori-TTS in the same environment, or make an existing checkout importable:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git ../Irodori-TTS
python -m pip install -e ../Irodori-TTS

# Same command shape when the upstream checkout is elsewhere:
# python -m pip install -e /path/to/Irodori-TTS

# Alternative for an uninstalled checkout:
# export PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}
```

The bridge runtime needs upstream `irodori_tts.codec.DACVAECodec` for the default `persistent` and `subprocess` codec modes. For reproducible setup details, see [docs/packaging.md](docs/packaging.md) and [docs/upstream_dependency.md](docs/upstream_dependency.md).

## Quickstart: Hosted Weights

The shortest current CLI path is an approved hosted/pre-converted weights layout loaded with `--weights-repo` or the same layout from disk with `--weights-dir`. Use only repositories whose `irodori_mlx_manifest.json` has `license_review.status: "approved"` and whose README/model card records provenance for the exact upstream checkpoint revision.

Published RF-DiT artifact status is tracked in [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md). VoiceDesign currently has an approved public hosted artifact; v3 remains on the local conversion fallback until an approved public repo and immutable revision are published.

VoiceDesign example:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-hosted.wav \
  --preset balanced \
  --json
```

v3 local fallback smoke example:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-metadata.json
```

This path still uses the upstream PyTorch DACVAE bridge unless you explicitly choose a local MLX codec artifact mode. For no-reference v3 and VoiceDesign runs, add a converted decode-capable DACVAE artifact to keep the generation path off PyTorch for codec decode and avoid reference encode entirely:

```bash
irodori-tts-generate \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-mlx-decode.wav \
  --preset balanced \
  --codec-runtime-mode mlx-decode \
  --codec-path /path/to/dacvae-codec.npz \
  --metadata-json /tmp/irodori-v3-mlx-decode-metadata.json
```

The metadata for that no-reference path reports `codec_decode_backend: "mlx"` and `codec_encode_backend: "not-required"`. Reference-audio generation with `mlx-decode` still needs the documented PyTorch encode fallback until an encode-capable MLX codec artifact is available. If no approved hosted repository is available, use the local conversion fallback below. See [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) for the full hosted/local layout flow, provenance checklist, `--weights-dir` examples, and fallback decision rules.

## Quickstart: Local Conversion Fallback

Use this path when a hosted repo is unavailable, unapproved, private, or outside the audited candidate families.

```bash
CHECKPOINT=/path/to/Irodori-TTS-500M-v3/model.safetensors
WORK=/tmp/irodori-quickstart
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

irodori-tts-convert "$CHECKPOINT" "$WORK/weights.npz" --dry-run --json \
  > "$WORK/convert-dry-run.json"
irodori-tts-convert "$CHECKPOINT" "$WORK/weights.npz"

PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output "$WORK/irodori.wav" \
  --preset balanced \
  --metadata-json "$WORK/metadata.json" \
  --json
```

For v3, omit `--seconds` to use predicted duration; add `--seconds N` only for a manual override. For base v2 speaker-conditioned checks, replace `--no-reference` with `--reference-wav /path/to/reference.wav`. Use only reference audio you have rights to use.

## mlx-audio Adapter

Do not pass `mlx-community/...` Irodori repos directly to `--weights-repo`; those artifacts use `config.json` + `model.safetensors`, not this project's `irodori_mlx_manifest.json` layout. Download or point at the local unquantized mlx-audio snapshot, adapt it, then load the emitted hosted layout:

```bash
irodori-tts-adapt-mlx-audio \
  /path/to/mlx-audio/Irodori-TTS-500M-v2-VoiceDesign-fp16 \
  /tmp/irodori-mlx-hosted-layout \
  --source-repo mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16

PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-dir /tmp/irodori-mlx-hosted-layout \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-adapted.wav
```

The adapter currently supports unquantized v2/base and VoiceDesign layouts. Quantized mlx-audio artifacts are rejected until quantized runtime support is designed.

## Main Commands

```bash
irodori-tts-inspect /path/to/model.safetensors --all-tensors
irodori-tts-convert /path/to/model.safetensors /path/to/weights.npz
irodori-tts-convert /path/to/model.safetensors --dry-run --json
irodori-tts-generate --help
irodori-tts-adapt-mlx-audio --help
python scripts/benchmark.py --self-test
```

Use the installed console scripts for normal workflows. Direct `python scripts/*.py` invocation is reserved for repository development and benchmark maintenance.

## Documentation Map

- Architecture: [docs/architecture.md](docs/architecture.md)
- DACVAE bridge and generation CLI: [docs/dacvae_bridge.md](docs/dacvae_bridge.md)
- v0.2 hosted/pre-converted MLX weights layout contract: [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md)
- Hosted weights usage and local conversion fallback: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- Hosted RF-DiT artifact publication status: [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md)
- mlx-audio interop and adapter boundary: [docs/mlx_audio_interop.md](docs/mlx_audio_interop.md)
- DACVAE artifact layout: [docs/codec_artifact_layout.md](docs/codec_artifact_layout.md)
- Checkpoint support matrix: [docs/checkpoint_support.md](docs/checkpoint_support.md)
- VoiceDesign support: [docs/caption_condition_support.md](docs/caption_condition_support.md)
- v3 support: [docs/v3_support.md](docs/v3_support.md)
- Text preprocessing contract: [docs/text_preprocessing.md](docs/text_preprocessing.md)
- Weight mapping: [docs/weight_mapping.md](docs/weight_mapping.md)
- RF sampler: [docs/rf_sampler.md](docs/rf_sampler.md)
- Benchmarking: [docs/benchmark.md](docs/benchmark.md)
- Packaging: [docs/packaging.md](docs/packaging.md)
- License and distribution policy: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- v0.2 cross-repository delivery plan and downstream consumer handoff boundary: [docs/v0_2_delivery_plan.md](docs/v0_2_delivery_plan.md)

## Non-goals

This prototype does not include:

- training or fine-tuning
- a bundled or fully redistributed Semantic-DACVAE codec
- guaranteed compatibility with arbitrary third-party checkpoints
- GUI, Gradio, or hosted demo support
- stable public Python API guarantees
- automatic legal approval for publishing converted weights or generated audio

## License Notes

This repository's own source code and documentation are licensed under the [MIT License](LICENSE), unless a file explicitly states otherwise.

The MIT License does not cover upstream code, checkpoint files, DACVAE weights, tokenizer assets, reference audio, converted `.npz` archives, generated audio, or other artifacts that are not redistributed here. Users must obtain upstream artifacts themselves and follow the relevant upstream repository/model-card terms. See [docs/license_and_distribution.md](docs/license_and_distribution.md) and [docs/preconverted_weights_redistribution_audit.md](docs/preconverted_weights_redistribution_audit.md).

## Related Resources

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
- DACVAE: <https://github.com/facebookresearch/dacvae>
