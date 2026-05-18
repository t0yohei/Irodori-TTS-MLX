# Irodori-TTS-MLX

[日本語 README](README.ja.md)

An unofficial Apple Silicon-focused MLX inference prototype for [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

> [!IMPORTANT]
> This repository is still an alpha inference prototype. The current runtime can generate WAV files with MLX RF-DiT weights and a DACVAE codec boundary, but it is CLI-first and does not promise a stable public Python API yet. This repository does **not** redistribute upstream code, checkpoints, Semantic-DACVAE weights, tokenizer assets, converted `.npz` archives, reference audio, generated audio, or Hugging Face cache snapshots.

## Current Scope

The implemented default path is:

> MLX text/caption conditioning + MLX RF-DiT sampling + hosted MLX DACVAE codec artifact by default

The project currently supports:

- inspecting local or Hugging Face Irodori-TTS `.safetensors` checkpoints
- converting supported checkpoints to MLX-friendly `.npz` RF-DiT weights
- loading direct local `.npz` weights, local hosted-layout directories/archives, or approved Hugging Face hosted-layout repositories
- adapting unquantized `mlx-audio` Irodori artifact directories into this project's hosted weights layout
- generating WAV files through `irodori-tts-generate` / `scripts/generate_wav.py`
- starting an optional local Gradio Web UI through `irodori-tts-web`
- using `--config-json`, `--requests-json`, `--cleanup-between-requests`, `--preset ultra-fast|fast|balanced|quality`, JSON metadata output, and runtime reuse for repeated local generation
- running local benchmarks and hosted Apple Silicon validation workflows

The public runtime default uses the approved hosted DACVAE codec artifact and does not require upstream `irodori_tts.codec.DACVAECodec`, `torch`, or `torchaudio`. The old PyTorch DACVAE bridge fallback modes have been removed from the generation runtime; public generation now uses `--codec-runtime-mode mlx` with hosted or local codec artifacts.

## Public API Stability

During the alpha phase, the only stable-ish user contract is the installed CLI:
`irodori-tts-generate`, `irodori-tts-convert`, `irodori-tts-convert-dacvae-codec`,
`irodori-tts-convert-dacvae-decoder`, `irodori-tts-inspect`, and
`irodori-tts-adapt-mlx-audio`, plus the documented artifact layouts, manifests,
metadata, and JSON outputs those commands use.

The `irodori_mlx` package, its top-level exports, and `scripts.*` modules are
internal implementation surfaces for the CLI, tests, and repository development.
They are importable, but they are not supported as a stable public Python API
yet and may change without deprecation in alpha releases. See
[docs/public_api_stability.md](docs/public_api_stability.md) for the full
boundary.

## Current Support Matrix

| Surface | Status | Public support boundary |
| --- | --- | --- |
| Project maturity | Alpha | CLI-first inference prototype. Console commands and documented artifact layouts are the supported interface; the Python module layout is not a stable public API. |
| VoiceDesign v2 hosted RF-DiT artifact | Supported | `--weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign` is approved for the documented no-reference caption quickstart. |
| v3 hosted RF-DiT artifact | Supported | `--weights-repo t0yohei/Irodori-TTS-MLX-500M-v3` is approved for the documented no-reference predicted-duration quickstart. |
| Base v2 speaker-conditioned generation | Experimental | Inspection and conversion are supported; generation is a manual reference-audio path and requires user-supplied audio that the user has rights to use. |
| Standalone MLX DACVAE codec artifact path | Supported default | The normal public runtime uses approved hosted/local codec artifacts with `--codec-runtime-mode mlx`, so it does not require upstream `irodori_tts.codec.DACVAECodec`, `torch`, or `torchaudio`. |
| PyTorch bridge-backed DACVAE codec path | Removed | `persistent`, `subprocess`, and `mlx-decode` are no longer public generation runtime modes. |
| MLX DACVAE decode for no-reference generation | Supported | Approved codec artifacts keep decode off upstream/PyTorch for no-reference v3 and VoiceDesign runs. |
| Fully MLX DACVAE encode/decode for reference audio | Experimental | Requires an executable local/hosted codec artifact with both encoder and decoder tensors; reference-audio speaker fidelity is still a maturing validation surface. |
| Local Web UI | Optional | `irodori-tts-web` is a local Gradio wrapper over the generation CLI for manual runs. It is not a hosted demo or a stable public Python API boundary. |
| Hosted artifacts outside the approved layouts | Blocked | Repositories without the documented manifest, checksum, provenance, and approved license review are not public support. Use local conversion instead. |
| Unsupported upstream product features | Non-goal | Training, LoRA fine-tuning, hosted demo operation, watermark guarantees, arbitrary checkpoint compatibility, and stable public Python API guarantees are intentionally outside this prototype. |

Paths such as `/path/to/...` and `/tmp/...` in examples are placeholders for user-managed files. They are not references to private caches, local maintainer machines, or unpublished public artifacts.

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
python -m pip install -e ".[runtime]"
```

For package users installing a release artifact instead of contributing from a checkout, install the built wheel or sdist in a clean environment and use the installed `irodori-tts-*` console scripts. Contributor editable installs, build validation, and the v0.3 alpha release-artifact checklist live in [docs/packaging.md](docs/packaging.md).

Install extra tooling only for the workflows that need it:

```bash
# benchmark helpers:
python -m pip install -e ".[bench]"
```

On Python 3.11, the `runtime`, `bench`, and `dev` extras pin
`sentencepiece>=0.1.99,<0.2` to stay compatible with the tokenizer ecosystem used by
the audited artifacts. On Python 3.12 and newer, the extras keep
`sentencepiece>=0.2,<1` for wheel availability.

The default standalone MLX runtime path uses the approved hosted codec artifact. For reproducible setup details, see [docs/packaging.md](docs/packaging.md) and [docs/upstream_dependency.md](docs/upstream_dependency.md).

## Optional Local Web UI

Install the optional Web UI dependencies in the same environment as the runtime:

```bash
python -m pip install -e ".[runtime,web]"
```

Start the local UI:

```bash
irodori-tts-web --host 127.0.0.1 --port 7860 --inbrowser
```

The Web UI is a local Gradio wrapper over the existing `irodori-tts-generate`
command. It provides presets for the approved VoiceDesign and v3 hosted
artifacts, plus fields for local weights, codec artifact inputs, reference audio,
caption text, sampling controls, generated audio, metadata, and logs.

Do not use reference audio unless you have the right to use that audio. The UI
does not redistribute checkpoints, codec weights, tokenizer assets, reference
audio, generated WAV files, or Hugging Face cache snapshots. The UI is an
optional local-use surface and does not make `irodori_mlx` or `scripts.*`
a stable public Python API.

## Quickstart: Hosted Weights

The shortest current CLI path is an approved hosted/pre-converted weights layout loaded with `--weights-repo` plus the default approved hosted DACVAE codec artifact. Use only repositories whose `irodori_mlx_manifest.json` has `license_review.status: "approved"` and whose README/model card records provenance for the exact upstream checkpoint revision.

Published RF-DiT artifact status is tracked in [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md). VoiceDesign and v3 currently have approved public hosted artifacts.

VoiceDesign example:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-hosted.wav \
  --preset balanced \
  --json
```

v3 hosted example:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-hosted-metadata.json
```

v3 local fallback smoke example:

```bash
irodori-tts-generate \
  --weights /path/to/converted-v3/weights.npz \
  --model-config-json /path/to/converted-v3/model_config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-metadata.json
```

By default, these examples use `--codec-runtime-mode mlx` and the approved hosted DACVAE codec artifact. You can pass `--codec-artifact-dir` or `--codec-path` to use a local approved/staged artifact instead.

Approved hosted DACVAE codec artifacts use a separate repo/layout from RF-DiT weights. The CLI defaults to this artifact for MLX codec modes, but you can pin it explicitly:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-runtime-mode mlx \
  --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
  --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted-codec.wav
```

The metadata for no-reference full-MLX runs reports `codec_decode_backend: "mlx"` and `codec_encode_backend: "not-required"`. Reference-audio generation with `mlx` uses executable Semantic-DACVAE encoder and decoder tensors from the codec artifact and reports both `codec_encode_backend: "mlx"` and `codec_decode_backend: "mlx"`. If no approved hosted repository is available, use the local conversion fallback below. See [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) for the full hosted/local layout flow, provenance checklist, `--weights-dir` / `--codec-artifact-dir` examples, and fallback decision rules.

### If the quickstart fails

Run a preflight first. It resolves the weights layout, model config, tokenizer repo names, codec runtime mode, and codec artifact path, but skips tokenizer loading, MLX weight loading, DACVAE bridge construction, and WAV generation:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --preflight \
  --json
```

Use the first failing surface to choose the next local action:

- tokenizer / Hugging Face cache: check network/cache access for the reported `text_tokenizer_repo` and, for VoiceDesign, `caption_tokenizer_repo`
- hosted RF-DiT weights: confirm `irodori_mlx_manifest.json`, `model_config.json`, `tokenizer_config.json`, `weights.npz`, `conversion_metadata.json`, `checksums.sha256`, and `license_review.status: "approved"`; otherwise use local conversion with `--weights`
- hosted DACVAE codec: confirm `irodori_dacvae_codec_manifest.json`, `dacvae-codec.npz`, `codec_metadata.json`, `checksums.sha256`, and `license_review.status: "approved"`; otherwise use a local `--codec-path` / `--codec-artifact-dir`

## Quickstart: Local Conversion Fallback

Use this path when a hosted repo is unavailable, unapproved, private, or outside the audited candidate families. Local conversion is a user-managed fallback; it does not make the input checkpoint or local paths part of this repository's public support surface.

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

For v3, omit `--seconds` to use predicted duration; add `--seconds N` only for a manual override. Very short prompts can sound repeated when the predicted duration is too long for the text. If the CLI prints a predicted-duration warning or playback repeats the final phrase, rerun with a shorter manual duration such as `--seconds 2.5`, or keep prediction but start with `--duration-scale 0.75`. The experimental `--preset ultra-fast` applies that short-prompt cap automatically when no manual duration controls are supplied. For base v2 speaker-conditioned checks, replace `--no-reference` with `--reference-wav /path/to/reference.wav`. Use only reference audio you have rights to use.

## mlx-audio Adapter

Do not pass `mlx-community/...` Irodori repos directly to `--weights-repo`; those artifacts use `config.json` + `model.safetensors`, not this project's `irodori_mlx_manifest.json` layout. Download or point at the local unquantized mlx-audio snapshot, adapt it, then load the emitted hosted layout:

```bash
irodori-tts-adapt-mlx-audio \
  /path/to/mlx-audio/Irodori-TTS-500M-v2-VoiceDesign-fp16 \
  /tmp/irodori-mlx-hosted-layout \
  --source-repo mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16

irodori-tts-generate \
  --weights-dir /tmp/irodori-mlx-hosted-layout \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-adapted.wav
```

The adapter currently supports unquantized v2/base and VoiceDesign layouts. Quantized mlx-audio artifacts are rejected until quantized runtime support is designed.

For VoiceDesign v2, omitting `--seconds` uses `duration_mode: "estimated"`: the runtime estimates duration primarily from `--text` and applies only small caption style-hint adjustments. Use `--seconds` for an exact duration or `--duration-scale` when a prompt clips or leaves too much tail.

For repeated local generation, put request objects in `--requests-json` to reuse one initialized runtime. Add `--cleanup-between-requests` when memory residency matters more than maximum throughput.

## Main Commands

```bash
irodori-tts-inspect /path/to/model.safetensors --all-tensors
irodori-tts-convert /path/to/model.safetensors /path/to/weights.npz
irodori-tts-convert /path/to/model.safetensors --dry-run --json
irodori-tts-generate --help
irodori-tts-generate --config-json config.json --requests-json requests.json
irodori-tts-web --help
irodori-tts-adapt-mlx-audio --help
python scripts/benchmark.py --self-test
```

Use the installed console scripts for normal workflows. Direct `python scripts/*.py` invocation is reserved for repository development and benchmark maintenance.

## Documentation Map

- Architecture: [docs/architecture.md](docs/architecture.md)
- DACVAE bridge and generation CLI: [docs/dacvae_bridge.md](docs/dacvae_bridge.md)
- Hosted/pre-converted MLX weights layout contract: [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md)
- Hosted weights/codec artifact usage and local conversion fallback: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- Hosted RF-DiT artifact publication status: [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md)
- Hosted DACVAE codec artifact publication status: [docs/hosted_dacvae_codec_artifacts.md](docs/hosted_dacvae_codec_artifacts.md)
- mlx-audio interop and adapter boundary: [docs/mlx_audio_interop.md](docs/mlx_audio_interop.md)
- DACVAE codec artifact layout and hosted repo contract: [docs/codec_artifact_layout.md](docs/codec_artifact_layout.md)
- Checkpoint support matrix: [docs/checkpoint_support.md](docs/checkpoint_support.md)
- VoiceDesign support: [docs/caption_condition_support.md](docs/caption_condition_support.md)
- v3 support: [docs/v3_support.md](docs/v3_support.md)
- Text preprocessing contract: [docs/text_preprocessing.md](docs/text_preprocessing.md)
- Weight mapping: [docs/weight_mapping.md](docs/weight_mapping.md)
- RF sampler: [docs/rf_sampler.md](docs/rf_sampler.md)
- Benchmarking: [docs/benchmark.md](docs/benchmark.md)
- Packaging: [docs/packaging.md](docs/packaging.md)
- Public API stability boundary: [docs/public_api_stability.md](docs/public_api_stability.md)
- License and distribution policy: [docs/license_and_distribution.md](docs/license_and_distribution.md)

## Non-goals

This prototype does not include:

- training
- LoRA or other fine-tuning workflows
- a bundled or fully redistributed Semantic-DACVAE codec
- guaranteed watermarking or watermark-detection behavior
- guaranteed compatibility with arbitrary third-party checkpoints
- hosted demo support
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
