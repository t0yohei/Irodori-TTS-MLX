# Irodori-TTS-MLX

[日本語 README](README.ja.md)

An unofficial Apple Silicon-focused MLX inference prototype for [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

> [!IMPORTANT]
> This is an alpha, CLI-first inference prototype. It can generate WAV files with MLX RF-DiT weights and an MLX DACVAE codec artifact, but it does not provide a stable public Python API yet. This repository does **not** redistribute upstream code, checkpoints, Semantic-DACVAE weights, tokenizer assets, converted `.npz` archives, reference audio, generated audio, or Hugging Face cache snapshots.

## What Works Now

The default runtime path is:

> MLX text/caption conditioning + MLX RF-DiT sampling + hosted MLX DACVAE codec artifact

Current CLI support:

- WAV generation with approved hosted VoiceDesign v2 and v3 RF-DiT artifacts
- local inspection/conversion for supported Irodori-TTS `.safetensors` checkpoints
- local hosted-layout directories/archives and direct local `.npz` fallback
- unquantized `mlx-audio` Irodori artifact adaptation
- optional local Gradio UI with `irodori-tts-web`
- repeated local generation with `--requests-json`, metadata JSON, and cleanup controls

## Install

~~~bash
git clone https://github.com/t0yohei/Irodori-TTS-MLX.git
cd Irodori-TTS-MLX

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
~~~

Optional extras:

~~~bash
python -m pip install -e ".[runtime,web]"  # local Gradio UI
python -m pip install -e ".[bench]"        # benchmark helpers
~~~

Python **3.11 through 3.14** are packaging targets. Python 3.11 is the benchmark reference environment in the docs.

## Quickstart

VoiceDesign v2:

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-voicedesign.wav \
  --preset balanced \
  --json
~~~

v3:

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-metadata.json
~~~

The CLI automatically uses the approved hosted DACVAE codec artifact. Pin it explicitly when you need a reproducible codec revision:

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
  --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-pinned-codec.wav
~~~

For local or staged codec artifacts, use `--codec-artifact-dir` or `--codec-path`.

For v3, omit `--seconds` to use predicted duration. If a very short prompt repeats, try a manual duration such as `--seconds 2.5` or keep prediction and start with `--duration-scale 0.75`.

Upstream-validated low-step recipes that use Sway Sampling can be carried into the MLX runtime with `--t-schedule-mode sway --sway-coeff -1.0`. The default remains `--t-schedule-mode linear`; matching upstream's timestep schedule is useful for recipe parity, but exact audio parity can still differ because the MLX runtime uses separate codec artifacts and execution details.

Upstream-validated temporal and speaker quality recipes can also pass through `--rescale-k`, `--rescale-sigma`, `--speaker-kv-scale`, `--speaker-kv-min-t`, and `--speaker-kv-max-layers`. These controls are intended for carrying known upstream recipes into MLX inference; keep primary tuning and validation in upstream Irodori-TTS before relying on a recipe in this optimized runtime.

Speaker Inversion embeddings trained and validated with upstream Irodori-TTS can be reused for MLX inference with `--ref-embed`. The embedding must be a `.speaker.safetensors` file containing one speaker-state tensor with shape `(speaker_dim)`, `(sequence, speaker_dim)`, or `(1, sequence, speaker_dim)`. `--ref-embed` is mutually exclusive with `--reference-wav` and `--no-reference`; it bypasses DACVAE reference encoding and records `speaker_condition_source: "embedding"` in JSON metadata.

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --text "こんにちは。今日は良い天気です。" \
  --ref-embed /path/to/voice.speaker.safetensors \
  --output /tmp/irodori-v3-speaker.wav \
  --preset balanced \
  --json
~~~

## If It Fails

Run preflight first:

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --preflight \
  --json
~~~

Preflight resolves the weights layout, model config, tokenizer repos, codec runtime mode, and codec artifact path, then exits before tokenizer loading, MLX weight loading, DACVAE bridge construction, or WAV generation.

Common next steps:

- tokenizer/cache issue: check the reported `text_tokenizer_repo` and, for VoiceDesign, `caption_tokenizer_repo`
- hosted RF-DiT issue: check `irodori_mlx_manifest.json` and `license_review.status: "approved"`, or use local conversion
- hosted codec issue: check `irodori_dacvae_codec_manifest.json`, or use `--codec-path` / `--codec-artifact-dir`

## Other Workflows

Local Web UI:

~~~bash
irodori-tts-web --host 127.0.0.1 --port 7860 --inbrowser
~~~

local conversion fallback:

~~~bash
CHECKPOINT=/path/to/model.safetensors
WORK=/tmp/irodori
mkdir -p "$WORK"

irodori-tts-inspect "$CHECKPOINT" --json > "$WORK/checkpoint-inspect.json"
python - "$WORK/checkpoint-inspect.json" > "$WORK/model_config.json" <<'PY'
import json
import sys
from dataclasses import fields
from irodori_mlx.config import ModelConfig
payload = json.load(open(sys.argv[1]))
allowed = {field.name for field in fields(ModelConfig)}
print(json.dumps({k: v for k, v in payload["config"].items() if k in allowed}, ensure_ascii=False, indent=2, sort_keys=True))
PY
irodori-tts-convert "$CHECKPOINT" "$WORK/weights.npz"
irodori-tts-generate \
  --weights "$WORK/weights.npz" \
  --model-config-json "$WORK/model_config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output "$WORK/irodori-local.wav"
~~~

Direct local v3 hosted-layout smoke path:

~~~bash
irodori-tts-generate \
  --weights /path/to/converted-v3/weights.npz \
  --model-config-json /path/to/converted-v3/model_config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-local.wav
~~~

`mlx-audio` adaptation:

~~~bash
irodori-tts-adapt-mlx-audio \
  /path/to/mlx-audio/Irodori-TTS-500M-v2-VoiceDesign-fp16 \
  /tmp/irodori-mlx-hosted-layout \
  --source-repo mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16
~~~

Use `irodori-tts-generate --help` for the full CLI surface.

## Support Boundary

Stable-ish during alpha:

- installed console scripts: `irodori-tts-generate`, `irodori-tts-convert`, `irodori-tts-convert-dacvae-codec`, `irodori-tts-convert-dacvae-decoder`, `irodori-tts-inspect`, `irodori-tts-adapt-mlx-audio`, `irodori-tts-web`
- documented artifact layouts, manifests, metadata, and JSON outputs

Not stable yet:

- `irodori_mlx` imports, top-level exports, and `scripts.*` modules as a public Python API
- arbitrary third-party, unmerged LoRA adapters, dynamic LoRA adapter loading, quantized, renamed, or architecture-modified checkpoints
- hosted demos, training, fine-tuning, watermark guarantees, or automatic legal approval for generated/converted artifacts

## Documentation

- Usage details and artifact layout: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- Hosted/pre-converted MLX weights layout contract: [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md)
- DACVAE codec artifact layout contract: [docs/codec_artifact_layout.md](docs/codec_artifact_layout.md)
- RF-DiT hosted artifact status: [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md)
- DACVAE codec artifact status: [docs/hosted_dacvae_codec_artifacts.md](docs/hosted_dacvae_codec_artifacts.md)
- Checkpoint support: [docs/checkpoint_support.md](docs/checkpoint_support.md)
- VoiceDesign support: [docs/caption_condition_support.md](docs/caption_condition_support.md)
- v3 support: [docs/v3_support.md](docs/v3_support.md)
- Public API stability: [docs/public_api_stability.md](docs/public_api_stability.md)
- Packaging: [docs/packaging.md](docs/packaging.md)
- License and distribution: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- Full docs index: [docs/README.md](docs/README.md)

## License

This repository's own source code and documentation are licensed under the [MIT License](LICENSE), unless a file explicitly states otherwise.

The MIT License does not cover upstream code, checkpoint files, DACVAE weights, tokenizer assets, reference audio, converted `.npz` archives, generated audio, or other artifacts that are not redistributed here. Users must obtain upstream artifacts themselves and follow the relevant upstream repository/model-card terms.

## Related Resources

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
