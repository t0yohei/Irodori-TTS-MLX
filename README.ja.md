# Irodori-TTS-MLX

[English README](README.md)

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を Apple Silicon / MLX で動かすための非公式 inference prototype です。

> [!IMPORTANT]
> まだ alpha 段階の CLI-first prototype です。現行実装は MLX RF-DiT weights と MLX DACVAE codec artifact で WAV 生成できますが、安定した public Python API はまだ提供しません。このリポジトリは upstream code、checkpoint、Semantic-DACVAE weights、tokenizer assets、converted `.npz`、reference audio、generated audio、Hugging Face cache snapshot を再配布しません。

## 今できること

デフォルトの runtime path は次の通りです。

> MLX text/caption conditioning + MLX RF-DiT sampling + hosted MLX DACVAE codec artifact

現行 CLI で対応していること:

- 承認済み hosted VoiceDesign v2 / v3 RF-DiT artifact による WAV 生成
- 対応 Irodori-TTS `.safetensors` checkpoint の local inspection / conversion
- local hosted-layout directory/archive と direct local `.npz` fallback
- unquantized `mlx-audio` Irodori artifact の adaptation
- `irodori-tts-web` による任意の local Gradio UI
- `--requests-json`、metadata JSON、cleanup control による複数リクエスト処理

## セットアップ

~~~bash
git clone https://github.com/t0yohei/Irodori-TTS-MLX.git
cd Irodori-TTS-MLX

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
~~~

任意の extra:

~~~bash
python -m pip install -e ".[runtime,web]"  # local Gradio UI
python -m pip install -e ".[bench]"        # benchmark helper
~~~

Python は **3.11 から 3.14** を packaging 対象にしています。benchmark docs の基準環境は Python 3.11 です。

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

CLI はデフォルトで承認済み hosted DACVAE codec artifact を使います。codec revision を明示固定したい場合は、次のように指定します。

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
  --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-pinned-codec.wav
~~~

local / staged codec artifact を使う場合は `--codec-artifact-dir` または `--codec-path` を使ってください。

v3 は `--seconds` を省略すると predicted duration を使います。短い prompt で反復が出る場合は `--seconds 2.5` のように手動 duration を短くするか、`--duration-scale 0.75` から調整してください。

## 失敗したら

まず preflight を実行します。

~~~bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --preflight \
  --json
~~~

preflight は weights layout、model config、tokenizer repo、codec runtime mode、codec artifact path を解決し、tokenizer loading、MLX weight loading、DACVAE bridge construction、WAV generation の前に終了します。

よく見る確認先:

- tokenizer/cache: report された `text_tokenizer_repo` と、VoiceDesign では `caption_tokenizer_repo`
- hosted RF-DiT: `irodori_mlx_manifest.json` と `license_review.status: "approved"`
- hosted codec: `irodori_dacvae_codec_manifest.json`、または local `--codec-path` / `--codec-artifact-dir`

## 他の使い方

Local Web UI:

~~~bash
irodori-tts-web --host 127.0.0.1 --port 7860 --inbrowser
~~~

Local conversion fallback:

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

`mlx-audio` adaptation:

~~~bash
irodori-tts-adapt-mlx-audio \
  /path/to/mlx-audio/Irodori-TTS-500M-v2-VoiceDesign-fp16 \
  /tmp/irodori-mlx-hosted-layout \
  --source-repo mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16
~~~

CLI 全体は `irodori-tts-generate --help` を見てください。

## サポート境界

alpha 期間中に stable-ish な対象:

- installed console scripts: `irodori-tts-generate`、`irodori-tts-convert`、`irodori-tts-convert-dacvae-codec`、`irodori-tts-convert-dacvae-decoder`、`irodori-tts-inspect`、`irodori-tts-adapt-mlx-audio`、`irodori-tts-web`
- documented artifact layout、manifest、metadata、JSON output

まだ stable ではないもの:

- `irodori_mlx` import、top-level export、`scripts.*` module を public Python API として使うこと
- arbitrary third-party / fine-tuned / quantized / LoRA / renamed / architecture-modified checkpoint
- hosted demo、training、fine-tuning、watermark guarantee、generated / converted artifact 公開に対する自動的な法務承認

## ドキュメント

- 使い方と artifact layout: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- RF-DiT hosted artifact status: [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md)
- DACVAE codec artifact status: [docs/hosted_dacvae_codec_artifacts.md](docs/hosted_dacvae_codec_artifacts.md)
- Checkpoint support: [docs/checkpoint_support.md](docs/checkpoint_support.md)
- VoiceDesign support: [docs/caption_condition_support.md](docs/caption_condition_support.md)
- v3 support: [docs/v3_support.md](docs/v3_support.md)
- Public API stability: [docs/public_api_stability.md](docs/public_api_stability.md)
- Packaging: [docs/packaging.md](docs/packaging.md)
- License / distribution: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- Full docs index: [docs/README.md](docs/README.md)

## ライセンス

このリポジトリ自身の source code と documentation は、個別ファイルに別記がない限り [MIT License](LICENSE) です。

MIT License は upstream code、checkpoint files、DACVAE weights、tokenizer assets、reference audio、converted `.npz`、generated audio など、この repo が再配布していない artifact には適用されません。利用者は upstream artifact を自分で取得し、各 repository / model card の条件に従う必要があります。

## 関連リンク

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
