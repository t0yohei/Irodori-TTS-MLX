# Irodori-TTS-MLX

[English README](README.md)

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を Apple Silicon / MLX で動かすための非公式 inference prototype です。

> [!IMPORTANT]
> まだ alpha 段階の CLI-first prototype です。現行実装は MLX RF-DiT weights と DACVAE codec 境界を使って WAV 生成できますが、安定した public Python API はまだ約束しません。このリポジトリは upstream code、checkpoint、Semantic-DACVAE weights、tokenizer assets、converted `.npz`、reference audio、generated audio、Hugging Face cache snapshot を再配布しません。

## 現在の実装範囲

実装済みの基本境界は次の通りです。

> MLX text/caption conditioning + MLX RF-DiT sampling + upstream PyTorch DACVAE bridge by default

現在できること:

- local / Hugging Face の Irodori-TTS `.safetensors` checkpoint inspection
- 対応 checkpoint から MLX 向け `.npz` RF-DiT weights への変換
- local `.npz`、local hosted-layout directory/archive、承認済み Hugging Face hosted-layout repo の読み込み
- unquantized `mlx-audio` Irodori artifact directory から、この repo の hosted weights layout への adaptation
- `irodori-tts-generate` / `scripts/generate_wav.py` による WAV 生成
- `--config-json`、`--requests-json`、`--preset fast|balanced|quality`、JSON metadata 出力、persistent runtime reuse
- benchmark、parity check、hosted Apple Silicon validation workflow

デフォルトの codec path は upstream `irodori_tts.codec.DACVAECodec` に依存します。v0.2 codec-port 用の local MLX codec artifact mode はありますが、この repo は codec weights を同梱せず、任意 codec artifact の acoustic parity も保証しません。

## 対応 checkpoint

対応対象は、この repo で layout と runtime semantics を明示検証している family に限ります。

| Checkpoint family | 例 | Inspect | Convert | Generate |
| --- | --- | --- | --- | --- |
| Base v2 speaker-conditioned | `Aratako/Irodori-TTS-500M-v2` | 対応 | 対応 | reference audio 付きの experimental manual path |
| VoiceDesign v2 caption-conditioned | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | 対応 | 対応 | `--caption` + `--no-reference` で対応 |
| v3 speaker-conditioned / duration predictor | `Aratako/Irodori-TTS-500M-v3` | 対応 | 対応 | `--seconds` 省略時の predicted duration に対応 |

historical / third-party / fine-tuned / quantized / LoRA / renamed / architecture-modified checkpoint は、別途 audit されるまで local-conversion-only です。

Python は **3.11 から 3.14** を packaging 対象にしています。benchmark doc の基準環境は Python 3.11 です。

## セットアップ

```bash
git clone https://github.com/t0yohei/Irodori-TTS-MLX.git
cd Irodori-TTS-MLX

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime,bench]"
```

checkout から開発するのではなく release artifact を使う package user は、clean environment に built wheel / sdist を install し、installed `irodori-tts-*` console script を使ってください。editable install、build validation、v0.2 release-artifact checklist は [docs/packaging.md](docs/packaging.md) にまとめています。

upstream Irodori-TTS も同じ環境に入れるか、既存 checkout を import 可能にします。

```bash
git clone https://github.com/Aratako/Irodori-TTS.git ../Irodori-TTS
python -m pip install -e ../Irodori-TTS

# upstream checkout が別の場所にある場合の形:
# python -m pip install -e /path/to/Irodori-TTS

# 未インストール checkout を使う場合:
# export PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}
```

default の `persistent` / `subprocess` codec mode では upstream `irodori_tts.codec.DACVAECodec` が必要です。再現性のある環境構築は [docs/packaging.md](docs/packaging.md) と [docs/upstream_dependency.md](docs/upstream_dependency.md) を参照してください。

## 入口: Hosted Weights

現行 CLI の最短経路は、承認済み hosted/pre-converted weights layout を `--weights-repo` で読むか、同じ layout を disk から `--weights-dir` で読む方法です。`irodori_mlx_manifest.json` の `license_review.status: "approved"` と、README/model card の upstream checkpoint revision provenance を確認してください。

RF-DiT artifact の公開状況は [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md) に固定しています。VoiceDesign と v3 は承認済み public hosted artifact があります。

VoiceDesign の例:

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

v3 hosted の例:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-hosted-metadata.json
```

v3 local fallback smoke の例:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights /path/to/converted-v3/weights.npz \
  --model-config-json /path/to/converted-v3/model_config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav \
  --preset balanced \
  --metadata-json /tmp/irodori-v3-metadata.json
```

この経路でも、明示的に local MLX codec artifact mode を選ばない限り upstream PyTorch DACVAE bridge を使います。v3 / VoiceDesign の no-reference 生成では、decode 可能な変換済み DACVAE artifact を渡すと codec decode を MLX に寄せ、reference encode も不要にできます:

```bash
irodori-tts-generate \
  --weights /path/to/converted-v3/weights.npz \
  --model-config-json /path/to/converted-v3/model_config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3-mlx-decode.wav \
  --preset balanced \
  --codec-runtime-mode mlx-decode \
  --codec-path /path/to/dacvae-codec.npz \
  --metadata-json /tmp/irodori-v3-mlx-decode-metadata.json
```

この no-reference 経路の metadata は `codec_decode_backend: "mlx"` と `codec_encode_backend: "not-required"` を報告します。`mlx-decode` で reference-audio 生成を行う場合は、従来どおり PyTorch encode fallback を使い、その backend を `codec_encode_backend` に記録します。reference-audio 生成も PyTorch DACVAE bridge から完全に外すには、実行可能な Semantic-DACVAE encoder / decoder tensor を含む artifact と `--codec-runtime-mode mlx` を使います。この full-MLX 経路では `codec_encode_backend: "mlx"` と `codec_decode_backend: "mlx"` を報告します。承認済み hosted repo がない場合は、下の local conversion fallback を使います。詳細は [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) を参照してください。

## 入口: Local Conversion Fallback

hosted repo が未公開・未承認・private・audit 外の場合は、local conversion fallback を使います。

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

v3 では predicted duration を使うために `--seconds` を省略します。Base v2 の speaker-conditioned check では `--no-reference` の代わりに、権利上問題のない `--reference-wav /path/to/reference.wav` を渡します。

## mlx-audio Adapter

`mlx-community/...` の Irodori repo は、この repo の `irodori_mlx_manifest.json` layout ではなく `config.json` + `model.safetensors` を持つため、直接 `--weights-repo` に渡しません。unquantized mlx-audio snapshot を local に置いて変換し、出力された hosted layout を使います。

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

adapter は現在 unquantized v2/base と VoiceDesign layout のみ対応です。quantized mlx-audio artifact は、quantized runtime support が設計されるまで拒否されます。

## 主なコマンド

```bash
irodori-tts-inspect /path/to/model.safetensors --all-tensors
irodori-tts-convert /path/to/model.safetensors /path/to/weights.npz
irodori-tts-convert /path/to/model.safetensors --dry-run --json
irodori-tts-generate --help
irodori-tts-adapt-mlx-audio --help
python scripts/benchmark.py --self-test
```

通常の操作は installed console script を使います。`python scripts/*.py` の直接実行は repository development と benchmark maintenance 用です。

## 主なドキュメント

- Architecture: [docs/architecture.md](docs/architecture.md)
- DACVAE bridge / generation CLI: [docs/dacvae_bridge.md](docs/dacvae_bridge.md)
- v0.2 hosted/pre-converted MLX weights layout contract: [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md)
- Hosted weights usage / local conversion fallback: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- Hosted RF-DiT artifact publication status: [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md)
- mlx-audio interop / adapter boundary: [docs/mlx_audio_interop.md](docs/mlx_audio_interop.md)
- DACVAE artifact layout: [docs/codec_artifact_layout.md](docs/codec_artifact_layout.md)
- Checkpoint support matrix: [docs/checkpoint_support.md](docs/checkpoint_support.md)
- VoiceDesign support: [docs/caption_condition_support.md](docs/caption_condition_support.md)
- v3 support: [docs/v3_support.md](docs/v3_support.md)
- Text preprocessing contract: [docs/text_preprocessing.md](docs/text_preprocessing.md)
- Weight mapping: [docs/weight_mapping.md](docs/weight_mapping.md)
- RF sampler: [docs/rf_sampler.md](docs/rf_sampler.md)
- Benchmark: [docs/benchmark.md](docs/benchmark.md)
- Packaging: [docs/packaging.md](docs/packaging.md)
- License / distribution policy: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- v0.2 cross-repository delivery plan / downstream consumer handoff boundary: [docs/v0_2_delivery_plan.md](docs/v0_2_delivery_plan.md)

## 対象外

- training / fine-tuning
- Semantic-DACVAE codec の同梱または完全再配布
- 任意 third-party checkpoint との互換保証
- GUI / Gradio / hosted demo
- stable public Python API
- converted weights や generated audio の公開に対する自動的な法務承認

## ライセンス

このリポジトリ自身の source code と documentation は、個別ファイルに別記がない限り [MIT License](LICENSE) です。

MIT License は upstream code、checkpoint files、DACVAE weights、tokenizer assets、reference audio、converted `.npz`、generated audio など、この repo が再配布していない artifact には適用されません。利用者は upstream artifact を自分で取得し、各 repository / model card の条件に従う必要があります。詳細は [docs/license_and_distribution.md](docs/license_and_distribution.md) と [docs/preconverted_weights_redistribution_audit.md](docs/preconverted_weights_redistribution_audit.md) を参照してください。

## 関連リンク

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
- DACVAE: <https://github.com/facebookresearch/dacvae>
