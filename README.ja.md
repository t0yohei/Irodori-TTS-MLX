# Irodori-TTS-MLX

[English README](README.md)

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を Apple Silicon / MLX で動かすための非公式 inference prototype です。

> [!IMPORTANT]
> まだ alpha 段階の CLI-first prototype です。現行実装は MLX RF-DiT weights と DACVAE codec 境界を使って WAV 生成できますが、安定した public Python API はまだ約束しません。このリポジトリは upstream code、checkpoint、Semantic-DACVAE weights、tokenizer assets、converted `.npz`、reference audio、generated audio、Hugging Face cache snapshot を再配布しません。

## 現在の実装範囲

実装済みの基本境界は次の通りです。

> MLX text/caption conditioning + MLX RF-DiT sampling + hosted MLX DACVAE codec artifact by default

現在できること:

- local / Hugging Face の Irodori-TTS `.safetensors` checkpoint inspection
- 対応 checkpoint から MLX 向け `.npz` RF-DiT weights への変換
- local `.npz`、local hosted-layout directory/archive、承認済み Hugging Face hosted-layout repo の読み込み
- unquantized `mlx-audio` Irodori artifact directory から、この repo の hosted weights layout への adaptation
- `irodori-tts-generate` / `scripts/generate_wav.py` による WAV 生成
- `irodori-tts-web` による任意の local Gradio Web UI 起動
- `--config-json`、`--requests-json`、`--cleanup-between-requests`、`--preset ultra-fast|fast|balanced|quality`、JSON metadata 出力、runtime reuse
- benchmark、hosted Apple Silicon validation workflow

public runtime のデフォルトは承認済み hosted DACVAE codec artifact を使い、upstream `irodori_tts.codec.DACVAECodec`、`torch`、`torchaudio` を必要としません。古い PyTorch DACVAE bridge fallback mode は generation runtime から削除し、公開 path は hosted / local codec artifact を使う `--codec-runtime-mode mlx` に一本化しています。

## Public API の安定性

alpha 期間中に stable-ish な user contract として扱うのは、installed CLI
（`irodori-tts-generate`、`irodori-tts-convert`、
`irodori-tts-convert-dacvae-codec`、`irodori-tts-convert-dacvae-decoder`、
`irodori-tts-inspect`、`irodori-tts-adapt-mlx-audio`）と、それらが使う
documented artifact layout、manifest、metadata、JSON output だけです。

`irodori_mlx` package、top-level export、`scripts.*` module は CLI、test、
repository development のための internal implementation surface です。import
はできますが、stable public Python API としてはまだ support しません。alpha
release では deprecation なしに変更・移動・rename・削除される可能性があります。
詳細な境界は [docs/public_api_stability.md](docs/public_api_stability.md) を参照してください。

## 現在のサポート境界

| Surface | Status | Public support boundary |
| --- | --- | --- |
| Project maturity | Alpha | CLI-first inference prototype です。console command と documented artifact layout が support 対象で、Python module layout は stable public API ではありません。 |
| VoiceDesign v2 hosted RF-DiT artifact | Supported | `--weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign` は documented no-reference caption quickstart 用に承認済みです。 |
| v3 hosted RF-DiT artifact | Supported | `--weights-repo t0yohei/Irodori-TTS-MLX-500M-v3` は documented no-reference predicted-duration quickstart 用に承認済みです。 |
| Base v2 speaker-conditioned generation | Experimental | inspection / conversion は対応済みです。generation は manual reference-audio path で、利用権のある user-supplied audio が必要です。 |
| Standalone MLX DACVAE codec artifact path | Supported default | 通常の public runtime は `--codec-runtime-mode mlx` で承認済み hosted/local codec artifact を使うため、upstream `irodori_tts.codec.DACVAECodec`、`torch`、`torchaudio` は不要です。 |
| PyTorch bridge-backed DACVAE codec path | Removed | `persistent`、`subprocess`、`mlx-decode` は公開 generation runtime mode ではありません。 |
| MLX DACVAE decode for no-reference generation | Supported | 承認済み codec artifact を使うと no-reference v3 / VoiceDesign の decode を upstream / PyTorch から外せます。 |
| Fully MLX DACVAE encode/decode for reference audio | Experimental | encoder / decoder tensor を含む executable local/hosted codec artifact が必要です。reference-audio speaker fidelity はまだ検証途上です。 |
| Local Web UI | Optional | `irodori-tts-web` は generation CLI の local Gradio wrapper です。hosted demo や stable public Python API 境界ではありません。 |
| Hosted artifacts outside the approved layouts | Blocked | documented manifest、checksum、provenance、approved license review がない repo は public support 外です。local conversion を使ってください。 |
| Unsupported upstream product features | Non-goal | training、LoRA fine-tuning、hosted demo 運用、watermark guarantee、任意 checkpoint compatibility、stable public Python API guarantee は対象外です。 |

`/path/to/...` や `/tmp/...` は user-managed file の placeholder です。private cache、maintainer local machine、未公開 public artifact を指すものではありません。

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
python -m pip install -e ".[runtime]"
```

checkout から開発するのではなく release artifact を使う package user は、clean environment に built wheel / sdist を install し、installed `irodori-tts-*` console script を使ってください。editable install、build validation、v0.2 release-artifact checklist は [docs/packaging.md](docs/packaging.md) にまとめています。

必要な workflow にだけ追加 dependency を入れます。

```bash
# benchmark helpers:
python -m pip install -e ".[bench]"
```

Python 3.11 では、`runtime`、`bench`、`dev` extra は audited artifact の tokenizer ecosystem と合わせて `sentencepiece>=0.1.99,<0.2` へ揃えています。Python 3.12 以降では wheel availability のため `sentencepiece>=0.2,<1` を使います。

デフォルトの standalone MLX runtime path は承認済み hosted codec artifact を使います。再現性のある環境構築は [docs/packaging.md](docs/packaging.md) と [docs/upstream_dependency.md](docs/upstream_dependency.md) を参照してください。

## 任意の Local Web UI

runtime と同じ環境に optional Web UI 依存を入れます。

```bash
python -m pip install -e ".[runtime,web]"
```

local UI を起動します。

```bash
irodori-tts-web --host 127.0.0.1 --port 7860 --inbrowser
```

Web UI は既存の `irodori-tts-generate` command に対する local Gradio
wrapper です。承認済み VoiceDesign / v3 hosted artifact の preset、local
weights、codec artifact 入力、reference audio、caption、sampling control、
生成 audio、metadata、log を扱えます。

reference audio は利用権のある音声だけを使ってください。UI は checkpoint、
codec weights、tokenizer assets、reference audio、generated WAV、Hugging Face
cache snapshot を再配布しません。この UI は任意の local-use surface であり、
`irodori_mlx` や `scripts.*` を stable public Python API にするものではありません。

## 入口: Hosted Weights

現行 CLI の最短経路は、承認済み hosted/pre-converted weights layout を `--weights-repo` で読み、デフォルトの承認済み hosted DACVAE codec artifact を使う方法です。`irodori_mlx_manifest.json` の `license_review.status: "approved"` と、README/model card の upstream checkpoint revision provenance を確認してください。

RF-DiT artifact の公開状況は [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md) に固定しています。VoiceDesign と v3 は承認済み public hosted artifact があります。

VoiceDesign の例:

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

v3 hosted の例:

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

v3 local fallback smoke の例:

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

デフォルトでは、これらの例は `--codec-runtime-mode mlx` と承認済み hosted DACVAE codec artifact を使います。local staged artifact を使う場合は `--codec-artifact-dir` または `--codec-path` を渡します。

承認済み hosted DACVAE codec artifact は RF-DiT weights とは別の repo/layout です。CLI は MLX codec mode でこの artifact をデフォルト利用しますが、明示的に pin することもできます。

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

no-reference の full-MLX run は metadata に `codec_decode_backend: "mlx"` と `codec_encode_backend: "not-required"` を報告します。`mlx` で reference-audio 生成する場合は、codec artifact 内の実行可能な Semantic-DACVAE encoder / decoder tensor を使い、`codec_encode_backend: "mlx"` と `codec_decode_backend: "mlx"` を報告します。承認済み hosted repo がない場合は、下の local conversion fallback を使います。詳細は [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) を参照してください。

### quickstart が失敗する場合

まず preflight を実行します。weights layout、model config、tokenizer repo 名、codec runtime mode、codec artifact path を解決し、tokenizer loading、MLX weight loading、DACVAE bridge construction、WAV generation は skip します。

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --preflight \
  --json
```

最初に失敗した surface に合わせて次の local action を選びます。

- tokenizer / Hugging Face cache: report された `text_tokenizer_repo` と、VoiceDesign では `caption_tokenizer_repo` の network/cache access を確認する
- hosted RF-DiT weights: `irodori_mlx_manifest.json`、`model_config.json`、`tokenizer_config.json`、`weights.npz`、`conversion_metadata.json`、`checksums.sha256`、`license_review.status: "approved"` を確認し、だめなら `--weights` の local conversion を使う
- hosted DACVAE codec: `irodori_dacvae_codec_manifest.json`、`dacvae-codec.npz`、`codec_metadata.json`、`checksums.sha256`、`license_review.status: "approved"` を確認し、だめなら local `--codec-path` / `--codec-artifact-dir` を使う

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

v3 では predicted duration を使うために `--seconds` を省略します。短い prompt では、予測 duration が長すぎると末尾の語句が反復して聞こえることがあります。CLI が predicted-duration warning を出す場合や、再生時に末尾が反復する場合は、`--seconds 2.5` のように短めの手動 duration を指定するか、予測は維持したまま `--duration-scale 0.75` を起点に調整してください。実験的な `--preset ultra-fast` は、手動 duration 指定がない場合だけこの短文 cap を自動適用します。Base v2 の speaker-conditioned check では `--no-reference` の代わりに、権利上問題のない `--reference-wav /path/to/reference.wav` を渡します。

## mlx-audio Adapter

`mlx-community/...` の Irodori repo は、この repo の `irodori_mlx_manifest.json` layout ではなく `config.json` + `model.safetensors` を持つため、直接 `--weights-repo` に渡しません。unquantized mlx-audio snapshot を local に置いて変換し、出力された hosted layout を使います。

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

adapter は現在 unquantized v2/base と VoiceDesign layout のみ対応です。quantized mlx-audio artifact は、quantized runtime support が設計されるまで拒否されます。

VoiceDesign v2 で `--seconds` を省略すると `duration_mode: "estimated"` になり、主に `--text` から duration を推定し、`--caption` は話速や雰囲気の小さな補正だけに使います。正確な長さにしたい場合は `--seconds`、推定値を少し短く/長くしたい場合は `--duration-scale` を使ってください。

複数リクエストをまとめて処理する場合は、`--requests-json` に request object を並べると初期化済み runtime を再利用できます。最大 throughput より memory residency を優先する場合は `--cleanup-between-requests` を追加してください。

## 主なコマンド

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

通常の操作は installed console script を使います。`python scripts/*.py` の直接実行は repository development と benchmark maintenance 用です。

## 主なドキュメント

- Architecture: [docs/architecture.md](docs/architecture.md)
- DACVAE bridge / generation CLI: [docs/dacvae_bridge.md](docs/dacvae_bridge.md)
- v0.2 hosted/pre-converted MLX weights layout contract: [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md)
- Hosted weights usage / local conversion fallback: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- Hosted RF-DiT artifact publication status: [docs/hosted_rf_dit_artifacts.md](docs/hosted_rf_dit_artifacts.md)
- Hosted DACVAE codec artifact publication status: [docs/hosted_dacvae_codec_artifacts.md](docs/hosted_dacvae_codec_artifacts.md)
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
- Public API stability boundary: [docs/public_api_stability.md](docs/public_api_stability.md)
- License / distribution policy: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- v0.2 cross-repository delivery plan / downstream consumer handoff boundary: [docs/v0_2_delivery_plan.md](docs/v0_2_delivery_plan.md)

## 対象外

- training / fine-tuning
- Semantic-DACVAE codec の同梱または完全再配布
- 任意 third-party checkpoint との互換保証
- hosted demo
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
