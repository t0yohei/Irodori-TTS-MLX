# irodori-tts-mlx

[English README](README.md)

Apple Silicon 向けに [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の推論系を MLX へ移植する、非公式プロトタイプです。

> [!IMPORTANT]
> このリポジトリは alpha 段階の推論プロトタイプです。互換 checkpoint と runtime 依存関係をローカルで用意すれば、MLX RF-DiT + upstream PyTorch DACVAE bridge で WAV 生成まで実行できます。ただし、checkpoint 再配布、学習、Web UI、完全な MLX DACVAE 移植は対象外です。

## この README の役割

- このファイルは、日本語話者向けの導線と要約を提供します
- 技術仕様・最新の厳密な定義は原則として [English README](README.md) を正とします
- 詳細な実装背景や手順は各 `docs/*.md` へ分離し、README 同士の重複を減らします

## v0.1 の現在スコープ

このリポジトリは、次の境界を持つ Apple Silicon 向け推論プロトタイプです。

> MLX RF-DiT inference + PyTorch DACVAE encode/decode bridge

Irodori-TTS のテキスト/条件エンコーダ、RF-DiT、rectified-flow sampler は MLX 側に寄せ、参照音声のエンコードと波形デコードは upstream `irodori_tts` の PyTorch `DACVAECodec` を使います。

v0.1 プロトタイプで現在できること:

- 対応 checkpoint のメタデータ / tensor layout 確認
- `.safetensors` から MLX 向け `.npz` への変換
- MLX RF-DiT + PyTorch DACVAE bridge による WAV 生成
- VoiceDesign caption 条件と v3 予測 duration パスの利用
- benchmark / parity / hosted Apple Silicon workflow による検証

## 現在の状態

現在のマイルストーン進捗は次の通りです。

- M0 Baseline: 完了
- M1 Weight conversion: 完了
- M2 MLX model parity: 完了
- M3 MLX inference prototype: 現行 CLI / runtime スコープでは完了
- M4 Performance and packaging: 現行プロトタイプ範囲では完了

現在のリポジトリには、以下のような成果物が含まれます。

- チェックポイント情報の確認 (`scripts/inspect_checkpoint.py`)
- `.safetensors` から MLX 向け `.npz` への変換 (`scripts/convert_weights.py`)
- MLX RF-DiT + PyTorch DACVAE bridge による WAV 生成 (`scripts/generate_wav.py`)
- ベンチマーク実行 (`scripts/benchmark.py`)

## 対応範囲と制約

v0.1 で対応対象としている checkpoint family:

- base `Aratako/Irodori-TTS-500M-v2` 系
- `Aratako/Irodori-TTS-500M-v2-VoiceDesign` caption 条件付き系
- `Aratako/Irodori-TTS-500M-v3` 系（`--seconds` 省略時の予測 duration を含む）

現時点の制約:

- Python は **3.11 から 3.14** を packaging 対象にしています（benchmark の基準は 3.11）
- bridge runtime は upstream `irodori_tts` の `DACVAECodec` に依存します
- 完全な MLX DACVAE 移植は未対応です
- 学習・微調整用途は対象外です
- Web UI / Gradio は対象外です
- すべての歴史的 / third-party checkpoint 互換を保証するものではありません
- このリポジトリ自身のコードとドキュメントは MIT License です。ただし upstream code、checkpoint、DACVAE weights、converted `.npz`、生成音声は原則再配布しません。v0.2 の hosted pre-converted weights に限る監査結果と provenance 要件は [docs/preconverted_weights_redistribution_audit.md](docs/preconverted_weights_redistribution_audit.md)、hosted 利用手順と local conversion fallback は [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)、通常の再配布方針は [docs/license_and_distribution.md](docs/license_and_distribution.md) を参照してください

## セットアップ

基本の導入例（checkpoint 変換から WAV 生成まで試す場合は `runtime` と `bench` の両方を入れます）:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime,bench]"  # WAV 生成 + checkpoint 変換
python -m pip install -e ".[dev]"            # contributor environment
```

bridge runtime を使う場合は、upstream `irodori_tts` の `irodori_tts.codec.DACVAECodec` が import できる必要があります。推奨は同じ仮想環境で `python -m pip install -e /path/to/Irodori-TTS` する方法で、`PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}` は未インストール checkout を使いたい場合の代替です。これは v0.1 の意図した境界で、この repo は text/caption conditioning、RF-DiT、変換、duration、sampler を持ち、PyTorch DACVAE encode/decode は upstream に残します。

```bash
python -m pip install -e /path/to/Irodori-TTS  # または PYTHONPATH=/path/to/Irodori-TTS
```

再現性のある環境構築や依存関係の詳細は [docs/upstream_dependency.md](docs/upstream_dependency.md) と [docs/packaging.md](docs/packaging.md) を参照してください.

## 使い始めの入口

### 0. v0.2 hosted converted weights を使う場合

v0.2 では、承認済みの hosted converted weights repository を `--weights-repo` で読み込む導線を用意します。この flag は #82 の hosted-loader CLI 実装が入った後の導線です。現在の `main` CLI では、下の local conversion path を使ってください。公開 repo は `irodori_mlx_manifest.json` の `license_review.status` が `approved` で、upstream checkpoint revision、変換元、ライセンス監査へのリンクを README/model card に明記しているものだけを使ってください。

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-repo t0yohei/irodori-tts-mlx-voicedesign-v2-500m \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-hosted.wav \
  --preset balanced
```

この経路でも upstream PyTorch DACVAE bridge は必要です。hosted repo が未公開・未承認・監査外の場合は、下の local conversion path を使います。詳細は [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) を参照してください。

### 1. チェックポイントの中身を確認する

```bash
python3 scripts/inspect_checkpoint.py Aratako/Irodori-TTS-500M-v2
python3 scripts/inspect_checkpoint.py /path/to/model.safetensors --all-tensors
```

### 2. 重みを MLX 向けに変換する

```bash
python3 scripts/convert_weights.py /path/to/model.safetensors /path/to/irodori-tts.npz
python3 scripts/convert_weights.py /path/to/model.safetensors --dry-run --json
```

### 3. WAV 生成を試す

```bash
python3 scripts/generate_wav.py \
  --weights /path/to/irodori-tts.npz \
  --text "こんにちは、いろどりです。" \
  --reference-wav /path/to/reference.wav \
  --output out.wav \
  --preset balanced
```

VoiceDesign checkpoint では [docs/caption_condition_support.md](docs/caption_condition_support.md) に従い、caption 対応の `--model-config-json` と `--caption "..."` を追加します。v3 checkpoint では [docs/v3_support.md](docs/v3_support.md) を参照し、v3 用の `--model-config-json` を渡したうえで、予測 duration を使う場合は `--seconds` を省略します。依存関係、引数、制約は [docs/dacvae_bridge.md](docs/dacvae_bridge.md) を参照してください。

### 4. ベンチマークを回す

```bash
python3 scripts/benchmark.py --self-test
```

本格的な計測フローや比較方法は [docs/benchmark.md](docs/benchmark.md) を参照してください。

## 主なドキュメント

- アーキテクチャ概要: [docs/architecture.md](docs/architecture.md)
- upstream PyTorch ベースライン: [docs/baseline.md](docs/baseline.md)
- 重み変換: [docs/weight_mapping.md](docs/weight_mapping.md)
- RF-DiT forward parity: [docs/rf_dit_forward.md](docs/rf_dit_forward.md)
- RF sampler: [docs/rf_sampler.md](docs/rf_sampler.md)
- DACVAE bridge / WAV 生成: [docs/dacvae_bridge.md](docs/dacvae_bridge.md)
- upstream `irodori_tts` 依存境界: [docs/upstream_dependency.md](docs/upstream_dependency.md)
- ライセンス / 再配布ポリシー: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- v0.2 hosted MLX weights layout: [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md)
- v0.2 hosted weights usage / local conversion fallback: [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md)
- packaging / install: [docs/packaging.md](docs/packaging.md)
- benchmark: [docs/benchmark.md](docs/benchmark.md)
- VoiceDesign サポート: [docs/caption_condition_support.md](docs/caption_condition_support.md)
- v3 サポート: [docs/v3_support.md](docs/v3_support.md)

## README の責務分担

ドキュメントのドリフトを避けるため、README の責務は次のように分けます。

- `README.md`: 正式な英語版。最新の仕様、互換性、細かい技術説明の基準点
- `README.ja.md`: 日本語の導入、全体像、主要な制約、入口リンク
- `docs/*.md`: 具体的な手順、検証結果、設計詳細

英語版に重要な仕様変更が入った場合、日本語版も同じ論点を保てているか見直してください。ただし、細かな全文対訳までは要求せず、日本語話者が迷わず使い始められることを優先します。

## 関連リンク

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
- DACVAE: <https://github.com/facebookresearch/dacvae>
