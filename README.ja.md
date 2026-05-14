# irodori-tts-mlx

[English README](README.md)

Apple Silicon 向けに [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) の推論系を MLX へ移植する、非公式プロトタイプです。

> [!IMPORTANT]
> このリポジトリは実験段階のプロトタイプです。現時点では、広く安定利用できる完成済みプロダクトではありません。

## この README の役割

- このファイルは、日本語話者向けの導線と要約を提供します
- 技術仕様・最新の厳密な定義は原則として [English README](README.md) を正とします
- 詳細な実装背景や手順は各 `docs/*.md` へ分離し、README 同士の重複を減らします

## プロジェクトの狙い

最初の実用目標は、次の境界を持つ v0 推論プロトタイプです。

> MLX RF-DiT inference + PyTorch DACVAE encode/decode bridge

つまり、Irodori-TTS のテキスト/条件エンコーダ、RF-DiT、rectified-flow sampler は MLX 側に寄せつつ、参照音声のエンコードと波形デコードは upstream の PyTorch DACVAE を使います。

この切り分けにより、最初の段階では「MLX 化の恩恵が大きい中核推論パス」に集中し、DACVAE の完全移植は後段に回せます。

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

## 現時点の制約

- Python は **3.11 系** を前提にしています
- bridge runtime は upstream `irodori_tts` の `DACVAECodec` に依存します
- 完全な MLX DACVAE 移植は未対応です
- 学習・微調整用途は対象外です
- Web UI / Gradio は対象外です
- すべての歴史的チェックポイント互換を保証するものではありません
- ライセンスは未確定です。派生物の再配布前に upstream / model card の条件確認が必要です

## セットアップ

基本の導入例:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[runtime]"  # WAV 生成 / bridge runtime
python -m pip install -e ".[bench]"    # benchmark + conversion workflow
python -m pip install -e ".[dev]"      # contributor environment
```

bridge runtime を使う場合は、同じ仮想環境に upstream `irodori_tts` を入れるか、`PYTHONPATH` から参照できるようにしてください。

再現性のある環境構築や依存関係の詳細は [docs/packaging.md](docs/packaging.md) を参照してください。

## 使い始めの入口

### 1. チェックポイントの中身を確認する

```bash
python3 scripts/inspect_checkpoint.py Aratako/Irodori-TTS-500M-v2
python3 scripts/inspect_checkpoint.py /path/to/model.safetensors --all-tensors
```

### 2. 重みを MLX 向けに変換する

```bash
python3 scripts/convert_weights.py /path/to/model.safetensors /path/to/irodori-tts-500m-v2.npz
python3 scripts/convert_weights.py /path/to/model.safetensors --dry-run --json
```

### 3. WAV 生成を試す

`scripts/generate_wav.py` が現在の主要な利用入口です。依存関係、引数、制約は [docs/dacvae_bridge.md](docs/dacvae_bridge.md) を参照してください。

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
