# Upstream `irodori_tts` dependency boundary

Issue: [#72 Document the upstream irodori_tts dependency boundary](https://github.com/t0yohei/Irodori-TTS-MLX/issues/72)  
Parent: [#68 v0.1 documentation and release readiness](https://github.com/t0yohei/Irodori-TTS-MLX/issues/68)

## When this repo still needs upstream `irodori_tts`

The v0.2 public runtime has a standalone MLX runtime path by default. Upstream `irodori_tts` is still needed only for explicit PyTorch bridge fallback, upstream parity checks, and comparison workflows.

- this repository owns the MLX text/caption conditioning, RF-DiT model path, weight conversion, duration handling, and Euler RF sampling surface;
- upstream `irodori_tts` owns the explicit PyTorch bridge fallback through `irodori_tts.codec.DACVAECodec`, used by `persistent` / `subprocess` modes and by `mlx-decode` when reference-audio encode falls back to PyTorch.

That boundary is deliberate fallback behavior, not an accidental missing import. The recommended public path uses approved hosted RF-DiT weights plus the approved hosted DACVAE codec artifact and does not import upstream `DACVAECodec`.

For local v0.2 DACVAE artifact work, `scripts/generate_wav.py --codec-runtime-mode mlx --codec-path /path/to/dacvae-codec.npz` can use a local MLX codec artifact instead of importing upstream PyTorch codec code. Hosted public examples default to `t0yohei/Irodori-TTS-MLX-DACVAE-Codec`. This repository still does not bundle Semantic-DACVAE weights or claim that arbitrary codec artifacts match the upstream acoustic model.

## Recommended install path

For WAV generation, first install this repository with its runtime dependencies:

```bash
python3.11 -m venv .venv  # or another supported Python from docs/packaging.md
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
```

On Python 3.11, the runtime extra uses the same
`sentencepiece>=0.1.99,<0.2` range as upstream `irodori-tts`, so the next
editable install should not force `sentencepiece==0.2.x` into the shared venv.
On Python 3.12 and newer, use Python 3.11 for same-venv upstream installs or
keep upstream on `PYTHONPATH`, because `sentencepiece==0.1.99` does not publish
wheels for the newer Python packaging targets.

For PyTorch bridge fallback or parity workflows, make a local upstream checkout importable from the same environment. The recommended option is an editable install:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git /path/to/Irodori-TTS
python -m pip install -e /path/to/Irodori-TTS
```

Use `PYTHONPATH` only when you intentionally do not want to install the upstream checkout into the venv for bridge/parity modes:

```bash
export PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}
```

Both methods must expose this import path:

```python
from irodori_tts.codec import DACVAECodec
```

Use a current upstream checkout whose `DACVAECodec.load` accepts the codec
keyword arguments used by this runtime, including `enable_watermark` and
`normalize_db`. When watermarking is disabled, the bridge can still retry older
`DACVAECodec.load` signatures that do not expose `enable_watermark`; requesting
watermarking requires an upstream checkout with that keyword.

## Responsibility split

| Area | Owner in v0.1 | Notes |
| --- | --- | --- |
| Text tokenizer loading and prompt token preparation | this MLX repo | Uses Transformers tokenizers configured by `ModelConfig`; tokenizer assets are not vendored. |
| Caption tokenizer and VoiceDesign-style conditioning | this MLX repo | Supported for the inspected public VoiceDesign family only. |
| RF-DiT forward pass and MLX layers | this MLX repo | Runs converted `.npz` weights through MLX modules. |
| Duration predictor / explicit `--seconds` semantics | this MLX repo | v3 predicted-duration behavior is implemented here. |
| Reference WAV loading and conditioning handoff | this MLX repo by default | Full-MLX mode uses executable Semantic-DACVAE encoder tensors from the codec artifact; `mlx-decode` can still fall back to upstream PyTorch encode. |
| DACVAE encode/decode | this MLX repo by default; upstream fallback explicit | Full-MLX mode uses hosted/local codec artifacts. Explicit `persistent` / `subprocess` modes call `irodori_tts.codec.DACVAECodec`. |
| WAV writing fallback | this MLX repo | Writes decoded audio with `soundfile` or the stdlib fallback; PyTorch bridge modes may also use `torchaudio`. |

## Import-failure behavior

If upstream is not importable, runtime construction fails early with a message like:

```text
The PyTorch DACVAE bridge currently reuses upstream irodori_tts.codec.DACVAECodec. Install the upstream Irodori-TTS package or add its checkout to PYTHONPATH.
```

That message is expected only for bridge-backed modes. Fix the environment by either running `python -m pip install -e /path/to/Irodori-TTS` in the active venv or exporting `PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}` before starting explicit bridge/parity commands or any code that constructs `PyTorchDACVAEBridge`.

If runtime construction fails with a message about `DACVAECodec.load` keyword
arguments, update the upstream checkout installed in the active environment. The
MLX bridge only retries the older no-`enable_watermark` signature when
watermarking is disabled.

## What this does not claim

This repository does provide standalone v0.2 WAV generation without upstream `irodori_tts` when using approved hosted/local MLX codec artifacts. It still does not vendor upstream code, checkpoints, tokenizer assets, generated audio, or codec weights.

Standalone v0.2 codec experiments require an approved hosted codec artifact, `--codec-artifact-dir`, or caller-provided `--codec-path`. Without an available codec artifact, use the upstream PyTorch DACVAE bridge explicitly for both reference encode and waveform decode.
