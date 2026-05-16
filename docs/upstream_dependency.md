# Upstream `irodori_tts` dependency boundary

Issue: [#72 Document the upstream irodori_tts dependency boundary](https://github.com/t0yohei/Irodori-TTS-MLX/issues/72)  
Parent: [#68 v0.1 documentation and release readiness](https://github.com/t0yohei/Irodori-TTS-MLX/issues/68)

## Why this repo still needs upstream `irodori_tts`

The v0.1 runtime is intentionally a mixed MLX + PyTorch bridge:

- this repository owns the MLX text/caption conditioning, RF-DiT model path, weight conversion, duration handling, and Euler RF sampling surface;
- upstream `irodori_tts` still owns the PyTorch `irodori_tts.codec.DACVAECodec` used to encode reference audio into DACVAE latents and decode generated latents back to waveform audio.

That boundary is a deliberate v0.1 constraint, not an accidental missing import. A full MLX DACVAE port is not required to validate v0.1 because the current milestone is about the latent-generation path: converted Irodori-TTS weights running through MLX and then crossing back through the known upstream DACVAE implementation for audio I/O.

For v0.2 DACVAE port work, `scripts/generate_wav.py --codec-runtime-mode mlx --codec-path /path/to/dacvae-codec.npz` can use a local MLX codec artifact instead of importing upstream PyTorch codec code. That path is covered by local contract tests and is suitable for fixed parity fixtures, but this repository still does not bundle Semantic-DACVAE weights or claim that arbitrary codec artifacts match the upstream acoustic model.

## Recommended install path

For WAV generation, first install this repository with its runtime dependencies:

```bash
python3.11 -m venv .venv  # or another supported Python from docs/packaging.md
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
```

Then make a local upstream checkout importable from the same environment. The recommended option is an editable install:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git /path/to/Irodori-TTS
python -m pip install -e /path/to/Irodori-TTS
```

Use `PYTHONPATH` only when you intentionally do not want to install the upstream checkout into the venv:

```bash
export PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}
```

Both methods must expose this import path:

```python
from irodori_tts.codec import DACVAECodec
```

## Responsibility split

| Area | Owner in v0.1 | Notes |
| --- | --- | --- |
| Text tokenizer loading and prompt token preparation | this MLX repo | Uses Transformers tokenizers configured by `ModelConfig`; tokenizer assets are not vendored. |
| Caption tokenizer and VoiceDesign-style conditioning | this MLX repo | Supported for the inspected public VoiceDesign family only. |
| RF-DiT forward pass and MLX layers | this MLX repo | Runs converted `.npz` weights through MLX modules. |
| Duration predictor / explicit `--seconds` semantics | this MLX repo | v3 predicted-duration behavior is implemented here. |
| Reference WAV loading and conditioning handoff | mixed boundary | This repo validates request semantics, then asks upstream DACVAE to produce latents. |
| DACVAE encode/decode | upstream `irodori_tts` | Calls `irodori_tts.codec.DACVAECodec`; the codec remains PyTorch-only in v0.1. |
| WAV writing fallback | this MLX repo | Writes decoded audio with `torchaudio`, `soundfile`, or the stdlib fallback. |

## Import-failure behavior

If upstream is not importable, runtime construction fails early with a message like:

```text
The PyTorch DACVAE bridge currently reuses upstream irodori_tts.codec.DACVAECodec. Install the upstream Irodori-TTS package or add its checkout to PYTHONPATH.
```

That message is expected. Fix the environment by either running `python -m pip install -e /path/to/Irodori-TTS` in the active venv or exporting `PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}` before starting `scripts/generate_wav.py`, `scripts/benchmark.py --mode mlx`, or any code that constructs `PyTorchDACVAEBridge` / `MLXDACVAERuntime`.

## What this does not claim

This repository does **not** provide standalone v0.1 WAV generation without upstream `irodori_tts`. It also does not vendor upstream code or replace the DACVAE model with MLX yet. Those are possible later milestones, but v0.1 keeps the DACVAE boundary in PyTorch so the MLX RF-DiT path can be validated first.

Standalone v0.2 codec experiments require a caller-provided converted MLX codec artifact. Without that artifact, keep using the upstream PyTorch DACVAE bridge for both reference encode and waveform decode.
