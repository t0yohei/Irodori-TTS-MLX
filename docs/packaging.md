# Packaging and reproducible environments

Issue: [#31 Package the project for reproducible runtime and benchmark environments](https://github.com/t0yohei/irodori-tts-mlx/issues/31)

The repository now exposes a project-level `pyproject.toml` so contributors can install dependencies by use case instead of guessing from ad hoc notes.

## Supported Python

The currently supported Python versions for packaging and editable installs are:

- **Python 3.11**
- **Python 3.12**
- **Python 3.13**
- **Python 3.14**

Python 3.11 remains the reference environment for the current Apple Silicon benchmark reports, so benchmark reproduction examples below continue to use 3.11 even though the packaging surface now supports 3.12 through 3.14 as well.

## Dependency groups

The project defines these install targets:

- base install: core MLX package modules (`irodori_mlx.layers`, encoders, model, sampler, weights)
- `.[runtime]`: end-to-end WAV generation with the PyTorch DACVAE bridge
- `.[bench]`: benchmark-oriented environment for `scripts/benchmark.py` and checkpoint conversion helpers
- `.[dev]`: local contributor environment for tests plus packaging validation helpers

## Quick start

### 1. Create a supported Python virtual environment

```bash
python3.11 -m venv .venv  # or: python3.12/3.13/3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install this repository in editable mode

#### Core MLX package only

```bash
python -m pip install -e .
```

#### Runtime / WAV generation environment

```bash
python -m pip install -e ".[runtime]"
```

#### Benchmark environment

```bash
python -m pip install -e ".[bench]"
```

#### Development environment

```bash
python -m pip install -e ".[dev]"
```

## Upstream dependency boundary

The v0 runtime still reuses upstream `irodori_tts.codec.DACVAECodec` for reference encode / waveform decode.
That means the local environment must also be able to import upstream `irodori_tts`.

Supported ways to provide it:

### Option A: install the upstream checkout into the same venv

```bash
git clone https://github.com/Aratako/Irodori-TTS.git /path/to/Irodori-TTS
python -m pip install -e /path/to/Irodori-TTS
```

### Option B: leave upstream uninstalled and expose it with `PYTHONPATH`

```bash
export PYTHONPATH=/path/to/Irodori-TTS:$PYTHONPATH
```

The local benchmark harness already supports this pattern through `--upstream-root`.

## Reproducible runtime setup

For the current bridge prototype, the minimal practical setup is:

```bash
python3.11 -m venv .venv  # or: python3.12/3.13/3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
python -m pip install -e /path/to/Irodori-TTS
```

Then run:

```bash
python scripts/generate_wav.py \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --reference-wav /path/to/reference.wav \
  --text "こんにちは。今日は良い天気です。" \
  --output /tmp/irodori.wav \
  --seconds 5 \
  --num-steps 40 \
  --codec-device cpu
```

## Reproducible benchmark setup

A benchmark-oriented environment should include the benchmark extra plus an accessible upstream checkout. The example keeps Python 3.11 because that is the benchmark reference environment used by the current reports, even though packaging support now extends through Python 3.14.

```bash
python3.11 -m venv .venv-bench311
. .venv-bench311/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[bench]"
python -m pip install -e /path/to/Irodori-TTS
```

If you prefer not to install upstream into the venv, you can instead keep it on `PYTHONPATH` and still pass `--upstream-root` to the benchmark harness.

Smoke-check the packaging surface with:

```bash
python scripts/benchmark.py --self-test
python -m unittest discover -s tests -v
```

## Notes on local scripts

The existing repository scripts remain repo-local entry points:

- `python scripts/generate_wav.py`
- `python scripts/benchmark.py`
- `python scripts/convert_weights.py`
- `python scripts/inspect_checkpoint.py`

They continue to work with the packaged dependency layout because the repo is installed editable (`-e`) and each script already resolves the repository root when importing local modules.

## Current limitations

- The packaging surface is still prototype-grade; there is no stable API guarantee yet.
- Python 3.11 through 3.14 are the currently supported packaged environments; newer Python versions should stay unsupported until they are validated.
- The repository's packaging smoke test workflow exercises editable-install resolution and metadata checks across Python 3.11, 3.12, 3.13, and 3.14 on GitHub Actions macOS runners.
- Benchmark reproducibility still depends on access to upstream `irodori_tts`, model weights, and Apple Silicon hardware.
