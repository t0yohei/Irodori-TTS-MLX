# Packaging and reproducible environments

Issue: [#31 Package the project for reproducible runtime and benchmark environments](https://github.com/t0yohei/Irodori-TTS-MLX/issues/31)

The repository now exposes a project-level `pyproject.toml` so package users and contributors can install dependencies by use case instead of guessing from ad hoc notes.

The next package release candidate is `0.2.0a1`. It is still alpha software:
the installed console scripts are intended for v0.2 consumers, while the
`irodori_mlx` modules, top-level exports, and `scripts.*` imports are internal
implementation details and are not a stable public Python API.

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

On Python 3.11, the `runtime`, `bench`, and `dev` extras intentionally use the
upstream-compatible `sentencepiece>=0.1.99,<0.2` range so a single venv can
install both this package and upstream `irodori-tts` without a resolver
conflict. On Python 3.12 and newer, those extras use `sentencepiece>=0.2,<1`
because `sentencepiece==0.1.99` does not publish wheels for the advertised
newer Python packaging targets; use Python 3.11 for same-venv upstream installs.

There is intentionally no separate codec-only extra yet. The v0.2 MLX codec path is artifact-driven through `--codec-runtime-mode mlx`, `--codec-runtime-mode mlx-decode`, and `--codec-path`; keep a future codec-only dependency split blocked until the redistributed DACVAE artifact contract is approved.

## Package users versus repository contributors

Package users should install a built wheel or source distribution into a clean virtual environment and use installed console scripts:

```bash
python3.11 -m venv .venv-irodori
. .venv-irodori/bin/activate
python -m pip install --upgrade pip
python -m pip install /path/to/irodori_tts_mlx-0.2.0a1-py3-none-any.whl
irodori-tts-generate --help
```

Package users should not import `irodori_mlx` modules or `scripts.*` modules as
a supported integration boundary yet. Those imports are for the CLI, tests, and
repository-local development, and may change without deprecation while the
project is alpha. See [public_api_stability.md](public_api_stability.md).

Repository contributors should install from a checkout in editable mode so tests and script changes are exercised directly:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

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

The v0.1 runtime still reuses upstream `irodori_tts.codec.DACVAECodec` for reference encode / waveform decode.
That means the local environment must also be able to import upstream `irodori_tts`.
This is intentional: this MLX repo owns the text/caption conditioning, RF-DiT, converted-weight runtime, duration handling, and sampler path, while upstream still owns the PyTorch DACVAE codec boundary.
A full MLX DACVAE port is not required for v0.1 WAV generation.

The v0.2 `--codec-runtime-mode mlx` path is artifact-driven: package users must provide `--codec-path /path/to/dacvae-codec.npz`, and the repository does not ship Semantic-DACVAE codec weights. The local contract keeps encode/decode math in MLX and is intended for converted codec artifacts and parity fixtures; default runtime packaging should continue to include the PyTorch bridge dependencies until a real redistributed codec artifact is approved.
See [upstream_dependency.md](upstream_dependency.md) for the full responsibility split and import-failure guidance.

Supported ways to provide upstream:

### Option A: install the upstream checkout into the same venv (recommended)

```bash
git clone https://github.com/Aratako/Irodori-TTS.git /path/to/Irodori-TTS
python -m pip install -e /path/to/Irodori-TTS
```

### Option B: leave upstream uninstalled and expose it with `PYTHONPATH`

```bash
export PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}
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

Then run the installed generation command:

```bash
irodori-tts-generate \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --reference-wav /path/to/reference.wav \
  --text "こんにちは。今日は良い天気です。" \
  --output /tmp/irodori.wav \
  --seconds 5 \
  --num-steps 40 \
  --codec-device cpu
```

For repository development, invoke scripts directly from a checkout:

```bash
python scripts/generate_wav.py --help
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

## CLI entry points and local scripts

Installed environments expose stable command names for the primary runtime surface:

- `irodori-tts-generate` for WAV generation
- `irodori-tts-convert` for checkpoint conversion
- `irodori-tts-convert-dacvae-codec` for Semantic-DACVAE codec conversion inspection/blocker reports
- `irodori-tts-convert-dacvae-decoder` for Semantic-DACVAE decoder artifact conversion
- `irodori-tts-inspect` for checkpoint inspection

Smoke-check them with:

```bash
irodori-tts-generate --help
irodori-tts-convert --help
irodori-tts-convert-dacvae-codec --help
irodori-tts-convert-dacvae-decoder --help
irodori-tts-inspect --help
irodori-tts-adapt-mlx-audio --help
```

The existing repository scripts remain supported repo-local entry points:

- `python scripts/generate_wav.py`
- `python scripts/benchmark.py`
- `python scripts/convert_weights.py`
- `python scripts/convert_dacvae_codec.py`
- `python scripts/convert_dacvae_decoder.py`
- `python scripts/inspect_checkpoint.py`

They continue to work with the packaged dependency layout because the repo is installed editable (`-e`) and each script already resolves the repository root when importing local modules.

## Current limitations

- The packaging surface is still prototype-grade; there is no stable API guarantee yet.
- The supported alpha user surface is the installed console scripts and documented artifact contracts, not Python module imports.
- Python 3.11 through 3.14 are the currently supported packaged environments; newer Python versions should stay unsupported until they are validated.
- The repository's packaging smoke test workflow exercises metadata checks, wheel/sdist builds, clean wheel installation, installed console-script help checks, and editable-install resolution across Python 3.11, 3.12, 3.13, and 3.14 on GitHub Actions macOS runners.
- Benchmark reproducibility still depends on access to upstream `irodori_tts`, model weights, and Apple Silicon hardware.

## v0.2 release artifact checklist

Do not publish or upload packages until the v0.2 release decision is explicit. For a local or CI release-candidate validation, run:

```bash
python -m pip install --upgrade pip
python -m pip install build
python -m build --wheel --sdist --outdir dist

python -m venv .venv-wheel-smoke
.venv-wheel-smoke/bin/python -m pip install --upgrade pip
.venv-wheel-smoke/bin/python -m pip install --no-deps dist/irodori_tts_mlx-0.2.0a1-py3-none-any.whl
.venv-wheel-smoke/bin/irodori-tts-generate --help
.venv-wheel-smoke/bin/irodori-tts-convert --help
.venv-wheel-smoke/bin/irodori-tts-inspect --help
.venv-wheel-smoke/bin/irodori-tts-adapt-mlx-audio --help
```

Before opening a release PR or tag:

- confirm `pyproject.toml` has the intended prerelease version and not `0.0.0`;
- confirm `dist/` contains exactly one wheel and one sdist for the candidate version;
- confirm the wheel smoke checks run from the clean environment, not from an editable checkout;
- confirm optional runtime gates that need upstream `irodori_tts`, PyTorch, model checkpoints, or DACVAE artifacts are either run in the appropriate environment or documented as unavailable for the packaging-only gate;
- keep generated `dist/`, virtual environments, Hugging Face caches, model weights, converted `.npz` archives, reference audio, and generated audio out of git.
