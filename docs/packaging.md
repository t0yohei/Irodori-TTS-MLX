# Packaging and reproducible environments

Issue: [#31 Package the project for reproducible runtime and benchmark environments](https://github.com/t0yohei/Irodori-TTS-MLX/issues/31)

The repository now exposes a project-level `pyproject.toml` so package users and contributors can install dependencies by use case instead of guessing from ad hoc notes.

The next package release candidate is `0.3.0a1`. It is still alpha software:
the installed console scripts are intended for v0.3 alpha consumers, while the
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
- `.[runtime]`: standalone MLX runtime WAV generation with hosted/local DACVAE codec artifacts; does not install `torch` or `torchaudio`
- `.[bench]`: benchmark-oriented environment for MLX artifact benchmarks and checkpoint conversion helpers
- `.[dev]`: local contributor environment for tests plus packaging validation helpers

On Python 3.11, the `runtime`, `bench`, and `dev` extras intentionally use the
`sentencepiece>=0.1.99,<0.2` range used by the audited artifacts. Python 3.12 and newer use `sentencepiece>=0.2,<1` because
`sentencepiece==0.1.99` does not publish wheels for the advertised newer
Python packaging targets.

The standalone MLX runtime path is artifact-driven through the default approved hosted DACVAE codec artifact, `--codec-artifact-dir`, or `--codec-path`. Keep PyTorch bridge dependencies out of `.[runtime]` so clean public installs can generate with approved hosted artifacts without upstream `irodori-tts`.

## Package users versus repository contributors

Package users should install a built wheel or source distribution into a clean virtual environment and use installed console scripts:

```bash
python3.11 -m venv .venv-irodori
. .venv-irodori/bin/activate
python -m pip install --upgrade pip
python -m pip install /path/to/irodori_tts_mlx-0.3.0a1-py3-none-any.whl
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

The v0.3 alpha public runtime defaults to full-MLX codec artifact mode and does not require upstream `irodori_tts.codec.DACVAECodec`.
The old PyTorch bridge fallback modes are no longer public generation runtime modes.
This is intentional: this MLX repo owns the text/caption conditioning, RF-DiT, converted-weight runtime, duration handling, sampler path, and artifact-backed DACVAE runtime.

The v0.3 alpha `--codec-runtime-mode mlx` path is artifact-driven: package users can use the default approved hosted codec artifact, `--codec-artifact-dir`, or `--codec-path /path/to/dacvae-codec.npz`. The repository does not ship Semantic-DACVAE codec weights.
See [upstream_dependency.md](upstream_dependency.md) for the full responsibility split and import-failure guidance.

## Reproducible runtime setup

For the current standalone MLX runtime, the minimal practical setup is:

```bash
python3.11 -m venv .venv  # or: python3.12/3.13/3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]"
```

Then run the installed generation command:

```bash
irodori-tts-generate \
  --weights /path/to/irodori-tts-500m-v2.npz \
  --ref-wav /path/to/reference.wav \
  --text "こんにちは。今日は良い天気です。" \
  --output-wav /tmp/irodori.wav \
  --seconds 5 \
  --num-steps 40 \
  --codec-runtime-mode mlx \
  --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec
```

For repository development, invoke scripts directly from a checkout:

```bash
python scripts/generate_wav.py --help
```

## Reproducible benchmark setup

A benchmark-oriented environment should include the benchmark extra. The example keeps Python 3.11 because that is the benchmark reference environment used by the current reports, even though packaging support now extends through Python 3.14.

```bash
python3.11 -m venv .venv-bench311
. .venv-bench311/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[bench]"
```

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

## v0.3 alpha release artifact checklist

Do not publish or upload packages until the v0.3 alpha release decision is explicit. For a local or CI release-candidate validation, run:

```bash
python -m pip install --upgrade pip
python -m pip install build
python -m build --wheel --sdist --outdir dist

python -m venv .venv-wheel-smoke
.venv-wheel-smoke/bin/python -m pip install --upgrade pip
.venv-wheel-smoke/bin/python -m pip install --no-deps dist/irodori_tts_mlx-0.3.0a1-py3-none-any.whl
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
