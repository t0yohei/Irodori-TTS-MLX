# Public API stability boundary

Issue: [#192 Decide and document the public API stability boundary](https://github.com/t0yohei/Irodori-TTS-MLX/issues/192)

This repository is alpha software and currently supports a CLI-first contract.
For downstream users, the stable-ish user surface is limited to:

- installed console scripts declared in `pyproject.toml`
- command-line flags and JSON outputs documented in the README and docs
- documented hosted/local artifact layouts, manifests, metadata, and checksum files
- package metadata needed to install wheels or sdists on the supported Python versions

The Python modules are intentionally importable so the console scripts, tests,
and repository-local tools can share implementation code. They are not a stable public Python API yet.

## Supported console scripts

The alpha user contract covers these installed commands:

- `irodori-tts-generate`
- `irodori-tts-convert`
- `irodori-tts-convert-dacvae-codec`
- `irodori-tts-convert-dacvae-decoder`
- `irodori-tts-inspect`
- `irodori-tts-adapt-mlx-audio`

Use `--help` on each command for the current argument surface. Normal package
usage should call these console scripts instead of importing `scripts.*` from
Python.
The documented artifact layouts are part of this CLI contract where the README
or docs explicitly describe manifests, checksums, metadata, and allowed files.

## Unsupported Python API

No `irodori_mlx` module, class, function, dataclass, or top-level
`irodori_mlx.__all__` export is currently supported as a stable public Python
API. That includes runtime helpers such as `MLXRuntimeConfig`,
`GenerationRequest`, `load_mlx_model`, sampler/model/layer classes, DACVAE
helpers, hosted artifact helpers, and repository `scripts.*` modules.

Downstream integrations may experiment with those imports, but they should
treat them as internal implementation details. They can change, move, be
renamed, or be removed in any alpha release without deprecation.

## Future Python API gate

A future public Python API should be added only after an explicit design issue
lists the supported modules, classes, functions, data contracts, versioning
policy, and migration/deprecation expectations. Until that happens, examples
and packaging metadata should remain console-script oriented.
