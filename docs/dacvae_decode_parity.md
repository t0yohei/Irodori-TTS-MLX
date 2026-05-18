# DACVAE decode parity validation

Issue #184 tracks the current real decode parity gate for the executable MLX DACVAE
decoder work under parent epic #169. The validation input is intentionally a fixed latent
`.npy` file plus a locally produced MLX codec `.npz`; this repository does not
commit upstream codec weights, converted codec weights, decoded audio, Hugging
Face cache contents, or other heavyweight derived assets.

## Current validation status

The 2026-05-18 real-artifact run passed with `run.status: complete` and
`comparison.status: passed`. The checked-in JSON report is
[parity-reports/2026-05-18-dacvae-decode-parity.json](parity-reports/2026-05-18-dacvae-decode-parity.json).

That run used:

- upstream codec repo: `Aratako/Semantic-DACVAE-Japanese-32dim`
- upstream codec snapshot: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- local converted codec artifact format: `irodori-tts-mlx-dacvae-codec` `0.2`
- fixed latent fixture seed: `20260518`, shape `(1, 8, 32)`
- `dacvae==1.0.0`, `codec_device=cpu`, watermark disabled

Observed metrics:

- `max_abs = 3.0517578125e-05`
- `mean_abs = 8.145968166672901e-08`
- `rmse = 1.5766902379255043e-06`
- `cosine = 1.0`

This validates the fixed real decode artifact and fixture above. It does not
redistribute the artifact or claim broad acoustic parity for arbitrary future
codec artifacts without rerunning this check.

## What the check compares

`scripts/check_dacvae_decode_parity.py` loads one fixed runtime-layout latent
fixture shaped `(1, T, 32)`, decodes it through the local MLX decode artifact
via `MLXDACVAEBridge`, and validates:

- decoded waveform sample rate
- float32 finite/range contract

This is an MLX-only artifact evidence check. The public runtime no longer keeps
the upstream PyTorch DACVAE bridge as a fallback path, so this script does not
import `irodori_tts.codec` or `torch`.

The fixture must use runtime latent layout `(1, T, 32)` by default. The script
accepts `--expected-sample-rate`, `--expected-hop-length`, and
`--expected-latent-dim` for synthetic tests, but real Semantic-DACVAE decode
evidence should leave the defaults `48000`, `1920`, and `32` in place.

## Local real-artifact command

Create or reuse a small latent fixture outside the repository, for example:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

out = Path("/tmp/irodori-dacvae-decode-fixtures")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260516)
latents = rng.normal(0.0, 0.35, size=(1, 8, 32)).astype("float32")
np.save(out / "decode-latents.npy", latents)
PY
```

Run the check after producing a local MLX DACVAE decode artifact:

```bash
python scripts/check_dacvae_decode_parity.py \
  --latents-npy /tmp/irodori-dacvae-decode-fixtures/decode-latents.npy \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae-decode-fixtures/parity \
  --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
  --codec-device cpu \
  --expected-sample-rate 48000 \
  --expected-hop-length 1920 \
  --expected-latent-dim 32
```

The command writes:

- `mlx-decode.wav`
- `dacvae-decode-parity.json`

The JSON report records issue links, fixture shape, codec metadata, output paths,
and pass/fail checks. A complete run has `run.status: complete` and
`comparison.status: passed` or `failed`. Keep generated WAVs and local
artifacts out of git unless their license/provenance has been reviewed for
redistribution.

A passing report is the gate for describing a specific executable decoder
artifact as checked. Runtime capability output may still say
`has_executable_mlx_decode=true` before a report exists, but docs and release
notes should only call a converted artifact checked when its current report is
complete and passed. In short, the report is complete and passed.

For CI or developer machines that should record deterministic skip evidence when
local artifacts are absent, add `--allow-partial`:

```bash
python scripts/check_dacvae_decode_parity.py \
  --latents-npy /tmp/irodori-dacvae-decode-fixtures/decode-latents.npy \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae-decode-fixtures/parity \
  --allow-partial
```

When preflight detects that the latent fixture, MLX codec artifact, or MLX
runtime is missing, the command writes
`dacvae-decode-parity.json` with `run.status: partial` and exits 0 only with
`--allow-partial`. Without `--allow-partial`, partial runs exit 2. Runtime
decode/write failures, shape mismatches, metadata mismatches, sample-rate
mismatches, and metric drift still write a failed report and exit non-zero.

## Lightweight test coverage

The checked-in unit tests use synthetic latents and mocked decode bridges so
local development can validate the contract without downloading external assets:

```bash
python -m pytest tests/test_check_dacvae_decode_parity_script.py tests/test_dacvae_mlx_parity_fixtures.py
```

The script-level real-artifact test is skipped unless the required local files
are provided through environment variables:

```bash
IRODORI_MLX_DACVAE_CODEC_NPZ=/path/to/dacvae-codec.npz \
IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY=/path/to/decode-latents.npy \
python -m pytest tests/test_check_dacvae_decode_parity_script.py -k real_decode
```

`tests/test_dacvae_mlx_parity_fixtures.py` remains available for lower-level
MLX bridge fixture validation through environment variables. For decode-only
fixture evidence, the required variables are the codec artifact, latent fixture,
and upstream decoded audio fixture:

```bash
IRODORI_MLX_DACVAE_CODEC_NPZ=/path/to/dacvae-codec.npz \
IRODORI_MLX_DACVAE_DECODE_LATENTS_NPY=/path/to/decode-latents.npy \
IRODORI_MLX_DACVAE_DECODE_AUDIO_NPY=/path/to/upstream-decode-audio.npy \
python -m pytest tests/test_dacvae_mlx_parity_fixtures.py
```

Encode parity is documented separately in
[dacvae_encode_parity.md](dacvae_encode_parity.md). Keep fixed-audio encode
evidence separate from this decode-only report so the two fixture contracts can
fail independently.
