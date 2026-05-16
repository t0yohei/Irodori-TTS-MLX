# DACVAE decode parity validation

Issue #152 tracks the real decode parity gate for the v0.2 MLX DACVAE work
under parent epic #160. The validation input is intentionally a fixed latent
`.npy` file plus a locally produced MLX codec `.npz`; this repository does not
commit upstream codec weights, converted codec weights, decoded audio, Hugging
Face cache contents, or other heavyweight derived assets.

## What the check compares

`scripts/check_dacvae_decode_parity.py` loads one fixed runtime-layout latent
fixture shaped `(1, T, 32)`, decodes it through:

- upstream `irodori_tts.codec.DACVAECodec` via the PyTorch bridge
- the local MLX decode artifact via `MLXDACVAEBridge`

It then compares:

- waveform sample rate and shape
- float32 finite/range contract
- max absolute error, mean absolute error, RMSE, and cosine similarity

Default pass/fail tolerances are:

- `max_abs <= 5e-3`
- `mean_abs <= 1e-3`
- `rmse <= 2e-3`
- `cosine >= 0.999`

These tolerances are explicit command-line defaults, not a claim that every
future converted codec artifact must already meet them. If a real artifact
drifts, keep the failed report and document the observed metrics in the PR or
issue before adjusting thresholds.

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

Run the parity check after installing upstream Irodori-TTS and producing a
local MLX DACVAE decode artifact:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
python scripts/check_dacvae_decode_parity.py \
  --latents-npy /tmp/irodori-dacvae-decode-fixtures/decode-latents.npy \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae-decode-fixtures/parity \
  --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
  --codec-device cpu
```

The command writes:

- `upstream-decode.wav`
- `mlx-decode.wav`
- `dacvae-decode-parity.json`

The JSON report records issue links, fixture shape, codec metadata, output paths,
tolerances, and pass/fail metrics. A complete run has `run.status: complete`
and `comparison.status: passed` or `failed`. Keep generated WAVs and local
artifacts out of git unless their license/provenance has been reviewed for
redistribution.

For CI or developer machines that should record deterministic skip evidence when
local artifacts are absent, add `--allow-partial`:

```bash
python scripts/check_dacvae_decode_parity.py \
  --latents-npy /tmp/irodori-dacvae-decode-fixtures/decode-latents.npy \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae-decode-fixtures/parity \
  --allow-partial
```

When preflight detects that the latent fixture, MLX codec artifact, MLX runtime,
or PyTorch/upstream dependency is missing, the command writes
`dacvae-decode-parity.json` with `run.status: partial` and exits 0 only with
`--allow-partial`. Without `--allow-partial`, partial runs exit 2. Runtime
decode/write failures, shape mismatches, metadata mismatches, sample-rate
mismatches, and metric drift still write a failed report and exit non-zero.

## Lightweight test coverage

The checked-in unit tests use synthetic latents and mocked decode bridges so
local development can validate the contract without downloading upstream assets:

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
