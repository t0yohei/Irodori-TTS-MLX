# DACVAE encode parity fixtures

Issue #174 tracks encode parity evidence for the Semantic-DACVAE MLX codec port
under parent epic #169. The fixture is intentionally a fixed reference
WAV plus a locally produced MLX codec `.npz`; this repository does not commit
upstream codec weights, converted codec weights, generated latents, reference
audio, Hugging Face cache contents, or other heavyweight derived assets.

## What the check compares

`scripts/check_dacvae_encode_parity.py` loads one fixed reference WAV and
encodes it through:

- upstream `irodori_tts.codec.DACVAECodec` via the PyTorch bridge
- the local MLX encode artifact via `MLXDACVAEBridge`

It then compares:

- codec sample rate, hop length, and latent dimension metadata
- runtime latent shape `(B, T, D)`
- latent-step length and the equivalent speaker-mask true count used by the
  MLX generation runtime
- finite float32 latent values
- max absolute error, mean absolute error, RMSE, and cosine similarity

Default pass/fail tolerances are:

- `max_abs <= 1e-3`
- `mean_abs <= 2e-4`
- `rmse <= 5e-4`
- `cosine >= 0.999`

These defaults are intentionally visible command-line settings. If a real
converted codec artifact drifts, keep the failed report and document the
observed metrics in the PR or issue before changing thresholds.

## Local fixture command

Create or reuse a short reference WAV outside the repository. The fixture should
be license-clean for local validation and small enough to encode quickly:

```bash
python - <<'PY'
from pathlib import Path
import math
import wave
import numpy as np

out = Path("/tmp/irodori-dacvae-encode-fixtures")
out.mkdir(parents=True, exist_ok=True)
sample_rate = 48000
seconds = 0.5
t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
samples = 0.2 * np.sin(2.0 * math.pi * 440.0 * t)
pcm = np.clip(samples, -1.0, 1.0)
pcm = (pcm * 32767.0).astype("<i2")
with wave.open(str(out / "reference.wav"), "wb") as fh:
    fh.setnchannels(1)
    fh.setsampwidth(2)
    fh.setframerate(sample_rate)
    fh.writeframes(pcm.tobytes())
PY
```

Run the parity check after installing upstream Irodori-TTS and producing a
local MLX DACVAE artifact that includes the executable
`dacvae_encoder_exec/` Semantic-DACVAE encoder tensors:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
python scripts/check_dacvae_encode_parity.py \
  --audio-wav /tmp/irodori-dacvae-encode-fixtures/reference.wav \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae-encode-fixtures/parity \
  --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
  --codec-device cpu \
  --expected-latent-dim 32
```

When a local checkpoint, converted codec artifact, or optional runtime
dependency is unavailable, add `--allow-partial` to write the same JSON report
shape with `comparison.status = "partial"` and exit successfully for evidence
collection without claiming parity passed.

The command writes:

- `upstream-encode-latents.npy`
- `mlx-encode-latents.npy`
- `dacvae-encode-parity.json`

The JSON report records issue links, source audio stats, codec metadata,
latent output paths, length/mask contract, tolerances, and pass/fail metrics.
Keep those generated files local unless their license/provenance has been
reviewed for redistribution.

The report uses a portable status contract:

- `passed`: upstream and MLX encode both ran, the comparison completed, and all
  configured checks passed.
- `failed`: upstream and MLX encode both ran, but shape, finite, or tolerance
  checks failed. Keep this report as measured parity evidence before changing
  thresholds.
- `partial`: preflight could not reach comparison because local artifacts or
  optional runtime dependencies are absent. The JSON report records the missing
  codec/audio/dependency inputs without writing fake parity artifacts or
  treating runtime metric failures as partial.

## Preprocessing caveats

The check passes the same `--max-seconds`, `--normalize-db`, and `--ensure-max`
options to both bridges. The upstream bridge still owns the reference behavior:
mono loading, resampling to the codec sample rate, optional loudness
normalization, peak safety, hop-multiple padding, deterministic mean-latent
encode, and runtime layout transposition to `(B, T, D)`.

The lightweight MLX codec artifact contract in this repository is a fixture
contract, not a redistributed Semantic-DACVAE checkpoint. Real acoustic parity
depends on a converted artifact produced from the supported upstream codec
weights and validated with fixed audio/latent fixtures.

## Lightweight test coverage

The checked-in unit tests use synthetic latents and mocked encode bridges so
local development can validate the contract without downloading upstream
assets:

```bash
python -m pytest tests/test_check_dacvae_encode_parity_script.py tests/test_dacvae_mlx_parity_fixtures.py
```

`tests/test_dacvae_mlx_parity_fixtures.py` remains available for real fixture
validation through environment variables. For issue #174 encode evidence, the
required variables are the codec artifact, reference WAV fixture, and upstream
encoded latent fixture:

```bash
IRODORI_MLX_DACVAE_CODEC_NPZ=/path/to/dacvae-codec.npz \
IRODORI_MLX_DACVAE_ENCODE_AUDIO_WAV=/path/to/reference.wav \
IRODORI_MLX_DACVAE_ENCODE_LATENTS_NPY=/path/to/upstream-encode-latents.npy \
python -m pytest tests/test_dacvae_mlx_parity_fixtures.py
```

Decode-only parity remains documented separately in
[dacvae_decode_parity.md](dacvae_decode_parity.md).
