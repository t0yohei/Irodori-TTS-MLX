# DACVAE encode parity fixtures

Issue #185 tracks real encode parity evidence for the Semantic-DACVAE MLX codec
port under parent epic #169. The fixture is intentionally a fixed reference
WAV plus a locally produced MLX codec `.npz`; this repository does not commit
upstream codec weights, converted codec weights, generated latents, reference
audio, Hugging Face cache contents, or other heavyweight derived assets.

## Current validation status

The current real-weight validation run completed on 2026-05-18 against the
public `Aratako/Semantic-DACVAE-Japanese-32dim` checkpoint revision
`47376ee24834d7a05a48ebabfe3cde29b3c5e214`. The local MLX codec artifact was
converted from `weights.pth` with `scripts/convert_dacvae_decoder.py`, using
`facebookresearch/dacvae` revision
`414c20785fc3a28373073ea8ef7a1316eeeaca6e`.

The fixed license-clean reference WAV is a generated 0.5 second, 48 kHz, mono
440 Hz sine fixture. It is not committed to the repository.

The report completed with `run.status = "complete"` and
`comparison.status = "passed"`:

- latent shape: `[1, 13, 32]` for MLX encode
- finite-value checks: passed for the MLX latent tensor
- length/mask contract: `hop_length = 1920`,
  `latent_steps = speaker_mask_true_count = 13`

The generated local report remains outside git with the converted codec
artifact and latent fixtures because those files are derived from upstream
codec weights and local fixture paths.

## What the check validates

`scripts/check_dacvae_encode_parity.py` loads one fixed reference WAV and
encodes it through the local MLX encode artifact via `MLXDACVAEBridge`, and
validates:

- codec sample rate, hop length, and latent dimension metadata
- runtime latent shape `(B, T, D)`
- latent-step length and the equivalent speaker-mask true count used by the
  MLX generation runtime
- finite float32 latent values

This is an MLX-only artifact evidence check. The public runtime no longer keeps
the upstream PyTorch DACVAE bridge as a fallback path, so this script does not
import `irodori_tts.codec` or `torch`.

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

Run the check after producing a local MLX DACVAE artifact that includes the executable
`dacvae_encoder_exec/` Semantic-DACVAE encoder tensors:

```bash
python scripts/check_dacvae_encode_parity.py \
  --audio-wav /tmp/irodori-dacvae-encode-fixtures/reference.wav \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir /tmp/irodori-dacvae-encode-fixtures/parity \
  --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
  --codec-device cpu \
  --expected-sample-rate 48000 \
  --expected-hop-length 1920 \
  --expected-latent-dim 32
```

When a local checkpoint, converted codec artifact, or optional runtime
dependency is unavailable, add `--allow-partial` to write the same JSON report
shape with `comparison.status = "partial"` and exit successfully for evidence
collection without claiming the artifact check passed.

The command writes:

- `mlx-encode-latents.npy`
- `dacvae-encode-parity.json`

The JSON report records issue links, source audio stats, codec metadata,
latent output paths, length/mask contract, and pass/fail checks.
Keep those generated files local unless their license/provenance has been
reviewed for redistribution.

The report uses a portable status contract:

- `passed`: MLX encode ran and all configured checks passed.
- `failed`: MLX encode ran, but shape, finite, or metadata checks failed.
  Keep this report as measured evidence before changing the artifact.
- `partial`: preflight could not reach comparison because local artifacts or
  optional runtime dependencies are absent. The JSON report records the missing
  codec/audio/dependency inputs without writing fake parity artifacts or
  treating runtime metric failures as partial.

## Preprocessing caveats

The check passes `--max-seconds`, `--normalize-db`, and `--ensure-max` into
the MLX runtime path. The artifact evidence covers mono loading, resampling to
the codec sample rate, optional loudness normalization, peak safety,
hop-multiple padding, deterministic mean-latent encode, and runtime layout
transposition to `(B, T, D)`.

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
validation through environment variables. For issue #185 encode evidence, the
required variables are the codec artifact, reference WAV fixture, and expected
encoded latent fixture:

```bash
IRODORI_MLX_DACVAE_CODEC_NPZ=/path/to/dacvae-codec.npz \
IRODORI_MLX_DACVAE_ENCODE_AUDIO_WAV=/path/to/reference.wav \
IRODORI_MLX_DACVAE_ENCODE_LATENTS_NPY=/path/to/expected-encode-latents.npy \
python -m pytest tests/test_dacvae_mlx_parity_fixtures.py
```

Decode-only parity remains documented separately in
[dacvae_decode_parity.md](dacvae_decode_parity.md).
