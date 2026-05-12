# PyTorch vs MLX parity tests

`tests/test_pytorch_parity.py` compares deterministic MLX component outputs against the upstream Irodori-TTS PyTorch implementation.

## Local command

These tests need optional local dependencies that are intentionally not required for the default unit-test path:

- `torch`
- `mlx`
- `numpy`
- a local checkout of upstream `Aratako/Irodori-TTS`

Run them with:

```bash
IRODORI_TTS_UPSTREAM_PATH=/path/to/Irodori-TTS \
  python3 -m unittest tests.test_pytorch_parity -v
```

If `IRODORI_TTS_UPSTREAM_PATH` is omitted, the test walks upward from the repository checkout and uses the first existing `_scratch/Irodori-TTS-upstream` directory it finds. In the OpenClaw worktree layout this resolves to:

```text
/Users/kouka/.openclaw/workspace/repos/_scratch/Irodori-TTS-upstream
```

When PyTorch, MLX, or the upstream checkout is unavailable, the parity tests are skipped rather than failing the regular test suite.

## Coverage

The deterministic fixtures currently compare:

- Core formulas/layers:
  - RoPE frequency generation
  - RoPE application
  - timestep embeddings
  - `RMSNorm`
  - `SwiGLU`
  - `LowRankAdaLN`
  - sequence patching and mask reduction
- Encoder components:
  - `SelfAttention`
  - `TextEncoder`
  - `ReferenceLatentEncoder`
  - `ConditionEncoders` against upstream `TextToLatentRFDiT.encode_conditions`

The fixtures fill matching PyTorch and MLX parameters from deterministic arrays keyed by parameter name. On failure, assertions report shape, tolerance, max absolute difference, max-difference index, and the mismatching values.

## Tolerances

The tests use float32 inputs and deterministic small shapes. Default tolerances are intentionally tight:

- Most layer and encoder comparisons: `rtol=2e-5..5e-5`, `atol=2e-5..5e-5`
- Pure RoPE frequency/application comparisons: `rtol=1e-6`, `atol=1e-6`
- Timestep embeddings: `rtol=5e-6`, `atol=5e-6` because PyTorch and MLX can differ by a few e-6 in the exponential frequency path.

These tolerances are meant to catch mapping and formula mistakes, not to certify checkpoint-level audio parity.

## Current caveat

This branch targets the components present on `main`. DiT block and full RF-DiT forward parity should be added when the MLX RF-DiT forward implementation is merged into `main`.
