# 2026-05-16 upstream-vs-MLX parity harness baseline

Issues: [#121](https://github.com/t0yohei/Irodori-TTS-MLX/issues/121), [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

This is the current checked-in baseline for the end-to-end upstream PyTorch vs
MLX generation parity harness. It records reproducible fixture and setup-audit
evidence for v0.2 without committing heavyweight upstream artifacts,
generated WAVs, Hugging Face caches, converted `.npz` weights, or reference
audio.

## Evidence summary

| Scenario | Command class | Expected status | Current result |
| --- | --- | --- | --- |
| `v3-no-reference` | fixture contract | `complete` | pass; upstream and MLX sides are deterministic fixtures, 24000 Hz mono, 1.5 s |
| `voicedesign-contrastive-caption` | fixture contract | `complete` | pass; caption/CFG metadata recorded, manual 2.0 s duration, 24000 Hz mono |
| `v3-no-reference` | partial setup audit without artifacts | `partial` | pass; upstream unavailable reason is `missing_upstream_root`, MLX unavailable reason is `missing_mlx_weights` |
| `voicedesign-contrastive-caption` | partial setup audit without artifacts | `partial` | pass; same explicit missing-artifact reasons, scenario metadata remains complete |

The fixture comparisons are classified as `expected_drift`. They prove report
shape, command construction, scenario metadata, WAV-property comparison, and
missing-artifact handling. They do not claim real acoustic parity.

## Reproduce fixture baselines

```bash
uv run python scripts/run_upstream_parity.py \
  --fixture \
  --scenario v3-no-reference \
  --output-dir parity-runs/baseline-fixture-v3 \
  --json

uv run python scripts/run_upstream_parity.py \
  --fixture \
  --scenario voicedesign-contrastive-caption \
  --output-dir parity-runs/baseline-fixture-voicedesign \
  --json
```

Expected report paths:

- `parity-runs/baseline-fixture-v3/v3-no-reference.parity.json`
- `parity-runs/baseline-fixture-voicedesign/voicedesign-contrastive-caption.parity.json`

## Reproduce partial setup audits

```bash
uv run python scripts/run_upstream_parity.py \
  --scenario v3-no-reference \
  --run-upstream \
  --run-mlx \
  --output-dir parity-runs/baseline-partial-v3 \
  --json

uv run python scripts/run_upstream_parity.py \
  --scenario voicedesign-contrastive-caption \
  --run-upstream \
  --run-mlx \
  --output-dir parity-runs/baseline-partial-voicedesign \
  --json
```

Expected report paths:

- `parity-runs/baseline-partial-v3/v3-no-reference.parity.json`
- `parity-runs/baseline-partial-voicedesign/voicedesign-contrastive-caption.parity.json`

On a clean machine without optional artifacts, the reports should stay
`partial`, not `failed`. The unavailable reasons should be machine-readable:

- upstream: `missing_upstream_root`
- MLX: `missing_mlx_weights`

## Real runtime boundary

Real upstream-vs-MLX audio execution needs:

- an `Aratako/Irodori-TTS` checkout with its upstream Python environment
- the public upstream checkpoint for the target family
- converted MLX `.npz` weights
- model config JSON for the converted checkpoint
- codec dependencies and local cache access

Those assets are intentionally local-only. When they are available, use the
real commands in [../upstream_parity_harness.md](../upstream_parity_harness.md)
and store generated JSON/WAV outputs outside git. When they are unavailable,
the partial setup audit above is the expected reproducible evidence.
