# Upstream vs MLX Generation Parity Harness

Issue: [#109](https://github.com/t0yohei/Irodori-TTS-MLX/issues/109)
Parent epic: [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

`scripts/run_upstream_parity.py` creates a JSON report for comparing an upstream PyTorch `infer.py` run with this repository's MLX `scripts/generate_wav.py` run under the same prompt, seed, sampling, conditioning, tokenizer, duration, and codec settings.

The harness is intentionally small for the first #109 slice:

- it records rerunnable upstream and MLX commands
- it captures scenario metadata for tokenizer, duration, sampling, and codec boundaries
- it records lightweight WAV properties when generated audio exists
- it classifies reports as `expected_drift`, `regression`, or `not_comparable`
- it supports deterministic fixture mode for CI and schema coverage without checkpoints

Full VoiceDesign/v3 baseline matrices, intermediate tensor comparisons, and richer audio metrics are deferred to the follow-up issues linked from #109.

## Contract / Fixture Command

Use fixture mode when upstream dependencies or real checkpoints are unavailable:

```bash
python scripts/run_upstream_parity.py \
  --fixture \
  --scenario v3-no-reference \
  --output-dir parity-runs/fixture-v3 \
  --json
```

This writes `parity-runs/fixture-v3/v3-no-reference.parity.json` with deterministic evidence. It does not download weights, import upstream `irodori_tts`, or generate audio.

## Real v3 Command

Prepare the upstream checkout and converted MLX checkpoint first:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git external/Irodori-TTS
cd external/Irodori-TTS
uv sync
```

Convert the v3 checkpoint with the existing quickstart flow in [README.md](../README.md), keeping:

- converted MLX weights, for example `/tmp/irodori-quickstart/irodori-v3.npz`
- model config JSON, for example `/tmp/irodori-quickstart/v3-model-config.json`

Then run both sides and write one report:

```bash
PYTHONPATH="$(pwd)/external/Irodori-TTS:${PYTHONPATH:-}" \
python scripts/run_upstream_parity.py \
  --scenario v3-no-reference \
  --run-upstream \
  --run-mlx \
  --upstream-root external/Irodori-TTS \
  --mlx-weights /tmp/irodori-quickstart/irodori-v3.npz \
  --mlx-model-config-json /tmp/irodori-quickstart/v3-model-config.json \
  --output-dir parity-runs/v3-real \
  --codec-device cpu \
  --num-steps 8 \
  --seed 20260516 \
  --json
```

The report records the exact upstream `uv run python infer.py ...` command and the exact MLX `scripts/generate_wav.py ...` command. Generated WAVs and caches remain under `parity-runs/` or your chosen local output directory and should not be committed.

## Real VoiceDesign Command

VoiceDesign uses the same harness but requires caption conditioning and currently uses manual duration:

```bash
PYTHONPATH="$(pwd)/external/Irodori-TTS:${PYTHONPATH:-}" \
python scripts/run_upstream_parity.py \
  --scenario voicedesign-no-reference \
  --run-upstream \
  --run-mlx \
  --upstream-root external/Irodori-TTS \
  --mlx-weights /tmp/irodori-voicedesign/irodori-voicedesign.npz \
  --mlx-model-config-json /tmp/irodori-voicedesign/voicedesign-model-config.json \
  --output-dir parity-runs/voicedesign-real \
  --codec-device cpu \
  --seconds 2.0 \
  --num-steps 8 \
  --seed 20260516 \
  --json
```

## Report Schema

Top-level fields:

- `schema_version`: currently `1`
- `scenario`: prompt, checkpoint family, checkpoint id, reference/caption settings, seed, sampling, duration, tokenizer, and codec settings
- `metadata_axes`: normalized diagnostic axes for tokenizer, duration, sampling, and codec differences
- `upstream`: command, status, command result when run, WAV properties when present
- `mlx`: command, status, command result when run, WAV properties and `generate_wav.py` metadata when present
- `comparison`: `expected_drift`, `regression`, or `not_comparable`
- `deferred_scope`: intentionally omitted work for follow-up issues

The first comparison check is conservative. It treats bit-level waveform mismatch as expected drift, flags sample-rate mismatch as a regression, and flags duration deltas larger than 250 ms as a regression.
