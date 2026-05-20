# Upstream vs MLX Generation Parity Harness

Issue: [#109](https://github.com/t0yohei/Irodori-TTS-MLX/issues/109)
Parent epic: [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

`scripts/run_upstream_parity.py` creates a JSON report for comparing an upstream PyTorch `infer.py` run with this repository's MLX `scripts/generate_wav.py` run under the same prompt, seed, sampling, conditioning, tokenizer, duration, and codec settings.

The harness is intentionally small for the first #109 slice:

- it records rerunnable upstream and MLX commands
- it captures scenario metadata for tokenizer, duration, sampling, and codec boundaries
- it records lightweight WAV properties and shareable audio metrics when generated audio exists
- it records compact intermediate checkpoints for tokenizer, duration, and sampled-latent shape/statistics when metadata is available
- it classifies reports as `expected_drift`, `regression`, or `not_comparable`
- it marks each report as `complete`, `partial`, or `failed`
- it records `unavailable` side states with a machine-readable reason when optional upstream checkouts or MLX weights are absent
- it supports deterministic fixture mode for CI and schema coverage without checkpoints

Full v3 baseline matrices and heavyweight perceptual audio metrics are deferred to the follow-up issues linked from #109.

Current baseline summary for the v0.2 delivery is recorded in
[baseline-reports/2026-05-16-upstream-mlx-parity-baseline.md](baseline-reports/2026-05-16-upstream-mlx-parity-baseline.md).
That report is the source of truth for the checked-in harness baseline: fixture
reports pass for the supported v3 and VoiceDesign scenarios, while real
upstream-vs-MLX audio execution remains partial on machines without an upstream
checkout and converted MLX checkpoint artifacts.

The current real-checkpoint Apple Silicon evidence is recorded in
[baseline-reports/2026-05-18-real-upstream-mlx-generation-parity.md](baseline-reports/2026-05-18-real-upstream-mlx-generation-parity.md).
That report summarizes v3 no-reference, VoiceDesign caption-conditioned, and v3
reference-audio generation with pinned RF-DiT artifacts plus a local DACVAE MLX
codec artifact.

## Setup Matrix

The harness has three useful setup levels:

| Setup | Required local assets | Expected report status |
| --- | --- | --- |
| Fixture contract | repository checkout only | `complete`, both sides `fixture` |
| Partial setup audit | repository checkout only, with `--run-upstream --run-mlx` but no artifacts | `partial`, unavailable reasons recorded |
| Real upstream-vs-MLX run | upstream `Aratako/Irodori-TTS` checkout, upstream dependencies, converted MLX `.npz`, model config JSON, codec dependencies/cache | `complete` when both commands pass |

The repository does not commit upstream checkouts, generated WAVs, Hugging Face
caches, converted `.npz` weights, model snapshots, or reference audio. Keep
those under `parity-runs/`, `/tmp`, or another ignored local path.

## Contract / Fixture Command

Use fixture mode when upstream dependencies or real checkpoints are unavailable:

```bash
uv run python scripts/run_upstream_parity.py \
  --fixture \
  --scenario v3-no-reference \
  --output-dir parity-runs/fixture-v3 \
  --json
```

This writes `parity-runs/fixture-v3/v3-no-reference.parity.json` with deterministic evidence. It does not download weights, import upstream `irodori_tts`, or generate audio.

The current v3 fixture baseline is:

- `report_status: "complete"`
- upstream side: `status: "fixture"`, sample rate 24000 Hz, duration 1.5 s
- MLX side: `status: "fixture"`, sample rate 24000 Hz, duration 1.5 s
- comparison: `status: "expected_drift"` because PyTorch and MLX generation are not expected to be bit-identical

## Partial Report Command

The runner can intentionally record only one side, or record why a requested side could not run. For example, this command asks for both sides but omits the external artifacts:

```bash
uv run python scripts/run_upstream_parity.py \
  --scenario v3-no-reference \
  --run-upstream \
  --run-mlx \
  --output-dir parity-runs/partial-v3 \
  --json
```

The JSON is still written with `report_status: "partial"`. The upstream side is marked `status: "unavailable"` with `availability.reason: "missing_upstream_root"`, and the MLX side is marked `status: "unavailable"` with `availability.reason: "missing_mlx_weights"`. This is the expected representation for CI or developer machines that do not have heavyweight checkpoints, caches, or an upstream checkout.

This partial report is not a failed harness run. It is the reproducible setup
audit used to prove that missing optional dependencies are represented
explicitly instead of causing ad hoc script errors.

## Real v3 Command

Issue [#119](https://github.com/t0yohei/Irodori-TTS-MLX/issues/119) adds the fixed `v3-reference-predicted` scenario. It uses a short Japanese text prompt, reference-audio speaker conditioning, fixed seed `20260519`, omitted `--seconds`, and `--num-steps 8` so the v3 duration predictor path stays visible in the report.

Use fixture mode for a schema-covered smoke run that does not need checkpoints, an upstream checkout, or the reference WAV:

```bash
python scripts/run_upstream_parity.py \
  --fixture \
  --scenario v3-reference-predicted \
  --output-dir parity-runs/fixture-v3-reference \
  --json
```

The fixture report records `duration_mode: "predicted"`, `requested_seconds: null`, synthetic predicted-duration details, runtime messages, reference-audio command arguments, and deterministic WAV properties for both sides. If you request a real `v3-reference-predicted` run without providing a readable reference WAV, the report remains `partial` and both sides are marked `unavailable` with `availability.reason: "missing_reference_wav"`.

Prepare the upstream checkout and converted MLX checkpoint first:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git external/Irodori-TTS
cd external/Irodori-TTS
uv sync
```

Convert the v3 checkpoint with the existing quickstart flow in [README.md](../README.md), keeping:

- converted MLX weights, for example `/tmp/irodori-quickstart/irodori-v3.npz`
- model config JSON, for example `/tmp/irodori-quickstart/v3-model-config.json`

Also choose a short local reference WAV for speaker conditioning. You can use any 16-bit PCM WAV that is safe to keep on your machine; for example:

```bash
mkdir -p /tmp/irodori-parity
cp /path/to/your/reference.wav /tmp/irodori-parity/v3-reference.wav
```

Do not commit this local reference audio. The scenario's default path is `tests/fixtures/v3-reference.wav` for deterministic fixture metadata, but this repository does not ship a real speaker sample.

Then run both sides and write one report:

```bash
PYTHONPATH="$(pwd)/external/Irodori-TTS:${PYTHONPATH:-}" \
uv run python scripts/run_upstream_parity.py \
  --scenario v3-reference-predicted \
  --run-upstream \
  --run-mlx \
  --upstream-root external/Irodori-TTS \
  --mlx-weights /tmp/irodori-quickstart/irodori-v3.npz \
  --mlx-model-config-json /tmp/irodori-quickstart/v3-model-config.json \
  --ref-wav /tmp/irodori-parity/v3-reference.wav \
  --codec-runtime-mode mlx \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir parity-runs/v3-real \
  --codec-device cpu \
  --num-steps 8 \
  --seed 20260519 \
  --json
```

The report records the exact upstream `uv run python infer.py ...` command and the exact MLX `scripts/generate_wav.py ...` command. Generated WAVs and caches remain under `parity-runs/` or your chosen local output directory and should not be committed.

## Real VoiceDesign Command

Issue [#118](https://github.com/t0yohei/Irodori-TTS-MLX/issues/118) adds the fixed `voicedesign-contrastive-caption` scenario. It uses a short Japanese text prompt with a deliberately contrasting caption, fixed seed `20260518`, manual `--seconds 2.0`, `--num-steps 12`, and the shared CFG defaults including `cfg_scale_caption: 3.0`.

Use fixture mode for a schema-covered smoke run that does not need checkpoints:

```bash
uv run python scripts/run_upstream_parity.py \
  --fixture \
  --scenario voicedesign-contrastive-caption \
  --output-dir parity-runs/fixture-voicedesign \
  --json
```

The report records the caption, caption tokenizer settings, `cfg_scale_caption`, manual duration mode, and fixture WAV properties for both sides. This is the contract evidence to use on machines without an upstream checkout or converted MLX weights.

The current VoiceDesign fixture baseline is:

- `scenario.name: "voicedesign-contrastive-caption"`
- `duration.expected_mode: "manual"` with `seconds: 2.0`
- `sampling.cfg_scale_caption: 3.0`
- upstream and MLX fixture WAV properties use 24000 Hz mono audio and 2.0 s duration
- comparison: `status: "expected_drift"`

For a real run, VoiceDesign uses the same harness but requires caption conditioning and currently uses manual duration:

```bash
PYTHONPATH="$(pwd)/external/Irodori-TTS:${PYTHONPATH:-}" \
uv run python scripts/run_upstream_parity.py \
  --scenario voicedesign-contrastive-caption \
  --run-upstream \
  --run-mlx \
  --upstream-root external/Irodori-TTS \
  --mlx-weights /tmp/irodori-voicedesign/irodori-voicedesign.npz \
  --mlx-model-config-json /tmp/irodori-voicedesign/voicedesign-model-config.json \
  --codec-runtime-mode mlx \
  --codec-path /path/to/dacvae-codec.npz \
  --output-dir parity-runs/voicedesign-real \
  --codec-device cpu \
  --seconds 2.0 \
  --num-steps 12 \
  --seed 20260518 \
  --json
```

If either `external/Irodori-TTS` or the converted MLX artifacts are unavailable, run the same command without those artifacts to produce a partial report. The upstream and MLX sides will be marked `unavailable` with machine-readable reasons such as `missing_upstream_root` or `missing_mlx_weights`, while the scenario metadata remains complete.

## When to Rerun

Rerun the fixture and partial commands when any of these change:

- `scripts/run_upstream_parity.py`
- [upstream_parity_report_schema.json](upstream_parity_report_schema.json)
- scenario defaults for v3 or VoiceDesign
- tokenizer, duration, sampling, CFG, or codec command-line behavior
- generated report interpretation in this document or the v0.2 delivery plan

Rerun the real commands before claiming real audio parity for a release, after
refreshing upstream checkpoints, after changing converted MLX weight layout, or
after changing the codec runtime. If real upstream dependencies are unavailable,
publish the partial report and keep the unavailable reasons in the report body.

## Report Schema

The canonical machine-readable schema is [upstream_parity_report_schema.json](upstream_parity_report_schema.json). It is intentionally permissive about additive fields so #118/#119 can add metrics without breaking existing consumers.

Top-level fields:

- `schema_version`: currently `1`
- `report_status`: `complete` when both sides produced fixture or passed evidence, `partial` when one or both sides were not requested or unavailable, and `failed` when a requested side ran and failed
- `scenario`: prompt, checkpoint family, checkpoint id, reference/caption settings, seed, sampling, duration, tokenizer, and codec settings
- `metadata_axes`: normalized diagnostic axes for tokenizer, duration, sampling, and codec differences
- `upstream`: command, status, availability, command result when run, WAV properties/metrics when present, and compact intermediate metadata when available
- `mlx`: command, status, availability, command result when run, WAV properties/metrics, `generate_wav.py` metadata, and compact intermediate metadata when present
- `comparison`: `expected_drift`, `regression`, or `not_comparable`, plus audio metric deltas and intermediate field comparisons when both sides expose comparable points
- `deferred_scope`: intentionally omitted work for follow-up issues

The first comparison check is conservative. It treats bit-level waveform mismatch as expected drift, flags sample-rate mismatch as a regression, and flags duration deltas larger than 250 ms as a regression.

Audio metrics are diagnostic rather than proof of perceptual equivalence. Reports include normalized peak, RMS, mean absolute amplitude, silence ratio, leading silence, tail RMS/silence, and zero-crossing rate. These are cheap enough for CI fixtures and safe to share, but they cannot prove voice quality or semantic parity.

Intermediate comparisons are intentionally compact. Fixture reports include token/mask counts, duration mode/resolved latent steps, and sampled latent shape/statistics for both sides. Real MLX runs derive duration and sampled-latent shape from `generate_wav.py` metadata; upstream real runs may leave intermediates empty until upstream-side dump points are wired in.
