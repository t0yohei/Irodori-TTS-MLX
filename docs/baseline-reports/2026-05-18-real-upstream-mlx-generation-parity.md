# Real Upstream-vs-MLX Generation Parity Report - 2026-05-18

Issue: [#189](https://github.com/t0yohei/Irodori-TTS-MLX/issues/189)

This report summarizes local real-checkpoint runs from
`scripts/run_upstream_parity.py`. The generated WAV files and machine-readable
JSON reports were written under `/tmp/irodori-issue-189-parity` and are not
committed because this repository must not redistribute generated audio,
upstream checkpoints, converted weights, codec artifacts, or cache snapshots.

## Environment

- repository commit under test: `2dfaa07`
- upstream checkout: `Aratako/Irodori-TTS` commit `07dfa74d19e961faa499d8d365f36914fd85a97e`
- Python: `3.11.15`
- platform: Apple Silicon macOS, MLX runtime with model device `mps` and codec device `cpu`
- upstream codec package: `dacvae` revision `414c20785fc3a28373073ea8ef7a1316eeeaca6e`
- codec weights: `Aratako/Semantic-DACVAE-Japanese-32dim` revision `47376ee24834d7a05a48ebabfe3cde29b3c5e214`

## Artifact revisions

- v3 RF-DiT MLX artifact: `t0yohei/Irodori-TTS-MLX-500M-v3` revision
  `078ffb11ffad92e6dde237a6abef730f4341b359`
- VoiceDesign RF-DiT MLX artifact:
  `t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign` revision
  `bf877a3beb7d921dc6bfb2b6812d02be07f39f2a`
- DACVAE MLX codec artifact: `t0yohei/Irodori-TTS-MLX-DACVAE-Codec` PR snapshot
  `16d64e0978afe79c46b971405bba4f464cc743f8`

The DACVAE codec artifact was still a Hugging Face PR snapshot at the time of
this run. It is acceptable local validation evidence, but it should not be
documented as an approved public `--codec-artifact-repo` dependency until the
artifact PR is merged and pinned.

## Scenario results

### v3 no-reference, MLX decode

Command summary:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
uv run --python 3.11 python scripts/run_upstream_parity.py \
  --scenario v3-no-reference \
  --scenario-name v3-no-reference-mlx-decode \
  --run-upstream \
  --run-mlx \
  --upstream-root /path/to/Irodori-TTS \
  --mlx-weights /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-500M-v3/078ffb11ffad92e6dde237a6abef730f4341b359/weights.npz \
  --mlx-model-config-json /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-500M-v3/078ffb11ffad92e6dde237a6abef730f4341b359/model_config.json \
  --codec-runtime-mode mlx-decode \
  --codec-path /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-DACVAE-Codec/16d64e0978afe79c46b971405bba4f464cc743f8/dacvae-codec.npz \
  --output-dir /tmp/irodori-issue-189-parity/v3-no-reference-mlx-decode \
  --codec-device cpu \
  --num-steps 8 \
  --seed 20260516
```

Result:

- report: `/tmp/irodori-issue-189-parity/v3-no-reference-mlx-decode/v3-no-reference-mlx-decode.parity.json`
- `report_status: complete`
- `comparison.status: expected_drift`
- upstream audio: 48,000 Hz, 215,040 samples, 4.48 s
- MLX audio: 48,000 Hz, 207,360 samples, 4.32 s
- duration delta: -0.16 s, within the harness regression threshold
- MLX metadata reported PyTorch reference encode not required and MLX decode

### VoiceDesign caption-conditioned, MLX decode

Command summary:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
uv run --python 3.11 python scripts/run_upstream_parity.py \
  --scenario voicedesign-contrastive-caption \
  --scenario-name voicedesign-contrastive-caption-mlx-decode \
  --run-upstream \
  --run-mlx \
  --upstream-root /path/to/Irodori-TTS \
  --mlx-weights /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign/bf877a3beb7d921dc6bfb2b6812d02be07f39f2a/weights.npz \
  --mlx-model-config-json /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign/bf877a3beb7d921dc6bfb2b6812d02be07f39f2a/model_config.json \
  --codec-runtime-mode mlx-decode \
  --codec-path /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-DACVAE-Codec/16d64e0978afe79c46b971405bba4f464cc743f8/dacvae-codec.npz \
  --output-dir /tmp/irodori-issue-189-parity/voicedesign-contrastive-caption-mlx-decode \
  --codec-device cpu \
  --seconds 2.0 \
  --num-steps 12 \
  --seed 20260518
```

Result:

- report: `/tmp/irodori-issue-189-parity/voicedesign-contrastive-caption-mlx-decode/voicedesign-contrastive-caption-mlx-decode.parity.json`
- `report_status: complete`
- `comparison.status: regression`
- upstream audio: 48,000 Hz, 368,640 samples, 7.68 s
- MLX audio: 48,000 Hz, 96,000 samples, 2.00 s
- duration delta: -5.68 s
- reason: the upstream VoiceDesign CLI path did not constrain output length to
  the requested `--seconds 2.0`, while the MLX path generated exactly 2.00 s

This is a real mismatch in current command semantics, not bitwise acoustic
drift. The report is useful evidence because both sides ran, but this scenario
should not be cited as passing parity until the upstream command semantics or
comparison expectation are narrowed.

### v3 reference-audio, full MLX codec

Command summary:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
uv run --python 3.11 python scripts/run_upstream_parity.py \
  --scenario v3-reference-predicted \
  --scenario-name v3-reference-predicted-mlx-codec \
  --run-upstream \
  --run-mlx \
  --upstream-root /path/to/Irodori-TTS \
  --mlx-weights /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-500M-v3/078ffb11ffad92e6dde237a6abef730f4341b359/weights.npz \
  --mlx-model-config-json /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-500M-v3/078ffb11ffad92e6dde237a6abef730f4341b359/model_config.json \
  --reference-wav /path/to/local/reference.wav \
  --codec-runtime-mode mlx \
  --codec-path /path/to/hf-cache/t0yohei/Irodori-TTS-MLX-DACVAE-Codec/16d64e0978afe79c46b971405bba4f464cc743f8/dacvae-codec.npz \
  --output-dir /tmp/irodori-issue-189-parity/v3-reference-predicted-mlx-codec \
  --codec-device cpu \
  --num-steps 8 \
  --seed 20260519
```

Result:

- report: `/tmp/irodori-issue-189-parity/v3-reference-predicted-mlx-codec/v3-reference-predicted-mlx-codec.parity.json`
- `report_status: complete`
- `comparison.status: expected_drift`
- upstream audio: 48,000 Hz, 359,040 samples, 7.48 s
- MLX audio: 48,000 Hz, 360,960 samples, 7.52 s
- duration delta: +0.04 s, within the harness regression threshold
- MLX metadata reported MLX encode and MLX decode

The local reference WAV was a generated local fixture and is not committed. Use
only reference audio that is safe for local validation and never commit it.

## Interpretation

These reports prove that representative real upstream and MLX commands can run
to completion with pinned artifacts and that the harness captures sample rate,
duration, audio length, audio metrics, command inputs, and comparison status.

They do not prove perceptual equivalence, speaker identity equivalence, or
bitwise waveform parity. The expected passing state for real generation remains
`expected_drift`, because PyTorch and MLX sampling/codec paths are not expected
to produce identical waveforms. The VoiceDesign result is explicitly a duration
semantics regression under the current command pair and needs a narrower command
contract or runtime fix before it can be used as passing parity evidence.
