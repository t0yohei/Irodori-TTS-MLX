# Apple Silicon ultra-fast low-step/CFG sweep plan

Issue: [#217](https://github.com/t0yohei/Irodori-TTS-MLX/issues/217)  
Parent: [#220](https://github.com/t0yohei/Irodori-TTS-MLX/issues/220)

## Status

This report adds the reproducible sweep harness and reporting contract for the sub-second latency track. I could not commit private Apple Silicon model-output measurements in this branch because the worker environment does not have the gated v3 model artifacts already materialized in the worktree. The exact commands below are the intended evidence path; the PR body records this as the remaining benchmark execution step.

The committed harness now records the fields the parent epic needs for each persistent request:

- persistent request latency: aggregates.measured_total_to_decode_ms
- RF sampling: aggregates.measured_sample_rf_ms
- DACVAE end-to-end decode/write: aggregates.measured_decode_dacvae_ms
- DACVAE model decode when available: aggregates.measured_decode_dacvae_model_ms
- audio serialization/write: aggregates.measured_audio_write_ms
- one-shot/process wall clock: process.wall_seconds
- output duration: aggregates.measured_output_duration_seconds

## Baselines

Use these reports as anchors when interpreting the new sweep:

- [#64 v3 no-reference one-shot](2026-05-14-apple-silicon-num-steps-v3-text.md): 8 steps measured total_to_decode=1818.5 ms, sample_rf=678.6 ms, decode_dacvae=1099.3 ms.
- [persistent mlx-decode after runtime cleanup](2026-05-18-apple-silicon-persistent-batch-runtime-cleanup.md): 12 steps persistent measured median total_to_decode=1670.2 ms, sample_rf=834.1 ms, decode_dacvae=801.9 ms.

## Sweep Matrix

Run v3 no-reference short Japanese text with predicted duration:

- text: 今日はいい天気ですね。
- seed start: 20260512
- num_steps: 4, 6, 8, 10, 12
- cfg_guidance_mode: independent, joint, reduced
- cfg_scale_text: 0, 1, 2, 3
- codec_runtime_mode: mlx-decode
- request shape: 1 warmup + 4 measured requests per candidate

## Commands

Set these paths for the local Apple Silicon host:

    PYTHON=/path/to/Irodori-TTS-MLX/.venv/bin/python
    WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v3
    WEIGHTS_REVISION=078ffb11ffad92e6dde237a6abef730f4341b359
    CODEC_ARTIFACT_REPO=t0yohei/Irodori-TTS-MLX-DACVAE-Codec
    CODEC_ARTIFACT_REVISION=bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78
    OUT=benchmark-runs/issue-217-ultra-fast-sweep

    for steps in 4 6 8 10 12; do
      for mode in independent joint reduced; do
        for cfg in 0 1 2 3; do
          label="issue-217-steps-$steps-$mode-cfg-$cfg"
          python scripts/benchmark_persistent_batch.py \
            --output-dir "$OUT/$label" \
            --report "docs/benchmark-reports/$label.md" \
            --case-label "$label" \
            --mlx-python "$PYTHON" \
            --weights-repo "$WEIGHTS_REPO" \
            --weights-revision "$WEIGHTS_REVISION" \
            --codec-artifact-repo "$CODEC_ARTIFACT_REPO" \
            --codec-artifact-revision "$CODEC_ARTIFACT_REVISION" \
            --codec-runtime-mode mlx-decode \
            --omit-seconds \
            --num-steps "$steps" \
            --cfg-guidance-mode "$mode" \
            --cfg-scale-text "$cfg" \
            --cfg-scale-speaker 0 \
            --cfg-scale-caption 0 \
            --requests 4 \
            --warmup-requests 1
        done
      done
    done

    python scripts/report_ultra_fast_sweep.py \
      $OUT/*/persistent-batch-summary.json \
      --output docs/benchmark-reports/2026-05-18-apple-silicon-ultra-fast-low-step-cfg-sweep.md

## Interpretation Rules

Latency-only fastest setting is expected to be one of the 4-step candidates, especially with low or zero text CFG. That is not enough for a public preset. The quality proxy labels 4-step and zero-CFG runs as experimental until subjective listening or parity checks confirm intelligibility and speaker/style stability.

The first plausible candidates to review by ear are:

| Priority | Steps | CFG mode | CFG text | Reason |
| ---: | ---: | --- | ---: | --- |
| 1 | 6 | joint | 1 | likely much faster than 8/12 steps while retaining some text guidance |
| 2 | 6 | reduced | 1 | same target latency band with fewer CFG branches |
| 3 | 8 | reduced | 1 | safer quality anchor against the known 8-step baseline |
| 4 | 8 | independent | 2 | conservative low-step comparison against current default guidance |

## Public Preset Recommendation

Do not ship a public ultra-fast preset from this branch. The repo now has the measurement path needed to decide it, but the setting should remain experimental until the full Apple Silicon sweep produces real candidate WAVs and a lightweight listening/parity review marks at least one low-step/CFG combination as intelligible and stable.

