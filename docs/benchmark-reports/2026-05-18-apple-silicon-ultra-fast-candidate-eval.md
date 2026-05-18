# Apple Silicon ultra-fast candidate evaluation

Issue: [#217](https://github.com/t0yohei/Irodori-TTS-MLX/issues/217)
Parent: [#220](https://github.com/t0yohei/Irodori-TTS-MLX/issues/220)

## Summary

This is the first real Apple Silicon candidate run after adding the
experimental `--preset ultra-fast` switch. It measures the four candidates
identified by the low-step/CFG sweep with real v3 hosted weights and the hosted
MLX DACVAE decode artifact. The intermediate sweep report was removed after
the final candidate evaluation captured the decision.

- Fastest measured request latency: `issue-220-ultra-fast-steps-6-reduced-cfg-1` at 1168.7 ms.
- Implemented `--preset ultra-fast` equivalent: `issue-220-ultra-fast-steps-6-joint-cfg-1` at 1198.2 ms.
- Both 6-step predicted-duration candidates are well below the prior 8-step one-shot v3 no-reference anchor, but human listening found audible non-Japanese/Chinese-like artifacts at the beginning or end across the candidate set. Treat the predicted-duration ultra-fast candidates as rejected for this short prompt until the duration policy is changed.
- The remaining latency floor is still DACVAE decode/materialization: measured audio write is only 1-2 ms median across these runs.
- A manual-duration follow-up with `--seconds 2.5` removed the Chinese-like artifact in listening. The implemented `6/joint/1` shape measured 768.3 ms median `total_to_decode`, and the `8/reduced/1` control measured 848.0 ms. Human listening preferred `8/reduced/1 --seconds 2.5` when accounting for audio quality.
- A later focused CFG comparison at 8 steps and `--seconds 2.5` measured `8/reduced/CFG1/0/0` at 894.6 ms median `total_to_decode` and `8/independent/default CFG 3/3/5` at 934.4 ms. Because the reduced-CFG win was only about 40 ms on complete-WAV latency, the preset keeps normal CFG defaults and only changes steps plus short-prompt duration policy.
- Baselines: [#64 v3 one-shot](2026-05-14-apple-silicon-num-steps-v3-text.md) and the persistent mlx-decode baseline summarized in [the runtime cleanup comparison](2026-05-18-apple-silicon-persistent-batch-runtime-cleanup-comparison.md).

## Environment

- machine: Apple Silicon arm64
- benchmark Python: Python 3.11.15
- text: `今日はいい天気ですね。`
- checkpoint family: v3
- duration mode: predicted duration (`--seconds` omitted)
- RF-DiT weights repo: `t0yohei/Irodori-TTS-MLX-500M-v3`
- RF-DiT weights revision: `078ffb11ffad92e6dde237a6abef730f4341b359`
- DACVAE codec artifact repo: `t0yohei/Irodori-TTS-MLX-DACVAE-Codec`
- DACVAE codec artifact revision: `bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78`
- codec runtime mode: `mlx-decode`
- request shape: 1 warmup request + 4 measured requests per candidate
- seed start: `20260512`

## Ranked Candidates

| Rank | Case | Steps | CFG mode | CFG text | Request wall | sample_rf | DACVAE decode | Decode model | Audio write | Output duration | Quality proxy |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | issue-220-ultra-fast-steps-6-reduced-cfg-1 | 6 | reduced | 1 | 1168.7 ms | 351.5 ms | 779.2 ms | 709.2 ms | 1.8 ms | 4.34 s | experimental-fastest |
| 2 | issue-220-ultra-fast-steps-6-joint-cfg-1 | 6 | joint | 1 | 1198.2 ms | 361.8 ms | 803.3 ms | 736.8 ms | 1.4 ms | 4.48 s | experimental-fastest |
| 3 | issue-220-ultra-fast-steps-8-reduced-cfg-1 | 8 | reduced | 1 | 1287.4 ms | 463.0 ms | 789.6 ms | 714.6 ms | 1.3 ms | 4.36 s | plausibly-usable |
| 4 | issue-220-ultra-fast-steps-8-independent-cfg-2 | 8 | independent | 2 | 1310.4 ms | 469.3 ms | 798.2 ms | 723.9 ms | 1.9 ms | 4.44 s | plausibly-usable |

## Readout

`6/reduced/1` is the latency winner, but it is not the implemented preset.
The current `ultra-fast` mapping uses `6/joint/1`, which is only about 30 ms
slower in this run while keeping the originally selected joint-guidance shape.
Human listening rejected the predicted-duration candidate set because each
candidate had a strange Chinese-like artifact at the beginning or end. That
means the measured latency win is real, but the predicted-duration short-prompt
audio is not acceptable evidence for promoting `ultra-fast`.

The 8-step candidates are slower by about 90-140 ms versus `6/joint/1`.
They remain useful as safer comparison samples for listening review, especially
if the 6-step outputs sound unstable.

Measured output duration stayed close to the same short-prompt band for all
predicted-duration candidates (4.34-4.48 s median). For the text `今日はいい天気ですね。`,
that is likely too long and matches the existing short-prompt warning that
over-allocated v3 predicted duration can produce tail repetition or artifacts.

## Manual Duration Follow-up

After the listening failure, two candidates were rerun with `--seconds 2.5`
instead of predicted duration:

| Case | Steps | CFG mode | CFG text | Request wall | sample_rf | DACVAE decode | Decode model | Audio write | Output duration | Status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| issue-220-ultra-fast-steps-6-joint-cfg-1-seconds-2p5 | 6 | joint | 1 | 768.3 ms | 257.3 ms | 499.6 ms | 437.0 ms | 1.4 ms | 2.50 s | artifact fixed; lower quality |
| issue-220-ultra-fast-steps-8-reduced-cfg-1-seconds-2p5 | 8 | reduced | 1 | 848.0 ms | 331.5 ms | 521.3 ms | 431.1 ms | 3.5 ms | 2.50 s | preferred listening candidate |

These numbers show that explicit short duration can bring complete-WAV latency
below one second, but it changes the UX contract: `ultra-fast` would need either
a short-prompt duration cap/scale or documentation telling users to combine it
with a manual duration for very short prompts. Listening feedback currently
favors the 8-step control over the lower-latency 6-step joint-CFG candidate
because the 8-step output has the better quality/speed tradeoff.

## CFG Defaults Follow-up

A focused follow-up reran the same short prompt with 8 steps, `--seconds 2.5`,
and identical seeds to isolate the cost of changing CFG away from the normal
defaults:

| Case | Steps | CFG mode | CFG text/caption/speaker | Request wall | sample_rf | DACVAE decode | Output duration | Status |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| issue-220-cfg-compare-8-reduced-cfg1-seconds-2p5 | 8 | reduced | 1/0/0 | 894.6 ms | 340.3 ms | 557.8 ms | 2.50 s | lower latency |
| issue-220-cfg-compare-8-independent-default-cfg-seconds-2p5 | 8 | independent | 3/3/5 | 934.4 ms | 385.4 ms | 547.9 ms | 2.50 s | preferred contract |

The reduced-CFG setting saves about 40 ms on complete-WAV latency in this run.
That is not enough to justify changing the preset's conditioning semantics, so
`ultra-fast` should keep the normal CFG defaults and rely on `--num-steps 8`
plus the short-prompt duration cap for the product-facing speedup.

## Listening Artifacts

Generated WAV files are intentionally not committed to the repository. They are
available in the local benchmark output tree from this run:

    benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-6-joint-cfg-1/
    benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-6-reduced-cfg-1/
    benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-8-reduced-cfg-1/
    benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-8-independent-cfg-2/

Use the request 02-05 WAVs in each directory for measured samples; request 01 is
the warmup sample. The predicted-duration candidate files are retained as
negative evidence because they exposed the Chinese-like start/end artifact. The
manual-duration `seconds-2p5` files should be the next listening target.

## Recommendation

Keep `--preset ultra-fast` experimental and do not promote the predicted-duration
short-prompt path. The best current product direction is 8 steps with normal CFG
defaults plus a short-prompt duration policy: it stays under one second in the
focused follow-up, avoids the Chinese-like artifacts, preserves the default CFG
contract, and avoids simply choosing the lowest-latency 6-step setting.

## Evidence Fields

Each source summary must include invocation.num_steps, invocation.cfg_guidance_mode, invocation.cfg_scale_text, process.wall_seconds, aggregates.measured_total_to_decode_ms, aggregates.measured_sample_rf_ms, aggregates.measured_decode_dacvae_ms, and, for current reports, aggregates.measured_audio_write_ms plus aggregates.measured_output_duration_seconds.

## Source Summaries

- issue-220-ultra-fast-steps-6-reduced-cfg-1: benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-6-reduced-cfg-1/persistent-batch-summary.json
- issue-220-ultra-fast-steps-6-joint-cfg-1: benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-6-joint-cfg-1/persistent-batch-summary.json
- issue-220-ultra-fast-steps-6-joint-cfg-1-seconds-2p5: benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-6-joint-cfg-1-seconds-2p5/persistent-batch-summary.json
- issue-220-ultra-fast-steps-8-reduced-cfg-1: benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-8-reduced-cfg-1/persistent-batch-summary.json
- issue-220-ultra-fast-steps-8-reduced-cfg-1-seconds-2p5: benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-8-reduced-cfg-1-seconds-2p5/persistent-batch-summary.json
- issue-220-ultra-fast-steps-8-independent-cfg-2: benchmark-runs/issue-220-ultra-fast-candidate-eval/issue-220-ultra-fast-steps-8-independent-cfg-2/persistent-batch-summary.json
