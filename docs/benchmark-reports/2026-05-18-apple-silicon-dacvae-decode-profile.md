# Persistent Batch Benchmark Report

Issue: [#221 Profile and reduce MLX DACVAE decode latency](https://github.com/t0yohei/Irodori-TTS-MLX/issues/221)

## Summary

- Case label: issue-221-dacvae-profile
- Requests: 4 (1 warmup, 3 measured)
- Prompt text: 今日はいい天気ですね。
- Seed start: 20260518
- Num steps: 12
- Seconds: predicted duration (--seconds omitted)
- Codec runtime mode: mlx-decode
- Cleanup between requests: False

## Process results

| Status | Wall | Setup/load overhead | Max RSS | Process throughput | Measured generation throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| passed | 16.05 s | 6951.6 ms | 2934865920 bytes (2.73 GiB) | 0.249 req/s | 0.543 req/s |

## Request timing aggregates

| Scope | first request | measured median | measured min/max |
| --- | ---: | ---: | --- |
| total_to_decode | 3572.5 ms | 1808.5 ms | 1763.0 ms / 1954.4 ms |
| sample_rf | 1048.5 ms | 881.7 ms | 875.2 ms / 940.1 ms |
| encode_dacvae |  |  |  |
| decode_dacvae | 2437.6 ms | 871.3 ms | 850.9 ms / 918.0 ms |

## MLX decode subphase aggregates

These rows are emitted by the MLX codec bridge when codec runtime mode uses MLX decode.

| Scope | first request | measured median | measured min/max |
| --- | ---: | ---: | --- |
| model compute/schedule | 0.4 ms | 0.5 ms | 0.4 ms / 0.8 ms |
| materialization/sync | 2279.9 ms | 804.3 ms | 781.9 ms / 833.8 ms |
| host transfer | 0.4 ms | 0.2 ms | 0.2 ms / 0.3 ms |
| postprocess | 0.0 ms | 0.0 ms | 0.0 ms / 0.0 ms |
| WAV serialization | 16.2 ms | 1.7 ms | 1.2 ms / 2.0 ms |
| cleanup | 140.8 ms | 66.3 ms | 65.1 ms / 81.4 ms |

## Interpretation

The measured short no-reference v3 `mlx-decode` path is already dominated by the realized MLX decoder execution at the materialization/synchronization boundary. Median measured decode was 871.3 ms; median materialization/sync alone was 804.3 ms, or about 92% of decode time. The host transfer, postprocess, and WAV serialization stages were each below 2 ms median, so avoiding extra NumPy copies or changing WAV serialization is not a meaningful sub-second lever for this prompt shape.

Cleanup is still visible at 66.3 ms median, but it is the explicit runtime boundary that stabilized repeated decode latency in the #213 follow-up. Removing or deferring it would trade about 65-80 ms of request time for a known repeated-request drift risk, while leaving the dominant 780-835 ms decoder materialization cost untouched. That is not a low-risk reduction for the persistent worker path.

The practical floor for this implementation is therefore close to the current 0.85-0.92 second decode band unless the Semantic-DACVAE decoder graph itself is changed, compiled differently by MLX, or moved off the complete-WAV critical path with first-audio/streaming behavior. For the #220 sub-second complete-WAV target, the current budget still leaves too little room for RF sampling; first-audio latency remains the more realistic track.

## Raw requests

| # | Phase | Seed | total_to_decode | sample_rf | encode_dacvae | decode_dacvae | Output |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | warmup | 20260518 | 3572.5 ms | 1048.5 ms |  | 2437.6 ms | <worktree>/benchmark-runs/issue-221-dacvae-profile/issue-221-dacvae-profile.request-01.wav |
| 2 | measured | 20260519 | 1954.4 ms | 940.1 ms |  | 918.0 ms | <worktree>/benchmark-runs/issue-221-dacvae-profile/issue-221-dacvae-profile.request-02.wav |
| 3 | measured | 20260520 | 1808.5 ms | 881.7 ms |  | 871.3 ms | <worktree>/benchmark-runs/issue-221-dacvae-profile/issue-221-dacvae-profile.request-03.wav |
| 4 | measured | 20260521 | 1763.0 ms | 875.2 ms |  | 850.9 ms | <worktree>/benchmark-runs/issue-221-dacvae-profile/issue-221-dacvae-profile.request-04.wav |

Command:

    /usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --requests-json <worktree>/benchmark-runs/issue-221-dacvae-profile/requests.json --metadata-json <worktree>/benchmark-runs/issue-221-dacvae-profile/metadata.json --json --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode mlx-decode --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78

- Requests JSON: <worktree>/benchmark-runs/issue-221-dacvae-profile/requests.json
- Metadata JSON: <worktree>/benchmark-runs/issue-221-dacvae-profile/metadata.json
- stdout log: <worktree>/benchmark-runs/issue-221-dacvae-profile/persistent-batch.stdout.log
- stderr log: <worktree>/benchmark-runs/issue-221-dacvae-profile/persistent-batch.stderr.log
