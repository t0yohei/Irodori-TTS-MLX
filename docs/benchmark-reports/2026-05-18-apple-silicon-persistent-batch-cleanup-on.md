# Persistent Batch Benchmark Report

## Summary

- Case label: issue-209-cleanup-on
- Requests: 5 (1 warmup, 4 measured)
- Prompt text: 今日はいい天気ですね。
- Seed start: 20260512
- Num steps: 12
- Seconds: predicted duration (--seconds omitted)
- Codec runtime mode: mlx-decode
- Cleanup between requests: True

## Process results

| Status | Wall | Setup/load overhead | Max RSS | Process throughput | Measured generation throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| passed | 14.35 s | 6289.4 ms | 3136192512 bytes (2.92 GiB) | 0.348 req/s | 0.624 req/s |

## Request timing aggregates

| Scope | first request | measured median | measured min/max |
| --- | ---: | ---: | --- |
| total_to_decode | 1653.9 ms | 1602.0 ms | 1589.7 ms / 1613.0 ms |
| sample_rf | 835.7 ms | 831.1 ms | 827.5 ms / 839.3 ms |
| encode_dacvae |  |  |  |
| decode_dacvae | 778.8 ms | 735.7 ms | 712.8 ms / 745.2 ms |

## Raw requests

| # | Phase | Seed | total_to_decode | sample_rf | encode_dacvae | decode_dacvae | Output |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | warmup | 20260512 | 1653.9 ms | 835.7 ms |  | 778.8 ms | benchmark-runs/issue-209-cleanup-on/issue-209-cleanup-on.request-01.wav |
| 2 | measured | 20260513 | 1589.7 ms | 832.2 ms |  | 712.8 ms | benchmark-runs/issue-209-cleanup-on/issue-209-cleanup-on.request-02.wav |
| 3 | measured | 20260514 | 1605.5 ms | 827.5 ms |  | 745.2 ms | benchmark-runs/issue-209-cleanup-on/issue-209-cleanup-on.request-03.wav |
| 4 | measured | 20260515 | 1598.5 ms | 829.9 ms |  | 734.8 ms | benchmark-runs/issue-209-cleanup-on/issue-209-cleanup-on.request-04.wav |
| 5 | measured | 20260516 | 1613.0 ms | 839.3 ms |  | 736.6 ms | benchmark-runs/issue-209-cleanup-on/issue-209-cleanup-on.request-05.wav |

Command:

    /usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --requests-json benchmark-runs/issue-209-cleanup-on/requests.json --metadata-json benchmark-runs/issue-209-cleanup-on/metadata.json --json --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode mlx-decode --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 --cleanup-between-requests

- Requests JSON: benchmark-runs/issue-209-cleanup-on/requests.json
- Metadata JSON: benchmark-runs/issue-209-cleanup-on/metadata.json
- stdout log: benchmark-runs/issue-209-cleanup-on/persistent-batch.stdout.log
- stderr log: benchmark-runs/issue-209-cleanup-on/persistent-batch.stderr.log
