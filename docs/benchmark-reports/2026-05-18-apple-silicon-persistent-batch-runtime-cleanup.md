# Persistent Batch Benchmark Report

## Summary

- Case label: issue-213-runtime-cleanup
- Requests: 5 (1 warmup, 4 measured)
- Prompt text: 今日はいい天気ですね。
- Seed start: 20260512
- Num steps: 12
- Seconds: predicted duration (--seconds omitted)
- Codec runtime mode: mlx-decode
- Cleanup between requests: False

## Process results

| Status | Wall | Setup/load overhead | Max RSS | Process throughput | Measured generation throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| passed | 13.78 s | 5406.1 ms | 3134816256 bytes (2.92 GiB) | 0.363 req/s | 0.596 req/s |

## Request timing aggregates

| Scope | first request | measured median | measured min/max |
| --- | ---: | ---: | --- |
| total_to_decode | 1664.8 ms | 1670.2 ms | 1644.9 ms / 1723.9 ms |
| sample_rf | 835.9 ms | 834.1 ms | 829.4 ms / 837.2 ms |
| encode_dacvae |  |  |  |
| decode_dacvae | 793.9 ms | 801.9 ms | 782.8 ms / 854.0 ms |

## Raw requests

| # | Phase | Seed | total_to_decode | sample_rf | encode_dacvae | decode_dacvae | Output |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | warmup | 20260512 | 1664.8 ms | 835.9 ms |  | 793.9 ms | /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/issue-213-runtime-cleanup.request-01.wav |
| 2 | measured | 20260513 | 1723.9 ms | 837.2 ms |  | 854.0 ms | /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/issue-213-runtime-cleanup.request-02.wav |
| 3 | measured | 20260514 | 1673.9 ms | 832.9 ms |  | 803.1 ms | /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/issue-213-runtime-cleanup.request-03.wav |
| 4 | measured | 20260515 | 1666.5 ms | 835.4 ms |  | 800.8 ms | /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/issue-213-runtime-cleanup.request-04.wav |
| 5 | measured | 20260516 | 1644.9 ms | 829.4 ms |  | 782.8 ms | /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/issue-213-runtime-cleanup.request-05.wav |

Command:

    /usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --requests-json /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/requests.json --metadata-json /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/metadata.json --json --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode mlx-decode --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78

- Requests JSON: /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/requests.json
- Metadata JSON: /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/metadata.json
- stdout log: /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/persistent-batch.stdout.log
- stderr log: /path/to/Irodori-TTS-MLX/benchmark-runs/issue-213-runtime-cleanup/persistent-batch.stderr.log
