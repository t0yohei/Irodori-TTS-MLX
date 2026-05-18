# Persistent Batch Benchmark Report

## Summary

- Case label: issue-209-cleanup-off
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
| passed | 19.72 s | 10192.0 ms | 914898944 bytes (0.85 GiB) | 0.254 req/s | 0.522 req/s |

## Request timing aggregates

| Scope | first request | measured median | measured min/max |
| --- | ---: | ---: | --- |
| total_to_decode | 1861.0 ms | 1918.2 ms | 1604.4 ms / 2226.2 ms |
| sample_rf | 951.0 ms | 873.8 ms | 864.7 ms / 898.3 ms |
| encode_dacvae |  |  |  |
| decode_dacvae | 827.9 ms | 1005.2 ms | 717.4 ms / 1231.2 ms |

## Raw requests

| # | Phase | Seed | total_to_decode | sample_rf | encode_dacvae | decode_dacvae | Output |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | warmup | 20260512 | 1861.0 ms | 951.0 ms |  | 827.9 ms | /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/issue-209-cleanup-off.request-01.wav |
| 2 | measured | 20260513 | 1684.8 ms | 874.7 ms |  | 786.3 ms | /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/issue-209-cleanup-off.request-02.wav |
| 3 | measured | 20260514 | 1604.4 ms | 864.7 ms |  | 717.4 ms | /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/issue-209-cleanup-off.request-03.wav |
| 4 | measured | 20260515 | 2151.6 ms | 898.3 ms |  | 1231.2 ms | /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/issue-209-cleanup-off.request-04.wav |
| 5 | measured | 20260516 | 2226.2 ms | 872.8 ms |  | 1224.2 ms | /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/issue-209-cleanup-off.request-05.wav |

Command:

    /usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 --requests-json /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/requests.json --metadata-json /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/metadata.json --json --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode mlx-decode --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78

- Requests JSON: /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/requests.json
- Metadata JSON: /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/metadata.json
- stdout log: /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/persistent-batch.stdout.log
- stderr log: /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-209-persistent-decode-slowdown/benchmark-runs/issue-209-cleanup-off/persistent-batch.stderr.log
