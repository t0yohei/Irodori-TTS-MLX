# Apple Silicon Benchmark Report

## Summary

- Prompt text: `今日はいい天気ですね。`
- Seed: `20260512`
- Measured repeats per case: `1`
- Warmup runs per case: `1`
- Cache-state labeling mode: `auto`

## Aggregate results

| Case | Phase | Cache | Runs | sample_rf median | sample_rf min/max | decode median | total median | wall median | max RSS median |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| mlx-bridge-v3-reference-predicted-steps-12 | measured | warm | 1 | 881.5 ms | 881.5 ms / 881.5 ms | 835.8 ms | 2436.1 ms | 7.01 s | 4676501504 bytes (4.36 GiB) |
| mlx-bridge-v3-reference-predicted-steps-12 | warmup | cold | 1 | 894.4 ms | 894.4 ms / 894.4 ms | 862.4 ms | 2470.7 ms | 6.99 s | 4718968832 bytes (4.39 GiB) |
| mlx-bridge-v3-reference-predicted-steps-16 | measured | warm | 1 | 1096.3 ms | 1096.3 ms / 1096.3 ms | 801.9 ms | 2572.2 ms | 7.15 s | 4656873472 bytes (4.34 GiB) |
| mlx-bridge-v3-reference-predicted-steps-16 | warmup | cold | 1 | 1112.8 ms | 1112.8 ms / 1112.8 ms | 861.0 ms | 2675.2 ms | 7.24 s | 4728487936 bytes (4.40 GiB) |
| mlx-bridge-v3-reference-predicted-steps-24 | measured | warm | 1 | 1667.8 ms | 1667.8 ms / 1667.8 ms | 841.1 ms | 3194.4 ms | 7.65 s | 4732502016 bytes (4.41 GiB) |
| mlx-bridge-v3-reference-predicted-steps-24 | warmup | cold | 1 | 1583.5 ms | 1583.5 ms / 1583.5 ms | 897.7 ms | 3190.5 ms | 7.62 s | 4777738240 bytes (4.45 GiB) |
| mlx-bridge-v3-reference-predicted-steps-40 | measured | warm | 1 | 2717.1 ms | 2717.1 ms / 2717.1 ms | 838.4 ms | 4241.1 ms | 8.70 s | 4765777920 bytes (4.44 GiB) |
| mlx-bridge-v3-reference-predicted-steps-40 | warmup | cold | 1 | 2814.5 ms | 2814.5 ms / 2814.5 ms | 881.5 ms | 4406.1 ms | 8.84 s | 4762910720 bytes (4.44 GiB) |
| mlx-bridge-v3-reference-predicted-steps-8 | measured | warm | 1 | 616.1 ms | 616.1 ms / 616.1 ms | 880.6 ms | 2223.4 ms | 6.92 s | 5394972672 bytes (5.02 GiB) |
| mlx-bridge-v3-reference-predicted-steps-8 | warmup | cold | 1 | 595.6 ms | 595.6 ms / 595.6 ms | 905.9 ms | 2265.5 ms | 6.79 s | 5340037120 bytes (4.97 GiB) |

## mlx-bridge-v3-reference-predicted-steps-12 · measured · warm

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `12`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 835.8 ms | 835.8 ms | 835.8 ms |
| `predict_duration` | 50.9 ms | 50.9 ms | 50.9 ms |
| `prepare_reference_condition` | 667.5 ms | 667.5 ms | 667.5 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 881.5 ms | 881.5 ms | 881.5 ms |
| `total_to_decode` | 2436.1 ms | 2436.1 ms | 2436.1 ms |
| `wall_seconds` | 7.01 s | 7.01 s | 7.01 s |
| `max_rss_bytes` | 4676501504 bytes (4.36 GiB) | 4676501504 bytes (4.36 GiB) | 4676501504 bytes (4.36 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-12-measured-run-01 | passed | 881.5 ms | 835.8 ms | 2436.1 ms | 7.01 s | 4676501504 bytes (4.36 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.measured.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-12-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 12 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-12 · warmup · cold

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `12`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 862.4 ms | 862.4 ms | 862.4 ms |
| `predict_duration` | 48.1 ms | 48.1 ms | 48.1 ms |
| `prepare_reference_condition` | 665.4 ms | 665.4 ms | 665.4 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 894.4 ms | 894.4 ms | 894.4 ms |
| `total_to_decode` | 2470.7 ms | 2470.7 ms | 2470.7 ms |
| `wall_seconds` | 6.99 s | 6.99 s | 6.99 s |
| `max_rss_bytes` | 4718968832 bytes (4.39 GiB) | 4718968832 bytes (4.39 GiB) | 4718968832 bytes (4.39 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-12-warmup-run-01 | passed | 894.4 ms | 862.4 ms | 2470.7 ms | 6.99 s | 4718968832 bytes (4.39 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.warmup.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-12-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-12.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 12 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-16 · measured · warm

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `16`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 801.9 ms | 801.9 ms | 801.9 ms |
| `predict_duration` | 45.8 ms | 45.8 ms | 45.8 ms |
| `prepare_reference_condition` | 627.9 ms | 627.9 ms | 627.9 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1096.3 ms | 1096.3 ms | 1096.3 ms |
| `total_to_decode` | 2572.2 ms | 2572.2 ms | 2572.2 ms |
| `wall_seconds` | 7.15 s | 7.15 s | 7.15 s |
| `max_rss_bytes` | 4656873472 bytes (4.34 GiB) | 4656873472 bytes (4.34 GiB) | 4656873472 bytes (4.34 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-16-measured-run-01 | passed | 1096.3 ms | 801.9 ms | 2572.2 ms | 7.15 s | 4656873472 bytes (4.34 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.measured.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-16-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 16 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-16 · warmup · cold

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `16`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 861.0 ms | 861.0 ms | 861.0 ms |
| `predict_duration` | 47.1 ms | 47.1 ms | 47.1 ms |
| `prepare_reference_condition` | 654.1 ms | 654.1 ms | 654.1 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1112.8 ms | 1112.8 ms | 1112.8 ms |
| `total_to_decode` | 2675.2 ms | 2675.2 ms | 2675.2 ms |
| `wall_seconds` | 7.24 s | 7.24 s | 7.24 s |
| `max_rss_bytes` | 4728487936 bytes (4.40 GiB) | 4728487936 bytes (4.40 GiB) | 4728487936 bytes (4.40 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-16-warmup-run-01 | passed | 1112.8 ms | 861.0 ms | 2675.2 ms | 7.24 s | 4728487936 bytes (4.40 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.warmup.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-16-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-16.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 16 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-24 · measured · warm

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `24`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 841.1 ms | 841.1 ms | 841.1 ms |
| `predict_duration` | 49.4 ms | 49.4 ms | 49.4 ms |
| `prepare_reference_condition` | 635.8 ms | 635.8 ms | 635.8 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1667.8 ms | 1667.8 ms | 1667.8 ms |
| `total_to_decode` | 3194.4 ms | 3194.4 ms | 3194.4 ms |
| `wall_seconds` | 7.65 s | 7.65 s | 7.65 s |
| `max_rss_bytes` | 4732502016 bytes (4.41 GiB) | 4732502016 bytes (4.41 GiB) | 4732502016 bytes (4.41 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-24-measured-run-01 | passed | 1667.8 ms | 841.1 ms | 3194.4 ms | 7.65 s | 4732502016 bytes (4.41 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.measured.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-24-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 24 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-24 · warmup · cold

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `24`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 897.7 ms | 897.7 ms | 897.7 ms |
| `predict_duration` | 43.7 ms | 43.7 ms | 43.7 ms |
| `prepare_reference_condition` | 665.2 ms | 665.2 ms | 665.2 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1583.5 ms | 1583.5 ms | 1583.5 ms |
| `total_to_decode` | 3190.5 ms | 3190.5 ms | 3190.5 ms |
| `wall_seconds` | 7.62 s | 7.62 s | 7.62 s |
| `max_rss_bytes` | 4777738240 bytes (4.45 GiB) | 4777738240 bytes (4.45 GiB) | 4777738240 bytes (4.45 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-24-warmup-run-01 | passed | 1583.5 ms | 897.7 ms | 3190.5 ms | 7.62 s | 4777738240 bytes (4.45 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.warmup.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-24-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-24.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 24 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-40 · measured · warm

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `40`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 838.4 ms | 838.4 ms | 838.4 ms |
| `predict_duration` | 46.1 ms | 46.1 ms | 46.1 ms |
| `prepare_reference_condition` | 639.1 ms | 639.1 ms | 639.1 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 2717.1 ms | 2717.1 ms | 2717.1 ms |
| `total_to_decode` | 4241.1 ms | 4241.1 ms | 4241.1 ms |
| `wall_seconds` | 8.70 s | 8.70 s | 8.70 s |
| `max_rss_bytes` | 4765777920 bytes (4.44 GiB) | 4765777920 bytes (4.44 GiB) | 4765777920 bytes (4.44 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-40-measured-run-01 | passed | 2717.1 ms | 838.4 ms | 4241.1 ms | 8.70 s | 4765777920 bytes (4.44 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.measured.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-40-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 40 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-40 · warmup · cold

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `40`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 881.5 ms | 881.5 ms | 881.5 ms |
| `predict_duration` | 50.1 ms | 50.1 ms | 50.1 ms |
| `prepare_reference_condition` | 659.8 ms | 659.8 ms | 659.8 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 2814.5 ms | 2814.5 ms | 2814.5 ms |
| `total_to_decode` | 4406.1 ms | 4406.1 ms | 4406.1 ms |
| `wall_seconds` | 8.84 s | 8.84 s | 8.84 s |
| `max_rss_bytes` | 4762910720 bytes (4.44 GiB) | 4762910720 bytes (4.44 GiB) | 4762910720 bytes (4.44 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-40-warmup-run-01 | passed | 2814.5 ms | 881.5 ms | 4406.1 ms | 8.84 s | 4762910720 bytes (4.44 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.warmup.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-40-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-40.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 40 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-8 · measured · warm

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `8`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 880.6 ms | 880.6 ms | 880.6 ms |
| `predict_duration` | 52.7 ms | 52.7 ms | 52.7 ms |
| `prepare_reference_condition` | 673.6 ms | 673.6 ms | 673.6 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 616.1 ms | 616.1 ms | 616.1 ms |
| `total_to_decode` | 2223.4 ms | 2223.4 ms | 2223.4 ms |
| `wall_seconds` | 6.92 s | 6.92 s | 6.92 s |
| `max_rss_bytes` | 5394972672 bytes (5.02 GiB) | 5394972672 bytes (5.02 GiB) | 5394972672 bytes (5.02 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-8-measured-run-01 | passed | 616.1 ms | 880.6 ms | 2223.4 ms | 6.92 s | 5394972672 bytes (5.02 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.measured.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-8-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 8 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-reference-predicted-steps-8 · warmup · cold

- Kind: `mlx`
- Reference mode: `reference`
- Num steps: `8`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 905.9 ms | 905.9 ms | 905.9 ms |
| `predict_duration` | 44.4 ms | 44.4 ms | 44.4 ms |
| `prepare_reference_condition` | 719.2 ms | 719.2 ms | 719.2 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 595.6 ms | 595.6 ms | 595.6 ms |
| `total_to_decode` | 2265.5 ms | 2265.5 ms | 2265.5 ms |
| `wall_seconds` | 6.79 s | 6.79 s | 6.79 s |
| `max_rss_bytes` | 5340037120 bytes (4.97 GiB) | 5340037120 bytes (4.97 GiB) | 5340037120 bytes (4.97 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-reference-predicted-steps-8-warmup-run-01 | passed | 595.6 ms | 905.9 ms | 2265.5 ms | 6.79 s | 5340037120 bytes (4.97 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.warmup.stderr.log` |

### mlx-bridge-v3-reference-predicted-steps-8-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-reference/mlx-bridge-v3-reference-predicted-steps-8.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 8 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --reference-wav /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-hosted.wav --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```
