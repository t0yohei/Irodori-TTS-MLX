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
| mlx-bridge-v3-no-reference-predicted-steps-12 | measured | warm | 1 | 1010.7 ms | 1010.7 ms / 1010.7 ms | 1194.5 ms | 2243.4 ms | 6.93 s | 4934828032 bytes (4.60 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-12 | warmup | cold | 1 | 1001.0 ms | 1001.0 ms / 1001.0 ms | 1070.5 ms | 2111.6 ms | 6.74 s | 4930142208 bytes (4.59 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-16 | measured | warm | 1 | 1316.0 ms | 1316.0 ms / 1316.0 ms | 1140.3 ms | 2491.6 ms | 7.14 s | 4941660160 bytes (4.60 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-16 | warmup | cold | 1 | 1370.0 ms | 1370.0 ms / 1370.0 ms | 1204.3 ms | 2613.0 ms | 7.58 s | 4928847872 bytes (4.59 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-24 | measured | warm | 1 | 1885.3 ms | 1885.3 ms / 1885.3 ms | 971.6 ms | 2901.6 ms | 7.36 s | 4918493184 bytes (4.58 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-24 | warmup | cold | 1 | 1900.5 ms | 1900.5 ms / 1900.5 ms | 1037.8 ms | 2976.5 ms | 7.58 s | 4955848704 bytes (4.62 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-40 | measured | warm | 1 | 3156.2 ms | 3156.2 ms / 3156.2 ms | 1033.7 ms | 4229.6 ms | 8.83 s | 4988059648 bytes (4.65 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-40 | warmup | cold | 1 | 3136.4 ms | 3136.4 ms / 3136.4 ms | 1036.7 ms | 4213.4 ms | 8.73 s | 4951949312 bytes (4.61 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-8 | measured | warm | 1 | 678.6 ms | 678.6 ms / 678.6 ms | 1099.3 ms | 1818.5 ms | 6.37 s | 4926013440 bytes (4.59 GiB) |
| mlx-bridge-v3-no-reference-predicted-steps-8 | warmup | cold | 1 | 685.3 ms | 685.3 ms / 685.3 ms | 1168.0 ms | 1897.1 ms | 7.68 s | 4928667648 bytes (4.59 GiB) |

## mlx-bridge-v3-no-reference-predicted-steps-12 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `12`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1194.5 ms | 1194.5 ms | 1194.5 ms |
| `predict_duration` | 37.8 ms | 37.8 ms | 37.8 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 1010.7 ms | 1010.7 ms | 1010.7 ms |
| `total_to_decode` | 2243.4 ms | 2243.4 ms | 2243.4 ms |
| `wall_seconds` | 6.93 s | 6.93 s | 6.93 s |
| `max_rss_bytes` | 4934828032 bytes (4.60 GiB) | 4934828032 bytes (4.60 GiB) | 4934828032 bytes (4.60 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-12-measured-run-01 | passed | 1010.7 ms | 1194.5 ms | 2243.4 ms | 6.93 s | 4934828032 bytes (4.60 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.measured.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-12-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 12 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-12 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `12`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1070.5 ms | 1070.5 ms | 1070.5 ms |
| `predict_duration` | 39.7 ms | 39.7 ms | 39.7 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 1001.0 ms | 1001.0 ms | 1001.0 ms |
| `total_to_decode` | 2111.6 ms | 2111.6 ms | 2111.6 ms |
| `wall_seconds` | 6.74 s | 6.74 s | 6.74 s |
| `max_rss_bytes` | 4930142208 bytes (4.59 GiB) | 4930142208 bytes (4.59 GiB) | 4930142208 bytes (4.59 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-12-warmup-run-01 | passed | 1001.0 ms | 1070.5 ms | 2111.6 ms | 6.74 s | 4930142208 bytes (4.59 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.warmup.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-12-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-12.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 12 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-16 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `16`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1140.3 ms | 1140.3 ms | 1140.3 ms |
| `predict_duration` | 35.0 ms | 35.0 ms | 35.0 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1316.0 ms | 1316.0 ms | 1316.0 ms |
| `total_to_decode` | 2491.6 ms | 2491.6 ms | 2491.6 ms |
| `wall_seconds` | 7.14 s | 7.14 s | 7.14 s |
| `max_rss_bytes` | 4941660160 bytes (4.60 GiB) | 4941660160 bytes (4.60 GiB) | 4941660160 bytes (4.60 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-16-measured-run-01 | passed | 1316.0 ms | 1140.3 ms | 2491.6 ms | 7.14 s | 4941660160 bytes (4.60 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.measured.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-16-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 16 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-16 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `16`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1204.3 ms | 1204.3 ms | 1204.3 ms |
| `predict_duration` | 38.5 ms | 38.5 ms | 38.5 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1370.0 ms | 1370.0 ms | 1370.0 ms |
| `total_to_decode` | 2613.0 ms | 2613.0 ms | 2613.0 ms |
| `wall_seconds` | 7.58 s | 7.58 s | 7.58 s |
| `max_rss_bytes` | 4928847872 bytes (4.59 GiB) | 4928847872 bytes (4.59 GiB) | 4928847872 bytes (4.59 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-16-warmup-run-01 | passed | 1370.0 ms | 1204.3 ms | 2613.0 ms | 7.58 s | 4928847872 bytes (4.59 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.warmup.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-16-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-16.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 16 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-24 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `24`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 971.6 ms | 971.6 ms | 971.6 ms |
| `predict_duration` | 44.3 ms | 44.3 ms | 44.3 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1885.3 ms | 1885.3 ms | 1885.3 ms |
| `total_to_decode` | 2901.6 ms | 2901.6 ms | 2901.6 ms |
| `wall_seconds` | 7.36 s | 7.36 s | 7.36 s |
| `max_rss_bytes` | 4918493184 bytes (4.58 GiB) | 4918493184 bytes (4.58 GiB) | 4918493184 bytes (4.58 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-24-measured-run-01 | passed | 1885.3 ms | 971.6 ms | 2901.6 ms | 7.36 s | 4918493184 bytes (4.58 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.measured.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-24-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 24 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-24 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `24`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1037.8 ms | 1037.8 ms | 1037.8 ms |
| `predict_duration` | 37.8 ms | 37.8 ms | 37.8 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 1900.5 ms | 1900.5 ms | 1900.5 ms |
| `total_to_decode` | 2976.5 ms | 2976.5 ms | 2976.5 ms |
| `wall_seconds` | 7.58 s | 7.58 s | 7.58 s |
| `max_rss_bytes` | 4955848704 bytes (4.62 GiB) | 4955848704 bytes (4.62 GiB) | 4955848704 bytes (4.62 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-24-warmup-run-01 | passed | 1900.5 ms | 1037.8 ms | 2976.5 ms | 7.58 s | 4955848704 bytes (4.62 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.warmup.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-24-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-24.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 24 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-40 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `40`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1033.7 ms | 1033.7 ms | 1033.7 ms |
| `predict_duration` | 39.4 ms | 39.4 ms | 39.4 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 3156.2 ms | 3156.2 ms | 3156.2 ms |
| `total_to_decode` | 4229.6 ms | 4229.6 ms | 4229.6 ms |
| `wall_seconds` | 8.83 s | 8.83 s | 8.83 s |
| `max_rss_bytes` | 4988059648 bytes (4.65 GiB) | 4988059648 bytes (4.65 GiB) | 4988059648 bytes (4.65 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-40-measured-run-01 | passed | 3156.2 ms | 1033.7 ms | 4229.6 ms | 8.83 s | 4988059648 bytes (4.65 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.measured.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-40-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 40 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-40 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `40`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1036.7 ms | 1036.7 ms | 1036.7 ms |
| `predict_duration` | 39.8 ms | 39.8 ms | 39.8 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 3136.4 ms | 3136.4 ms | 3136.4 ms |
| `total_to_decode` | 4213.4 ms | 4213.4 ms | 4213.4 ms |
| `wall_seconds` | 8.73 s | 8.73 s | 8.73 s |
| `max_rss_bytes` | 4951949312 bytes (4.61 GiB) | 4951949312 bytes (4.61 GiB) | 4951949312 bytes (4.61 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-40-warmup-run-01 | passed | 3136.4 ms | 1036.7 ms | 4213.4 ms | 8.73 s | 4951949312 bytes (4.61 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.warmup.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-40-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-40.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 40 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-8 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `8`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1099.3 ms | 1099.3 ms | 1099.3 ms |
| `predict_duration` | 40.2 ms | 40.2 ms | 40.2 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.3 ms | 0.3 ms | 0.3 ms |
| `sample_rf` | 678.6 ms | 678.6 ms | 678.6 ms |
| `total_to_decode` | 1818.5 ms | 1818.5 ms | 1818.5 ms |
| `wall_seconds` | 6.37 s | 6.37 s | 6.37 s |
| `max_rss_bytes` | 4926013440 bytes (4.59 GiB) | 4926013440 bytes (4.59 GiB) | 4926013440 bytes (4.59 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-8-measured-run-01 | passed | 678.6 ms | 1099.3 ms | 1818.5 ms | 6.37 s | 4926013440 bytes (4.59 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.measured.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.measured.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-8-measured-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.measured.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 8 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```

## mlx-bridge-v3-no-reference-predicted-steps-8 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `8`
- Seconds: predicted duration (`--seconds` omitted)
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 1168.0 ms | 1168.0 ms | 1168.0 ms |
| `predict_duration` | 42.6 ms | 42.6 ms | 42.6 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 1.2 ms | 1.2 ms | 1.2 ms |
| `sample_rf` | 685.3 ms | 685.3 ms | 685.3 ms |
| `total_to_decode` | 1897.1 ms | 1897.1 ms | 1897.1 ms |
| `wall_seconds` | 7.68 s | 7.68 s | 7.68 s |
| `max_rss_bytes` | 4928667648 bytes (4.59 GiB) | 4928667648 bytes (4.59 GiB) | 4928667648 bytes (4.59 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-v3-no-reference-predicted-steps-8-warmup-run-01 | passed | 685.3 ms | 1168.0 ms | 1897.1 ms | 7.68 s | 4928667648 bytes (4.59 GiB) | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.warmup.stdout.log` | `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.warmup.stderr.log` |

### mlx-bridge-v3-no-reference-predicted-steps-8-warmup-run-01

- Output WAV: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.warmup.run-01.wav`
- CWD: `/Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets`

Command:

```bash
/usr/bin/time -l /Users/kouka/.openclaw/workspace/repos/irodori-tts-mlx/.venv/bin/python scripts/generate_wav.py --weights /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/irodori-v3.npz --output /Users/kouka/.openclaw/workspace/repos/_worktrees/irodori-tts-mlx/issue-64-num-steps-presets/benchmark-runs/issue-64-v3-text/mlx-bridge-v3-no-reference-predicted-steps-8.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 8 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --no-reference --model-config-json /Users/kouka/.openclaw/workspace/tmp/irodori-v3-smoke/v3-model-config.json
```
