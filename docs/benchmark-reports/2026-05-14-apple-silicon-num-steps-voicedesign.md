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
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12 | measured | warm | 1 | 582.8 ms | 582.8 ms / 582.8 ms | 480.5 ms | 1063.8 ms | 6.21 s | 4220895232 bytes (3.93 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12 | warmup | cold | 1 | 626.7 ms | 626.7 ms / 626.7 ms | 528.6 ms | 1155.8 ms | 6.37 s | 4222828544 bytes (3.93 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16 | measured | warm | 1 | 811.1 ms | 811.1 ms / 811.1 ms | 530.0 ms | 1341.6 ms | 6.65 s | 4230348800 bytes (3.94 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16 | warmup | cold | 1 | 788.1 ms | 788.1 ms / 788.1 ms | 536.4 ms | 1324.9 ms | 6.52 s | 4231151616 bytes (3.94 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24 | measured | warm | 1 | 1163.9 ms | 1163.9 ms / 1163.9 ms | 538.8 ms | 1703.1 ms | 7.06 s | 4245159936 bytes (3.95 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24 | warmup | cold | 1 | 1145.3 ms | 1145.3 ms / 1145.3 ms | 530.6 ms | 1676.4 ms | 7.04 s | 4246011904 bytes (3.95 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40 | measured | warm | 1 | 1898.2 ms | 1898.2 ms / 1898.2 ms | 491.3 ms | 2390.0 ms | 7.85 s | 4279058432 bytes (3.99 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40 | warmup | cold | 1 | 1852.2 ms | 1852.2 ms / 1852.2 ms | 520.9 ms | 2373.6 ms | 7.72 s | 4279664640 bytes (3.99 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8 | measured | warm | 1 | 471.5 ms | 471.5 ms / 471.5 ms | 507.3 ms | 979.3 ms | 6.11 s | 4214374400 bytes (3.92 GiB) |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8 | warmup | cold | 1 | 436.3 ms | 436.3 ms / 436.3 ms | 508.0 ms | 944.8 ms | 6.68 s | 4212834304 bytes (3.92 GiB) |

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `12`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 480.5 ms | 480.5 ms | 480.5 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 582.8 ms | 582.8 ms | 582.8 ms |
| `total_to_decode` | 1063.8 ms | 1063.8 ms | 1063.8 ms |
| `wall_seconds` | 6.21 s | 6.21 s | 6.21 s |
| `max_rss_bytes` | 4220895232 bytes (3.93 GiB) | 4220895232 bytes (3.93 GiB) | 4220895232 bytes (3.93 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12-measured-run-01 | passed | 582.8 ms | 480.5 ms | 1063.8 ms | 6.21 s | 4220895232 bytes (3.93 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.measured.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.measured.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12-measured-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.measured.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 12 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `12`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 528.6 ms | 528.6 ms | 528.6 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 626.7 ms | 626.7 ms | 626.7 ms |
| `total_to_decode` | 1155.8 ms | 1155.8 ms | 1155.8 ms |
| `wall_seconds` | 6.37 s | 6.37 s | 6.37 s |
| `max_rss_bytes` | 4222828544 bytes (3.93 GiB) | 4222828544 bytes (3.93 GiB) | 4222828544 bytes (3.93 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12-warmup-run-01 | passed | 626.7 ms | 528.6 ms | 1155.8 ms | 6.37 s | 4222828544 bytes (3.93 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.warmup.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.warmup.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12-warmup-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.warmup.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-12.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 12 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `16`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 530.0 ms | 530.0 ms | 530.0 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 811.1 ms | 811.1 ms | 811.1 ms |
| `total_to_decode` | 1341.6 ms | 1341.6 ms | 1341.6 ms |
| `wall_seconds` | 6.65 s | 6.65 s | 6.65 s |
| `max_rss_bytes` | 4230348800 bytes (3.94 GiB) | 4230348800 bytes (3.94 GiB) | 4230348800 bytes (3.94 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16-measured-run-01 | passed | 811.1 ms | 530.0 ms | 1341.6 ms | 6.65 s | 4230348800 bytes (3.94 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.measured.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.measured.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16-measured-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.measured.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 16 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `16`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 536.4 ms | 536.4 ms | 536.4 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 788.1 ms | 788.1 ms | 788.1 ms |
| `total_to_decode` | 1324.9 ms | 1324.9 ms | 1324.9 ms |
| `wall_seconds` | 6.52 s | 6.52 s | 6.52 s |
| `max_rss_bytes` | 4231151616 bytes (3.94 GiB) | 4231151616 bytes (3.94 GiB) | 4231151616 bytes (3.94 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16-warmup-run-01 | passed | 788.1 ms | 536.4 ms | 1324.9 ms | 6.52 s | 4231151616 bytes (3.94 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.warmup.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.warmup.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16-warmup-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.warmup.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-16.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 16 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `24`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 538.8 ms | 538.8 ms | 538.8 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.4 ms | 0.4 ms | 0.4 ms |
| `sample_rf` | 1163.9 ms | 1163.9 ms | 1163.9 ms |
| `total_to_decode` | 1703.1 ms | 1703.1 ms | 1703.1 ms |
| `wall_seconds` | 7.06 s | 7.06 s | 7.06 s |
| `max_rss_bytes` | 4245159936 bytes (3.95 GiB) | 4245159936 bytes (3.95 GiB) | 4245159936 bytes (3.95 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24-measured-run-01 | passed | 1163.9 ms | 538.8 ms | 1703.1 ms | 7.06 s | 4245159936 bytes (3.95 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.measured.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.measured.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24-measured-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.measured.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 24 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `24`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 530.6 ms | 530.6 ms | 530.6 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 1145.3 ms | 1145.3 ms | 1145.3 ms |
| `total_to_decode` | 1676.4 ms | 1676.4 ms | 1676.4 ms |
| `wall_seconds` | 7.04 s | 7.04 s | 7.04 s |
| `max_rss_bytes` | 4246011904 bytes (3.95 GiB) | 4246011904 bytes (3.95 GiB) | 4246011904 bytes (3.95 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24-warmup-run-01 | passed | 1145.3 ms | 530.6 ms | 1676.4 ms | 7.04 s | 4246011904 bytes (3.95 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.warmup.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.warmup.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24-warmup-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.warmup.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-24.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 24 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `40`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 491.3 ms | 491.3 ms | 491.3 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 1898.2 ms | 1898.2 ms | 1898.2 ms |
| `total_to_decode` | 2390.0 ms | 2390.0 ms | 2390.0 ms |
| `wall_seconds` | 7.85 s | 7.85 s | 7.85 s |
| `max_rss_bytes` | 4279058432 bytes (3.99 GiB) | 4279058432 bytes (3.99 GiB) | 4279058432 bytes (3.99 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40-measured-run-01 | passed | 1898.2 ms | 491.3 ms | 2390.0 ms | 7.85 s | 4279058432 bytes (3.99 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.measured.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.measured.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40-measured-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.measured.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 40 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `40`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 520.9 ms | 520.9 ms | 520.9 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 1852.2 ms | 1852.2 ms | 1852.2 ms |
| `total_to_decode` | 2373.6 ms | 2373.6 ms | 2373.6 ms |
| `wall_seconds` | 7.72 s | 7.72 s | 7.72 s |
| `max_rss_bytes` | 4279664640 bytes (3.99 GiB) | 4279664640 bytes (3.99 GiB) | 4279664640 bytes (3.99 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40-warmup-run-01 | passed | 1852.2 ms | 520.9 ms | 2373.6 ms | 7.72 s | 4279664640 bytes (3.99 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.warmup.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.warmup.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40-warmup-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.warmup.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-40.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 40 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8 · measured · warm

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `8`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 507.3 ms | 507.3 ms | 507.3 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 471.5 ms | 471.5 ms | 471.5 ms |
| `total_to_decode` | 979.3 ms | 979.3 ms | 979.3 ms |
| `wall_seconds` | 6.11 s | 6.11 s | 6.11 s |
| `max_rss_bytes` | 4214374400 bytes (3.92 GiB) | 4214374400 bytes (3.92 GiB) | 4214374400 bytes (3.92 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8-measured-run-01 | passed | 471.5 ms | 507.3 ms | 979.3 ms | 6.11 s | 4214374400 bytes (3.92 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.measured.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.measured.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8-measured-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.measured.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.measured.run-01.wav --text '今日はいい天気ですね。' --num-steps 8 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```

## mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8 · warmup · cold

- Kind: `mlx`
- Reference mode: `no-reference`
- Num steps: `8`
- Seconds: `2.0`
- Runs: `1`
- Status counts: `{"passed": 1}`

Aggregate timings:

| Metric | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `decode_dacvae` | 508.0 ms | 508.0 ms | 508.0 ms |
| `prepare_reference_condition` | 0.0 ms | 0.0 ms | 0.0 ms |
| `prepare_text_condition` | 0.5 ms | 0.5 ms | 0.5 ms |
| `sample_rf` | 436.3 ms | 436.3 ms | 436.3 ms |
| `total_to_decode` | 944.8 ms | 944.8 ms | 944.8 ms |
| `wall_seconds` | 6.68 s | 6.68 s | 6.68 s |
| `max_rss_bytes` | 4212834304 bytes (3.92 GiB) | 4212834304 bytes (3.92 GiB) | 4212834304 bytes (3.92 GiB) |

Raw runs:

| Run | Status | sample_rf | decode | total_to_decode | wall | max RSS | stdout | stderr |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8-warmup-run-01 | passed | 436.3 ms | 508.0 ms | 944.8 ms | 6.68 s | 4212834304 bytes (3.92 GiB) | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.warmup.stdout.log` | `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.warmup.stderr.log` |

### mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8-warmup-run-01

- Output WAV: `/path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.warmup.run-01.wav`
- CWD: `/path/to/Irodori-TTS-MLX`

Command:

```bash
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz --output /path/to/Irodori-TTS-MLX/benchmark-runs/issue-64-voicedesign/mlx-bridge-voicedesign-caption-no-reference-seconds-2-steps-8.warmup.run-01.wav --text '今日はいい天気ですね。' --num-steps 8 --seed 20260512 --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim --codec-device cpu --codec-runtime-mode persistent --seconds 2.0 --no-reference --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json
```
