# Apple Silicon PyTorch Baseline Report: 2026-05-11

Related issue: [#2](https://github.com/t0yohei/Irodori-TTS-MLX/issues/2)

## Summary

- Baseline status: passed with limitations
- Host: Apple Silicon Mac, macOS 26.4.1, arm64
- Upstream Irodori-TTS commit: `2708d3cadf726d4389d25eb4bb7a0344517a9a40`
- PyTorch: `2.10.0`
- MPS: built and available
- Main result:
  - no-reference base-model inference succeeded on MPS
  - reference-audio base-model inference succeeded on MPS using the no-reference output as a local bootstrap reference
- Known limitation: generated WAV artifacts are intentionally not committed; only local paths, audio metadata, timings, and SHA-256 hashes are recorded.

## Environment

Environment collected from the upstream `uv` environment:

```json
{
  "python": "3.10.20 (main, Mar 25 2026, 03:21:46) [Clang 22.1.1 ]",
  "platform": "macOS-26.4.1-arm64-arm-64bit",
  "mac_ver": "26.4.1",
  "machine": "arm64",
  "torch": "2.10.0",
  "mps_built": true,
  "mps_available": true,
  "git_commit": "2708d3cadf726d4389d25eb4bb7a0344517a9a40"
}
```

Repository helper script was also run successfully:

```bash
python3 scripts/collect_baseline_env.py > baseline-runs/env-repo.json
```

Additional notes:

- Hugging Face cache state: cold for the first no-reference run, warm for the second reference-audio run.
- Memory measurement method: `/usr/bin/time -l`, using `maximum resident set size`.
- Subjective audio quality was not evaluated in this report; this report records reproducibility, timings, and artifact metadata.

## Upstream source

- Repository: <https://github.com/Aratako/Irodori-TTS>
- Commit: `2708d3cadf726d4389d25eb4bb7a0344517a9a40`
- Install command:

```bash
uv sync
```

## Checkpoints

| Component | Identifier | Resolved revision / file | Notes |
| --- | --- | --- | --- |
| Irodori-TTS base model | `Aratako/Irodori-TTS-500M-v2` | `8fd631cafb911dde466bc30dd558a0dc55e8ccae/model.safetensors` | downloaded by `infer.py` |
| DACVAE codec | `Aratako/Semantic-DACVAE-Japanese-32dim` | `47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth` | downloaded by codec loader |

## Common sampling parameters

| Parameter | Value |
| --- | --- |
| `--text` | `今日はいい天気ですね。` |
| `--model-device` | `mps` |
| `--codec-device` | `mps` |
| `--model-precision` | `fp32` |
| `--codec-precision` | `fp32` |
| `--num-steps` | `40` |
| `--seed` | `20260511` |
| `--show-timings` | enabled |

## Run 1: base model, no reference audio

Command:

```bash
/usr/bin/time -l uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav ../../baseline-runs/base-no-ref-seed-20260511.wav \
  --model-device mps \
  --codec-device mps \
  --model-precision fp32 \
  --codec-precision fp32 \
  --num-steps 40 \
  --seed 20260511 \
  --show-timings
```

Result:

- Status: passed
- Output path: `baseline-runs/base-no-ref-seed-20260511.wav`
- SHA-256: `c93e567062be29a0543993244555845f783b5d7f90688ec0259167470522540f`
- Used seed: `20260511`
- Audio: WAV, PCM 16-bit, mono, 48,000 Hz, 213,120 frames, 4.44 seconds
- Wall clock: 122.86 seconds
- Maximum resident set size: 1,718,976,512 bytes (about 1.60 GiB)

Timing:

| Stage | Time |
| --- | ---: |
| tokenize_text | 1.8 ms |
| prepare_reference | 0.7 ms |
| sample_rf | 23,713.9 ms |
| unpatchify_latent | 0.0 ms |
| decode_latent | 5,648.5 ms |
| total_to_decode | 29.367 s |
| wall clock | 122.86 s |

Warnings or limitations:

```text
FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
```

The wall-clock time includes first-run model and codec downloads. Use `total_to_decode` for the post-load inference baseline.

## Run 2: base model, reference audio

Reference audio:

- Source: local output from Run 1, `baseline-runs/base-no-ref-seed-20260511.wav`
- License / sharing status: generated local artifact; not committed
- Audio: WAV, PCM 16-bit, mono, 48,000 Hz, 213,120 frames, 4.44 seconds

Command:

```bash
/usr/bin/time -l uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav ../../baseline-runs/base-no-ref-seed-20260511.wav \
  --output-wav ../../baseline-runs/base-ref-from-no-ref-seed-20260511.wav \
  --model-device mps \
  --codec-device mps \
  --model-precision fp32 \
  --codec-precision fp32 \
  --num-steps 40 \
  --seed 20260511 \
  --max-ref-seconds 30 \
  --ref-normalize-db -16 \
  --show-timings
```

Result:

- Status: passed
- Output path: `baseline-runs/base-ref-from-no-ref-seed-20260511.wav`
- SHA-256: `33a7109a40e814be43d92e0c12ed9f70efe0ada992a08bf4222ac734d960354c`
- Used seed: `20260511`
- Audio: WAV, PCM 16-bit, mono, 48,000 Hz, 213,120 frames, 4.44 seconds
- Wall clock: 40.77 seconds
- Maximum resident set size: 2,216,558,592 bytes (about 2.06 GiB)

Timing:

| Stage | Time |
| --- | ---: |
| tokenize_text | 2.8 ms |
| prepare_reference | 1,399.6 ms |
| sample_rf | 24,285.4 ms |
| unpatchify_latent | 0.0 ms |
| decode_latent | 5,637.5 ms |
| total_to_decode | 31.327 s |
| wall clock | 40.77 s |

Warnings or limitations:

```text
FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
```

## Artifacts

Generated artifacts were saved locally but are not committed to git.

| Artifact | Local path | SHA-256 | Committed? | Notes |
| --- | --- | --- | --- | --- |
| no-reference WAV | `baseline-runs/base-no-ref-seed-20260511.wav` | `c93e567062be29a0543993244555845f783b5d7f90688ec0259167470522540f` | no | source for reference-audio run |
| reference-audio WAV | `baseline-runs/base-ref-from-no-ref-seed-20260511.wav` | `33a7109a40e814be43d92e0c12ed9f70efe0ada992a08bf4222ac734d960354c` | no | generated from local bootstrap reference |
| no-reference console log | `baseline-runs/base-no-ref-seed-20260511.log` | not recorded | no | summarized above |
| reference-audio console log | `baseline-runs/base-ref-from-no-ref-seed-20260511.log` | not recorded | no | summarized above |

## Conclusions for MLX work

- Reproducible upstream baseline command: the no-reference command in Run 1.
- Baseline post-load no-reference latency target: 29.367 seconds to decode on PyTorch MPS for the fixed command above.
- Baseline post-load reference-audio latency target: 31.327 seconds to decode on PyTorch MPS for the fixed command above.
- The RF sampler is the dominant measured stage: about 23.7-24.3 seconds in these two runs.
- DACVAE decode is the next major stage: about 5.6 seconds.
- For the first MLX prototype, compare against `total_to_decode` instead of cold wall-clock time, and keep checkpoint/cache state separate from inference timing.
