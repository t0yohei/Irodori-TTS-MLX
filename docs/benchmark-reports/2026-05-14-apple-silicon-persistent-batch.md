# Apple Silicon persistent batch generation smoke benchmark

Related issue: [#66](https://github.com/t0yohei/Irodori-TTS-MLX/issues/66)

## Question

Can a local repeated-generation workflow reuse one initialized MLX + PyTorch DACVAE runtime and show a warm-state benefit for follow-up requests?

## Setup

- repo worktree: `/path/to/Irodori-TTS-MLX`
- Python: `/path/to/Irodori-TTS-MLX/.venv/bin/python`
- upstream checkout on `PYTHONPATH`: `/path/to/Irodori-TTS`
- weights: `/path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz`
- model config: `/path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json`
- request mode: two VoiceDesign caption-conditioned requests in one `scripts/generate_wav.py --requests-json` process
- preset: `fast` (12 RF sampling steps)
- duration: 2 seconds per request
- codec: `--codec-device cpu --codec-runtime-mode persistent`

## Repro command

```bash
cat > benchmark-runs/issue-66-persistent-batch/requests.json <<'JSON'
[
  {
    "text": "今日はいい天気ですね。",
    "caption": "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。",
    "output": "benchmark-runs/issue-66-persistent-batch/batch-01.wav",
    "seconds": 2,
    "preset": "fast",
    "seed": 20260514
  },
  {
    "text": "続けてもう一度、同じランタイムで生成します。",
    "caption": "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。",
    "output": "benchmark-runs/issue-66-persistent-batch/batch-02.wav",
    "seconds": 2,
    "preset": "fast",
    "seed": 20260515
  }
]
JSON

PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
/usr/bin/time -l /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/generate_wav.py \
  --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz \
  --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json \
  --no-reference \
  --codec-device cpu \
  --codec-runtime-mode persistent \
  --requests-json benchmark-runs/issue-66-persistent-batch/requests.json \
  --metadata-json benchmark-runs/issue-66-persistent-batch/metadata.json \
  --json
```

## Result

| Request | `prepare_text_condition` | `sample_rf` | `decode_dacvae` | `total_to_decode` |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.2 ms | 1823.1 ms | 822.0 ms | 2647.3 ms |
| 2 | 0.5 ms | 753.6 ms | 643.8 ms | 1397.9 ms |

Process-level observations:

- wall time for both requests in one process: **13.08 s**
- max resident set size: **1.84 GiB** (`1973747712` bytes)
- peak memory footprint from `/usr/bin/time -l`: **4.54 GiB** (`4878798808` bytes)

## Interpretation

The second request reused the same initialized `MLXDACVAERuntime`, including loaded RF-DiT weights, tokenizers, and the persistent DACVAE bridge. Within this two-request smoke run, the follow-up request reduced:

- `sample_rf` by **1069.4 ms** (**58.7%**)
- `total_to_decode` by **1249.4 ms** (**47.2%**)

This is not a production service benchmark and does not isolate every contributor to process-level wall time, but it does demonstrate the intended local workflow benefit: prompt/seed iteration can stay in one warm runtime instead of paying the full startup/load path for every generated WAV.

## Choice guidance

Use one-shot CLI mode for a single output, shell scripts that want independent process isolation, or commands where cold-start overhead is irrelevant. Use `--requests-json` batch mode when generating several variations from the same weights/model/codec configuration and when the request-specific fields are limited to text, output path, reference/caption, duration, preset/step count, CFG, seed, or cache knobs.
