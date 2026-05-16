# Apple Silicon hosted weights loading benchmark

Issue: [#103 v0.2: Measure real hosted weights loading performance](https://github.com/t0yohei/Irodori-TTS-MLX/issues/103)

## Summary

This measured the first approved hosted/pre-converted MLX weights artifact from #83.

- hosted repo: t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign
- resolved snapshot: bf877a3beb7d921dc6bfb2b6812d02be07f39f2a
- upstream checkpoint family: Aratako/Irodori-TTS-500M-v2-VoiceDesign
- manifest family: voicedesign
- license review: approved
- benchmark commit: 3aa5a377c6a76bf4c8741c29367c67319851cea9

The practical result is straightforward: hosted loading changes setup and distribution UX, not steady-state generation latency. The first hosted run paid about three minutes of download/setup overhead for the 1.9 GiB artifact cache. Once cached, hosted repo, local hosted-layout directory, and direct local .npz fallback all stayed within the same small latency band for the measured VoiceDesign generation path.

## Environment

- machine: Apple M4
- OS: macOS 26.4.1 25E253
- architecture: arm64
- benchmark Python: Python 3.11.15
- mlx: 0.31.2
- torch: 2.11.0
- huggingface_hub: 0.36.2
- numpy: 2.4.5
- upstream runtime import path: /path/to/Irodori-TTS
- benchmark harness: /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py

## Benchmark configuration

These runs use the same short VoiceDesign configuration used by the existing local num_steps benchmark reports.

- text: 今日はいい天気ですね。
- caption: 落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。
- reference mode: no-reference
- seconds: 2
- num_steps: 24
- seed: 20260512
- codec repo: Aratako/Semantic-DACVAE-Japanese-32dim
- codec device: cpu
- codec runtime mode: persistent

## Results

The setup/load overhead below is derived as wall clock minus total_to_decode. It includes Python process startup, hosted-layout validation, Hugging Face snapshot resolution when applicable, MLX weight loading, tokenizer/model initialization, and other pre-generation work. It should be read as practical user-perceived setup overhead, not as a pure low-level .npz deserialization timer.

| Source | Cache state | Wall | Setup/load overhead | sample_rf | decode_dacvae | total_to_decode | Max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hosted repo | cold empty HF_HOME | 180.44 s | 178.58 s | 1221.8 ms | 636.4 ms | 1861.7 ms | 4.20 GiB |
| hosted repo | warm same HF_HOME | 8.74 s | 6.79 s | 1229.0 ms | 717.4 ms | 1946.9 ms | 3.96 GiB |
| local hosted-layout directory | warm local snapshot | 8.03 s | 6.17 s | 1225.9 ms | 636.7 ms | 1863.1 ms | 3.96 GiB |
| direct local .npz fallback | warm local file | 7.80 s | 5.88 s | 1259.0 ms | 659.6 ms | 1919.4 ms | 3.96 GiB |

Key read:

- first-run hosted repo overhead is dominated by downloading/resolving the 1.9 GiB hosted artifact
- warm hosted repo loading adds about 0.9 s over direct local .npz in this one-shot process benchmark
- local hosted-layout directory loading adds about 0.3 s over direct local .npz
- generation latency is effectively unchanged across all three warm paths; sample_rf stayed around 1.23 to 1.26 s and total_to_decode stayed around 1.86 to 1.95 s
- warm max RSS was the same practical value across hosted repo, local hosted-layout, and local .npz fallback

Compared with the existing [VoiceDesign 24-step local report](2026-05-14-apple-silicon-num-steps-voicedesign.md), this run is in the same range:

- prior local VoiceDesign 24-step run: sample_rf 1163.9 ms, total_to_decode 1703.1 ms, wall 7.06 s, max RSS 3.95 GiB
- this local .npz fallback run: sample_rf 1259.0 ms, total_to_decode 1919.4 ms, wall 7.80 s, max RSS 3.96 GiB

The difference is small enough that hosted loading should be treated as a setup/UX path, not as a generation-latency optimization.

## Artifact and cache size

- downloaded weights.npz: 1.9 GiB, 2045193660 bytes in the #83 publication note
- isolated first-run HF_HOME: 2.3 GiB
- hosted repo cache directory: 1.9 GiB
- hosted repo blobs directory: 1.9 GiB
- snapshot directory uses Hugging Face symlinks, so du -sh on the snapshot path itself reports 0B
- direct local fallback .npz: 1.9 GiB

## Setup-time impact versus manual local conversion

The hosted path does not make warm generation faster. Its value is that users no longer need to manually inspect the upstream checkpoint, run conversion, extract/write the compatible model_config.json, and keep local provenance metadata aligned before trying generation.

For first use, the hosted path replaces that manual conversion workflow with one approved artifact download plus layout validation. In this measurement that first hosted run took 180.44 s end to end, then subsequent hosted runs behaved like the local paths. This is a setup and reproducibility win, not a runtime optimization.

## Commands

Hosted repo cold/warm, using an empty isolated Hugging Face cache:

    mkdir -p benchmark-runs/issue-103-hosted
    HF_HOME="$PWD/benchmark-runs/issue-103-hosted/hf-home-cold" \
    PYTHONPATH="/path/to/Irodori-TTS:${PYTHONPATH:-}" \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
      --text '今日はいい天気ですね。' \
      --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' \
      --seconds 2 \
      --num-steps 24 \
      --seed 20260512 \
      --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
      --codec-device cpu \
      --codec-runtime-mode persistent \
      --repeat 2 \
      --cache-state auto \
      --case-label hosted-repo-voicedesign \
      --output-dir benchmark-runs/issue-103-hosted/hosted-repo \
      --report benchmark-runs/issue-103-hosted/hosted-repo-report.md

Local hosted-layout directory from the resolved snapshot:

    PYTHONPATH="/path/to/Irodori-TTS:${PYTHONPATH:-}" \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights-dir benchmark-runs/issue-103-hosted/hf-home-cold/hub/models--t0yohei--Irodori-TTS-MLX-500M-v2-VoiceDesign/snapshots/bf877a3beb7d921dc6bfb2b6812d02be07f39f2a \
      --text '今日はいい天気ですね。' \
      --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' \
      --seconds 2 \
      --num-steps 24 \
      --seed 20260512 \
      --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
      --codec-device cpu \
      --codec-runtime-mode persistent \
      --repeat 1 \
      --cache-state warm \
      --case-label local-hosted-layout-voicedesign \
      --output-dir benchmark-runs/issue-103-hosted/local-layout \
      --report benchmark-runs/issue-103-hosted/local-layout-report.md

Direct local .npz fallback:

    PYTHONPATH="/path/to/Irodori-TTS:${PYTHONPATH:-}" \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights /path/to/irodori-voicedesign-artifacts/irodori-voicedesign.npz \
      --model-config-json /path/to/irodori-voicedesign-artifacts/voicedesign-model-config.json \
      --text '今日はいい天気ですね。' \
      --caption '落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。' \
      --seconds 2 \
      --num-steps 24 \
      --seed 20260512 \
      --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
      --codec-device cpu \
      --codec-runtime-mode persistent \
      --repeat 1 \
      --cache-state warm \
      --case-label local-npz-fallback-voicedesign \
      --output-dir benchmark-runs/issue-103-hosted/local-npz \
      --report benchmark-runs/issue-103-hosted/local-npz-report.md

## Follow-ups

No generation-latency regression was found, so no runtime optimization issue is needed from this measurement.

Potential follow-ups, if the setup UX becomes important enough to optimize:

- make hosted download progress clearer in the user-facing CLI
- add an explicit metadata field or CLI output line for resolved hosted snapshot revision and cache path
- add a pure load-only benchmark mode if future reports need to separate snapshot resolution, checksum validation, .npz load, tokenizer initialization, and model construction
