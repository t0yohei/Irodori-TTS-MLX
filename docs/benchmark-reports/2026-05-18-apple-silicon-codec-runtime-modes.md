# Apple Silicon codec runtime mode benchmark

Issue: [#193 Update benchmarks by codec runtime mode](https://github.com/t0yohei/Irodori-TTS-MLX/issues/193)

## Summary

This measured the v3 hosted RF-DiT artifact with the approved hosted DACVAE codec artifact across the current codec runtime modes.

- RF-DiT weights repo: `t0yohei/Irodori-TTS-MLX-500M-v3`
- RF-DiT weights revision: `078ffb11ffad92e6dde237a6abef730f4341b359`
- DACVAE codec artifact repo: `t0yohei/Irodori-TTS-MLX-DACVAE-Codec`
- DACVAE codec artifact revision: `bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78`
- benchmark commit: `2d5bb1971ca4dd1f6be421c588dc16bdbf7f4b27`

The practical result is that MLX codec modes mainly reduce dependency surface and peak RSS. They do not make RF sampling faster, and the current no-reference MLX decode path is not consistently faster than the PyTorch bridge for decode latency. For reference-audio generation, full `mlx` encode/decode removed the PyTorch DACVAE bridge from the measured path, reduced max RSS by about 1.3 GiB versus the warm PyTorch bridge reference run, and reduced codec encode time substantially.

## Environment

- machine: Apple Silicon arm64
- OS: macOS 26.5
- benchmark Python: Python 3.11.15
- mlx: 0.31.2
- torch: 2.12.0
- huggingface_hub: 0.36.2
- numpy: 2.4.5
- soundfile: 0.13.1
- sentencepiece: 0.2.1
- upstream runtime import path: /path/to/Irodori-TTS
- benchmark harness: /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py
- reference audio: fixed local validation clip, not redistributed

## Benchmark configuration

- text: 今日はいい天気ですね。
- checkpoint family: v3
- duration mode: predicted duration (`--seconds` omitted)
- num_steps: 12
- seed: 20260512
- codec repo: `Aratako/Semantic-DACVAE-Japanese-32dim`
- codec device for PyTorch fallback: cpu
- cache labels: `cold` and `warm` are process-order labels within a warmed local Hugging Face cache, except where noted below

The setup/load overhead below is derived as wall clock minus `total_to_decode`. It includes Python startup, hosted layout validation, Hugging Face snapshot resolution, MLX weight loading, tokenizer/model initialization, codec construction, and other pre-generation work. It is practical user-perceived setup overhead, not a pure low-level deserialization timer.

## Results

| Scenario | Codec runtime mode | Cache | Codec encode | sample_rf | Codec decode | total_to_decode | Wall | Setup/load overhead | Max RSS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 no-reference | `persistent` PyTorch bridge | cold | n/a | 885.2 ms | 1053.1 ms | 1974.8 ms | 7.61 s | 5635.2 ms | 4.59 GiB |
| v3 no-reference | `persistent` PyTorch bridge | warm | n/a | 901.9 ms | 945.2 ms | 1887.7 ms | 7.85 s | 5962.3 ms | 4.56 GiB |
| v3 no-reference | `mlx-decode` hosted codec artifact | cold | n/a | 877.2 ms | 1745.9 ms | 2657.9 ms | 8.08 s | 5422.1 ms | 2.93 GiB |
| v3 no-reference | `mlx-decode` hosted codec artifact | warm | n/a | 881.6 ms | 853.6 ms | 1769.4 ms | 6.93 s | 5160.6 ms | 2.93 GiB |
| v3 reference audio | `persistent` PyTorch bridge | cold | 979.0 ms | 688.4 ms | 775.7 ms | 2551.5 ms | 8.38 s | 5828.5 ms | 4.26 GiB |
| v3 reference audio | `persistent` PyTorch bridge | warm | 156.6 ms | 758.5 ms | 800.7 ms | 1753.8 ms | 7.21 s | 5456.2 ms | 4.33 GiB |
| v3 reference audio | `mlx` hosted codec artifact | warm | 40.0 ms | 744.0 ms | 654.2 ms | 1466.4 ms | 6.74 s | 5273.6 ms | 3.08 GiB |

## Interpretation

- RF sampling time stayed in the same band across codec modes, as expected; codec runtime selection does not change RF-DiT sampling.
- On no-reference generation, `mlx-decode` removed the PyTorch DACVAE dependency from the generation path and reduced max RSS from about 4.56-4.59 GiB to 2.93 GiB. Warm decode latency was slightly lower than the PyTorch bridge in this run, while the first `mlx-decode` measured run was slower.
- On reference-audio generation, full `mlx` encode/decode is the cleanest dependency-surface result: metadata reports both `codec_encode_backend: mlx` and `codec_decode_backend: mlx`.
- The warm reference-audio `mlx` run reduced codec encode from 156.6 ms to 40.0 ms, codec decode from 800.7 ms to 654.2 ms, total_to_decode from 1753.8 ms to 1466.4 ms, and max RSS from 4.33 GiB to 3.08 GiB versus the warm PyTorch bridge run.
- The hosted codec artifact path is therefore mainly a deployment simplicity and memory-pressure improvement today. Treat speed as a secondary benefit until repeated larger prompts and longer outputs show a stable decode advantage.

## Hosted artifact cold/warm note

The warm-cache runs above use pinned hosted artifact revisions and a pre-existing authenticated local Hugging Face cache. An attempted empty-`HF_HOME` cold hosted download without an inherited token failed with Hugging Face 401 for the gated v3 weights repository. With authentication available, the existing [hosted weights loading benchmark](2026-05-16-apple-silicon-hosted-weights.md) remains the baseline for first-download setup cost: hosted artifact cold download dominates wall time, while warm hosted paths behave like local hosted-layout or local `.npz` paths.

For #193, the codec-mode comparison should therefore be read as process cold/warm within an authenticated cache, plus the separate hosted-loading baseline for first-download behavior.

## Commands

PyTorch bridge no-reference:

    PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
      --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
      --text '今日はいい天気ですね。' \
      --omit-seconds \
      --num-steps 12 \
      --seed 20260512 \
      --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
      --codec-device cpu \
      --codec-runtime-mode persistent \
      --repeat 2 \
      --cache-state auto \
      --case-label v3-pytorch-bridge-no-reference

MLX decode no-reference:

    PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
      --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
      --text '今日はいい天気ですね。' \
      --omit-seconds \
      --num-steps 12 \
      --seed 20260512 \
      --codec-runtime-mode mlx-decode \
      --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
      --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
      --repeat 2 \
      --cache-state auto \
      --case-label v3-mlx-decode-no-reference

PyTorch bridge reference-audio:

    PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
      --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
      --text '今日はいい天気ですね。' \
      --reference-wav /path/to/reference.wav \
      --omit-seconds \
      --num-steps 12 \
      --seed 20260512 \
      --codec-repo Aratako/Semantic-DACVAE-Japanese-32dim \
      --codec-device cpu \
      --codec-runtime-mode persistent \
      --repeat 2 \
      --cache-state auto \
      --case-label v3-pytorch-bridge-reference

MLX encode/decode reference-audio:

    PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
    /path/to/Irodori-TTS-MLX/.venv/bin/python scripts/benchmark.py \
      --mode mlx \
      --mlx-python /path/to/Irodori-TTS-MLX/.venv/bin/python \
      --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
      --weights-revision 078ffb11ffad92e6dde237a6abef730f4341b359 \
      --text '今日はいい天気ですね。' \
      --reference-wav /path/to/reference.wav \
      --omit-seconds \
      --num-steps 12 \
      --seed 20260512 \
      --codec-runtime-mode mlx \
      --codec-artifact-repo t0yohei/Irodori-TTS-MLX-DACVAE-Codec \
      --codec-artifact-revision bb89840af0deb729cc7a8e4ba5ebddb49e2b3e78 \
      --repeat 1 \
      --cache-state warm \
      --case-label v3-mlx-encode-decode-reference

