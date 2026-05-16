# Irodori-TTS-MLX

[日本語 README](README.ja.md)

An unofficial MLX inference port of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) for Apple Silicon.

> [!IMPORTANT]
> This is an alpha inference prototype, not a polished product. The shortest supported path is local checkpoint inspection/conversion plus `scripts/generate_wav.py`; the MLX RF-DiT path can generate WAV files through the upstream PyTorch DACVAE bridge when you provide compatible local checkpoints and runtime dependencies. Model weights, upstream `irodori_tts`, DACVAE assets, checkpoint redistribution, training, Web UI, and a full MLX DACVAE port are out of scope.

## Current v0.1 scope

`irodori-tts-mlx` currently provides an Apple Silicon-focused path for:

- inspecting supported Irodori-TTS checkpoints without loading all tensor payloads
- converting supported `.safetensors` checkpoints into MLX-friendly `.npz` RF-DiT weights
- running MLX text/condition encoders, RF-DiT, and rectified-flow sampling
- encoding reference audio and decoding generated latents through upstream `irodori_tts` / PyTorch `DACVAECodec`
- writing generated WAV files with `scripts/generate_wav.py`
- benchmarking and validating the prototype through local scripts and hosted Apple Silicon workflows

The implementation boundary is still:

> MLX RF-DiT inference + PyTorch DACVAE encode/decode bridge

This keeps the MLX port focused on the model path most likely to benefit from Apple Silicon while relying on the upstream DACVAE implementation for audio-codec behavior.

## What works and what does not

Supported for the v0.1 prototype:

- base `Aratako/Irodori-TTS-500M-v2`-style checkpoints
- `Aratako/Irodori-TTS-500M-v2-VoiceDesign` caption-conditioned checkpoints
- `Aratako/Irodori-TTS-500M-v3` checkpoints with predicted-duration semantics when `--seconds` is omitted
- Python **3.11 through 3.14** packaging targets; Python 3.11 remains the benchmark reference environment

Not supported yet:

- training or fine-tuning
- full MLX DACVAE encode/decode
- checkpoint or generated-model redistribution
- GUI / Gradio / hosted demo
- broad compatibility with every historical or third-party Irodori-TTS checkpoint
- stable public Python API guarantees

## Architecture

```text
text prompt ───────────────┐
reference audio ── PyTorch DACVAE encode ──┐
caption/style text ────────┐               │
                            ▼               ▼
                     MLX encoders + RF-DiT sampler
                                      │
                                      ▼
                          generated DACVAE latents
                                      │
                                      ▼
                         PyTorch DACVAE decode
                                      │
                                      ▼
                                  output WAV
```

For more detail, see [docs/architecture.md](docs/architecture.md).

For the upstream PyTorch baseline workflow used to compare future MLX work, see [docs/baseline.md](docs/baseline.md).

For checkpoint metadata and state_dict layout notes that will guide weight conversion, see [docs/weight_mapping.md](docs/weight_mapping.md).

For optional upstream PyTorch vs MLX deterministic component parity tests, see [docs/pytorch_parity.md](docs/pytorch_parity.md).

For the end-to-end upstream PyTorch vs MLX generation parity harness, deterministic fixture mode, current v3/VoiceDesign baseline summary, and real-checkpoint commands, see [docs/upstream_parity_harness.md](docs/upstream_parity_harness.md).

For the initial MLX rectified-flow Euler sampler and CFG behavior, see [docs/rf_sampler.md](docs/rf_sampler.md).

For Apple Silicon benchmark workflow, current baseline conclusions, and the benchmark harness for upstream vs MLX bridge comparison, see [docs/benchmark.md](docs/benchmark.md).

For the first end-to-end MLX RF-DiT + PyTorch DACVAE bridge and WAV-generation CLI, see [docs/dacvae_bridge.md](docs/dacvae_bridge.md). For DACVAE decode-only parity fixture commands and metrics, see [docs/dacvae_decode_parity.md](docs/dacvae_decode_parity.md); for encode parity fixture commands, length/mask checks, and tolerances, see [docs/dacvae_encode_parity.md](docs/dacvae_encode_parity.md). For the v0.1 upstream dependency boundary and install choices, see [docs/upstream_dependency.md](docs/upstream_dependency.md).

For the v0.1 checkpoint-family support contract, including supported / experimental / unsupported status and redistribution caveats, see [docs/checkpoint_support.md](docs/checkpoint_support.md). For the v0.2 hosted pre-converted weights eligibility audit and required provenance language, see [docs/preconverted_weights_redistribution_audit.md](docs/preconverted_weights_redistribution_audit.md).

For the v0.2 hosted/pre-converted MLX weights layout contract, including local-directory versus Hugging Face repository resolution, required metadata files, provenance, and license-review boundaries, see [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md). For the separate hosted/local DACVAE codec artifact layout and PyTorch bridge fallback policy, see [docs/codec_artifact_layout.md](docs/codec_artifact_layout.md). For the user-facing hosted-weights quick path, local hosted-layout directory flow, and fallback local conversion recipe, see [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md).

For the mlx-audio interoperability evaluation, including the public `mlx-community/Irodori-TTS-*` artifact layout comparison, direct-loader boundary, DACVAE reference notes, and recommended adapter follow-ups, see [docs/mlx_audio_interop.md](docs/mlx_audio_interop.md). For the DACVAE-specific artifact comparison, shape conventions, local parity commands, and #131 ingestion recommendation, see [docs/mlx_audio_dacvae_contract.md](docs/mlx_audio_dacvae_contract.md).

For the v0.2 cross-repository delivery plan that maps Linear TOY-5 to the GitHub issue cluster, dependency order, validation gates, and downstream local-assistant/OpenClaw smoke path, see [docs/v0_2_delivery_plan.md](docs/v0_2_delivery_plan.md).

For the current `Aratako/Irodori-TTS-500M-v3` support statement, manual validation recipe, and hosted Apple Silicon coverage, see [docs/v3_support.md](docs/v3_support.md).

For the packaged install story, supported Python versions, and reproducible runtime / benchmark environment setup, see [docs/packaging.md](docs/packaging.md).

For the v0.1 text preprocessing contract, including upstream-compatible prompt normalization, tokenizer padding/truncation semantics, caption-tokenizer boundaries, and representative contract tests, see [docs/text_preprocessing.md](docs/text_preprocessing.md).

## v0.2 hosted converted weights path

The hosted-loader CLI work in #82 defines the intended v0.2 shortest path: load an approved hosted converted weights repository with `--weights-repo`, or the same layout from disk with `--weights-dir`. Those flags are not part of the current `main` CLI until the #82 implementation lands; on the current CLI, use the local conversion fallback below.

When the hosted-loader support is available, use only repos whose manifest has `license_review.status: "approved"` and whose README/model card links provenance for the exact upstream checkpoint revision. The planned VoiceDesign shape is:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --text "こんにちは。今日は良い天気です。" \
  --caption "落ち着いた女性の声" \
  --no-reference \
  --output /tmp/irodori-hosted.wav \
  --preset balanced
```

This path still uses the upstream PyTorch DACVAE bridge for codec encode/decode and does not bundle upstream code, codec weights, reference audio, generated samples, or Hugging Face cache snapshots. If a hosted repo is unavailable, unapproved, or outside the audited candidate families, use the local conversion fallback below. See [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) for the full hosted/local layout flow, v3 no-reference example, provenance checklist, and fallback decision rules.

VoiceDesign v2 checkpoints do not have the v3 duration predictor. When `--seconds` is omitted, the runtime estimates a bounded fallback from normalized text length and reports the resolved duration in metadata; pass `--seconds` when a specific prompt needs a manual duration to avoid clipping or an over-extended tail.

## Quickstart: checkpoint to WAV

This is the supported local fallback path from a fresh checkout to a generated WAV. It assumes an Apple Silicon macOS host and a checkpoint you are allowed to download and use locally. This repository does **not** redistribute upstream code, model weights, DACVAE assets, or reference audio; check the upstream repository and model cards before reusing or sharing outputs.

The recommended first smoke path is **v3 no-reference generation** because it does not require a committed reference WAV and it exercises the predicted-duration runtime. Use the reference-WAV variant below when you want speaker-conditioned output from a local sample.

### 1. Create the Python environment

```bash
git clone https://github.com/t0yohei/Irodori-TTS-MLX.git
cd Irodori-TTS-MLX

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime]" safetensors
```

Use Python 3.11 through 3.14. Python 3.11 is the reference version used by the project examples and benchmark notes. The explicit `safetensors` install is required for the non-dry-run checkpoint conversion step.

### 2. Make upstream `irodori_tts` importable

The WAV path still uses the upstream PyTorch DACVAE bridge. Clone/install upstream Irodori-TTS into the same environment, or point `PYTHONPATH` at an existing checkout:

```bash
git clone https://github.com/Aratako/Irodori-TTS.git ../Irodori-TTS
python -m pip install -e ../Irodori-TTS

# Alternative if you do not install it:
# export PYTHONPATH="$(pwd)/../Irodori-TTS:${PYTHONPATH}"
```

If your upstream checkout requires extra DACVAE/audio dependencies, install them in this same venv. See [docs/packaging.md](docs/packaging.md) and [docs/dacvae_bridge.md](docs/dacvae_bridge.md) for runtime dependency details.

### 3. Inspect and convert a local checkpoint

Download or otherwise place a supported upstream `model.safetensors` on disk. For v3, keep the upstream config metadata as a small JSON file so generation can enable the duration predictor:

```bash
CHECKPOINT=/path/to/Irodori-TTS-500M-v3/model.safetensors
WORK=/tmp/irodori-quickstart
mkdir -p "$WORK"

python scripts/inspect_checkpoint.py "$CHECKPOINT" --json > "$WORK/checkpoint-inspect.json"
python - "$WORK/checkpoint-inspect.json" > "$WORK/v3-model-config.json" <<'PY'
import json
import sys
from dataclasses import fields
from irodori_mlx.config import ModelConfig
payload = json.load(open(sys.argv[1]))
allowed = {field.name for field in fields(ModelConfig)}
config = {key: value for key, value in payload['config'].items() if key in allowed}
if config.get('use_duration_predictor') is not True:
    raise SystemExit('expected a v3 checkpoint config with use_duration_predictor=true')
print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
PY

python scripts/convert_weights.py "$CHECKPOINT" "$WORK/irodori-v3.npz" --dry-run --json \
  > "$WORK/convert-dry-run.json"
python scripts/convert_weights.py "$CHECKPOINT" "$WORK/irodori-v3.npz"
```

Expected files after this step:

- `$WORK/checkpoint-inspect.json` — checkpoint metadata and tensor header summary
- `$WORK/convert-dry-run.json` — converter validation report, including detected checkpoint family
- `$WORK/v3-model-config.json` — model config consumed by `generate_wav.py`
- `$WORK/irodori-v3.npz` — converted MLX weights

### 4. Generate a WAV

```bash
PYTHONPATH="$(pwd)/../Irodori-TTS:${PYTHONPATH}" \
python scripts/generate_wav.py \
  --weights "$WORK/irodori-v3.npz" \
  --model-config-json "$WORK/v3-model-config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output "$WORK/irodori-v3.wav" \
  --preset balanced \
  --metadata-json "$WORK/irodori-v3-metadata.json" \
  --json > "$WORK/irodori-v3-result.json"
```

Expected output files:

- `$WORK/irodori-v3.wav` — generated audio
- `$WORK/irodori-v3-metadata.json` — generation metadata, request fields, timings, and runtime boundary details
- `$WORK/irodori-v3-result.json` — stdout JSON for shell/CI pipelines

For this v3 smoke path, omit `--seconds` intentionally. Successful metadata should report `duration_mode: "predicted"`. Add `--seconds N` only when you want a manual duration override.

### Reference-WAV variant

For speaker-conditioned local checks, replace `--no-reference` with a local sample you have rights to use:

```bash
python scripts/generate_wav.py \
  --weights "$WORK/irodori-v3.npz" \
  --model-config-json "$WORK/v3-model-config.json" \
  --text "こんにちは。今日は良い天気です。" \
  --reference-wav /path/to/reference.wav \
  --output "$WORK/irodori-v3-reference.wav" \
  --preset balanced \
  --metadata-json "$WORK/irodori-v3-reference-metadata.json"
```

Base v2 checkpoints normally use this reference-audio path and can use the default model config unless your converted checkpoint requires an explicit `--model-config-json`.

### If the quickstart fails

- `No module named irodori_tts`: install upstream Irodori-TTS in the active venv or set `PYTHONPATH` to its checkout.
- converter rejects the checkpoint family or shapes: confirm the checkpoint is one of the supported inspected families in [docs/v3_support.md](docs/v3_support.md) and [docs/caption_condition_support.md](docs/caption_condition_support.md).
- `--reference-wav` / `--no-reference` validation errors: choose exactly the conditioning mode supported by your checkpoint/config; see [docs/dacvae_bridge.md](docs/dacvae_bridge.md).
- runtime dependency or audio I/O failures: revisit [docs/packaging.md](docs/packaging.md) and the upstream Irodori-TTS dependency setup.

## Supported Python and install targets

The current packaged environment supports **Python 3.11 through 3.14**.
Python 3.11 remains the reference environment for the published benchmark notes and examples in this repository.

Install this repo in editable mode depending on your use case:

```bash
python3.11 -m venv .venv  # or: python3.12/3.13/3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[runtime]"  # WAV generation / bridge runtime
python -m pip install -e ".[bench]"    # benchmark + conversion workflow
python -m pip install -e ".[dev]"      # local contributor environment
```

The bridge runtime still depends on upstream `irodori_tts` for `irodori_tts.codec.DACVAECodec` on the default `persistent` and `subprocess` codec modes. Prefer installing the upstream checkout into the same venv with `python -m pip install -e /path/to/Irodori-TTS`; use `PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-}` only when you intentionally want an uninstalled checkout. This is an intentional v0.1 boundary: the MLX repo owns the text/caption conditioning, RF-DiT, conversion, duration, and sampler path, while upstream still owns PyTorch DACVAE encode/decode. For v0.2 codec-port experiments, `--codec-runtime-mode mlx --codec-path /path/to/dacvae-codec.npz` uses the local MLX codec artifact contract documented in [docs/dacvae_bridge.md](docs/dacvae_bridge.md). See [docs/upstream_dependency.md](docs/upstream_dependency.md) and [docs/packaging.md](docs/packaging.md).

## README split

To keep both READMEs discoverable without letting them drift too far:

- [README.md](README.md) is the canonical source for exact technical scope, compatibility, and milestone status.
- [README.ja.md](README.ja.md) provides a Japanese overview, setup entry points, current limitations, and links into the detailed docs.
- Detailed procedures and validation notes should continue to live under `docs/*.md` so both READMEs can stay concise.

## v0.1 checkpoint family support

v0.1 support is limited to checkpoint families whose tensor layouts and runtime semantics are explicitly validated in this repository. The shorthand is:

| Checkpoint family | Example checkpoint | Inspect | Convert | Generate | v0.1 status |
| --- | --- | --- | --- | --- | --- |
| Base v2 speaker-conditioned | `Aratako/Irodori-TTS-500M-v2` | Supported | Supported | Experimental manual path | **Experimental** |
| VoiceDesign v2 caption-conditioned | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | Supported | Supported | Supported with `--caption` for the inspected public family | **Supported** |
| v3 speaker-conditioned / duration-predictor | `Aratako/Irodori-TTS-500M-v3` | Supported | Supported | Supported; omit `--seconds` for predicted duration | **Supported** |
| Other historical, fine-tuned, LoRA, architecture-modified, or renamed Irodori-TTS checkpoints | Any non-matching layout/config | Best-effort metadata inspection only | Unsupported | Unsupported | **Unsupported** |

Unsupported means outside the v0.1 conversion/runtime contract, not merely untested. This repository also does **not** redistribute checkpoints, Semantic-DACVAE weights, Hugging Face cache contents, converted `.npz` archives, or generated audio artifacts. Users must obtain upstream checkpoints themselves and follow the relevant upstream repository/model-card terms. See [docs/checkpoint_support.md](docs/checkpoint_support.md) for the full support matrix, family boundaries, validation evidence, and redistribution caveats, [docs/hosted_weights_layout.md](docs/hosted_weights_layout.md) for the v0.2 hosted MLX weights repository contract, [docs/hosted_weights_usage.md](docs/hosted_weights_usage.md) for hosted usage and fallback local conversion, and [docs/license_and_distribution.md](docs/license_and_distribution.md) for the repository license and non-redistribution policy.

## Checkpoint inspection

Use the installed `irodori-tts-inspect` command to inspect local or Hugging Face `model.safetensors` checkpoints without loading tensor payloads:

```bash
irodori-tts-inspect Aratako/Irodori-TTS-500M-v2
irodori-tts-inspect Aratako/Irodori-TTS-500M-v2 --json > checkpoint.json
irodori-tts-inspect /path/to/model.safetensors --all-tensors
```

The legacy `python3 scripts/inspect_checkpoint.py ...` path remains supported for repository checkouts.

The script prints metadata/config, tensor names, shapes, dtypes, and parameter totals for weight-converter planning.

## Weight conversion

Use the installed `irodori-tts-convert` command to convert a local base v2, v3, or VoiceDesign checkpoint into an MLX-friendly `.npz` archive:

```bash
irodori-tts-convert /path/to/model.safetensors /path/to/irodori-tts-500m-v2.npz
irodori-tts-convert /path/to/model.safetensors --dry-run
irodori-tts-convert /path/to/model.safetensors --dry-run --json
```

The legacy `python3 scripts/convert_weights.py ...` path remains supported for repository checkouts.

The converter now supports the base `Aratako/Irodori-TTS-500M-v2` layout, the `Aratako/Irodori-TTS-500M-v2-VoiceDesign` caption-conditioned layout, and the `Aratako/Irodori-TTS-500M-v3` duration-predictor layout. It validates the documented key mapping, shape expectations, float32 dtypes, and family-specific config assumptions before writing output. Use `--dry-run --json` to confirm the detected `checkpoint_family` before exporting large checkpoints. V3 is now supported through conversion plus the MLX bridge runtime, with duration semantics documented in [docs/dacvae_bridge.md](docs/dacvae_bridge.md) and reproducible validation coverage documented in [docs/v3_support.md](docs/v3_support.md). See [docs/caption_condition_support.md](docs/caption_condition_support.md) for the separate VoiceDesign support matrix.

The initial converter accepts only local `.safetensors` checkpoints. Converting them requires the optional `safetensors` Python package. Header-only `--dry-run` validation works without loading the multi-GiB tensor payload.

For standing integration coverage against the real public VoiceDesign checkpoint, use `scripts/run_voicedesign_integration.py` or the scheduled/manual GitHub Actions workflow in `.github/workflows/voicedesign-real-checkpoint.yml`. That lightweight automation validates inspect + converter family detection without forcing full `.npz` export on every run.

For the v0.1 release gate, use `scripts/run_v0_1_release_gate.py` or `.github/workflows/v0.1-release-gate.yml`; see [docs/v0_1_release_gate.md](docs/v0_1_release_gate.md). The required gate downloads the public v3 checkpoint, inspects it, converts it, runs no-reference predicted-duration WAV generation, validates JSON metadata, and preserves artifacts. VoiceDesign caption-conditioned generation is available as an optional heavier gate via `--include-optional-voicedesign` / the workflow input.

For focused full end-to-end hosted coverage of `scripts/generate_wav.py --caption ...`, use `scripts/run_voicedesign_generation_ci.py` or `.github/workflows/voicedesign-hosted-generation.yml`. For equivalent v3 coverage on the predicted-duration path, use `scripts/run_v3_generation_ci.py` or `.github/workflows/v3-hosted-generation.yml`. These workflows now target the standard GitHub-hosted Apple Silicon M1 runner (`macos-14`), so public-repository runs stay on the free hosted macOS tier without needing self-hosted infrastructure.

## Benchmarking

Use `scripts/benchmark.py` to orchestrate reproducible upstream PyTorch and MLX bridge timing runs, collect `/usr/bin/time -l` memory observations, repeat runs with warm/cold labeling, and emit Markdown + JSON summaries. Python 3.11 through 3.14 are supported for packaging, while Python 3.11 remains the recommended benchmark reference environment described in [docs/packaging.md](docs/packaging.md).


```bash
python3 scripts/benchmark.py --self-test
python3 scripts/benchmark.py --mode upstream --upstream-root /path/to/Irodori-TTS
python3 scripts/benchmark.py --mode mlx --weights /path/to/irodori-tts-500m-v2.npz --upstream-root /path/to/Irodori-TTS
python3 scripts/benchmark.py --mode mlx --weights /path/to/irodori-tts-500m-v2.npz --upstream-root /path/to/Irodori-TTS --repeat 3 --warmup-runs 1 --reference-wav /path/to/reference.wav
python3 scripts/benchmark.py --mode mlx --weights /path/to/irodori-tts-500m-v2.npz --upstream-root /path/to/Irodori-TTS --seconds-sweep 3,5,8 --num-steps-sweep 20,40
```

The MLX bridge runtime emits benchmark-friendly `[timing]` lines for text/reference preparation, RF sampling, DACVAE decode, and total inference time.
For reference-path memory experiments, `--codec-runtime-mode persistent|subprocess` can compare the normal in-process bridge against a helper-process DACVAE boundary.
The benchmark summary JSON now records per-run metadata (`phase`, `cache_state`, sweep parameters) plus aggregated min/median/max statistics so future reports can diff repeated runs instead of relying on one-off measurements.

## Core MLX layers

The `irodori_mlx.layers` module contains the first reusable MLX primitives for model parity work:

- `RMSNorm`
- RoPE frequency generation and application helpers
- sinusoidal timestep embeddings
- `SwiGLU`
- low-rank AdaLN modulation
- latent sequence patch/unpatch helpers

These implementations follow the upstream PyTorch formulas used by Irodori-TTS and keep normalization/embedding math in `float32` where practical. Floating-point inputs are cast back to their original dtype after operations such as RMSNorm and RoPE application, so future bf16 inference paths can keep bf16 activations while still using fp32 statistics for numerically sensitive steps.

## Condition encoders

The `irodori_mlx.encoders` module contains the first MLX conditioning stack:

- token `TextEncoder` for prompt text
- `ReferenceLatentEncoder` for base-checkpoint speaker/reference latent conditioning
- optional caption encoder wiring for VoiceDesign-style checkpoints
- `ConditionEncoders` wrapper for text, speaker, and caption masks/dropout
- narrow `.npz` weight assignment helpers for converted upstream encoder weights

Masked positions are hard-zeroed after embedding and after each residual block so fully masked conditioning becomes an unconditional path. Speaker/reference conditioning also patches the latent sequence when configured and prepends the upstream-style masked-mean summary token.

The first `irodori_mlx.model.TextToLatentRFDiT` forward path is now available for MLX model-parity work. It wires the condition encoders into joint RF-DiT attention, timestep-conditioned AdaLN blocks, static conditioning K/V projection caches, and final latent velocity projection. See [docs/rf_dit_forward.md](docs/rf_dit_forward.md) for implementation and numerical-comparison notes.

`irodori_mlx.sampling.sample_euler_rf_cfg` adds the first RF Euler sampling loop on top of the MLX model path. It can generate patched latent sequences with fixed-seed noise, upstream-style timesteps, optional context K/V cache, and text/speaker/caption CFG modes.

`irodori-tts-generate` (also available as the legacy `python scripts/generate_wav.py` path) and `irodori_mlx.runtime.MLXDACVAERuntime` provide the first prototype WAV-generation path: normalize and tokenize prompt text, encode reference audio with upstream/PyTorch DACVAE or a local MLX codec artifact, sample generated latents with MLX RF-DiT, decode them back to waveform, and save a WAV. The CLI now supports repeatable `--config-json` presets, user-facing `--preset fast|balanced|quality` step-count shortcuts, `--requests-json` persistent batch mode for repeated local generations that reuse one initialized runtime, `--codec-runtime-mode mlx --codec-path ...` for v0.2 DACVAE artifact contract tests, plus `--json` / `--metadata-json` output for automation-friendly metadata and timings. Caption-conditioned checkpoints can now use the documented conversion + runtime path as long as their metadata and tensor layout match the inspected VoiceDesign family, and the public `Aratako/Irodori-TTS-500M-v3` path is supported with predicted-duration semantics when `--seconds` is omitted. See [docs/dacvae_bridge.md](docs/dacvae_bridge.md) for dependencies, invocation patterns, preset mappings, persistent batch examples, and boundary notes, [docs/text_preprocessing.md](docs/text_preprocessing.md) for the v0.1 normalization/tokenization contract, [docs/caption_condition_support.md](docs/caption_condition_support.md) for the VoiceDesign support statement, and [docs/v3_support.md](docs/v3_support.md) for the v3 validation story.

## Public API direction

The project is currently CLI-first. `scripts/generate_wav.py`, `scripts/convert_weights.py`, `scripts/inspect_checkpoint.py`, and `scripts/benchmark.py` are the supported user entry points for local experimentation. Internal Python modules are available for the CLI and tests, but no stable public Python API is promised before v0.1 is finalized.

## Non-goals for v0.1

The v0.1 prototype does not include:

- training or fine-tuning support
- bundled Semantic-DACVAE weights or guaranteed acoustic parity for arbitrary MLX codec artifacts
- Gradio or web UI support
- model distribution or checkpoint redistribution
- broad compatibility with every historical Irodori-TTS checkpoint

## Related resources

- Upstream Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- License and distribution policy: [docs/license_and_distribution.md](docs/license_and_distribution.md)
- MLX: <https://github.com/ml-explore/mlx>
- Irodori-TTS 500M v2 model card: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2>
- Irodori-TTS 500M v2 VoiceDesign model card: <https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign>
- Irodori-TTS 500M v3 model card: <https://huggingface.co/Aratako/Irodori-TTS-500M-v3>
- Semantic-DACVAE Japanese 32-dim codec: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>
- DACVAE: <https://github.com/facebookresearch/dacvae>

## Roadmap

The current project milestones are organized as follows:

1. **M0 Baseline**: define scope and reproduce upstream PyTorch inference on Apple Silicon.
2. **M1 Weight conversion**: inspect checkpoints and implement PyTorch/safetensors to MLX weight conversion.
3. **M2 MLX model parity**: port model components and compare against PyTorch outputs.
4. **M3 MLX inference prototype**: generate audio with MLX RF-DiT and a PyTorch DACVAE bridge.
5. **M4 Performance and packaging**: benchmark, optimize, document, and package the prototype.

Current status by milestone:

- **M0 Baseline**: completed.
- **M1 Weight conversion**: completed, including VoiceDesign / caption-conditioned checkpoint conversion support from [#41 Add VoiceDesign / caption-conditioned checkpoint conversion support](https://github.com/t0yohei/Irodori-TTS-MLX/issues/41).
- **M2 MLX model parity**: completed for the currently supported checkpoint families, with VoiceDesign follow-up work captured and closed in [#33 Audit and expand VoiceDesign / caption-conditioned checkpoint support](https://github.com/t0yohei/Irodori-TTS-MLX/issues/33).
- **M3 MLX inference prototype**: completed for the current CLI/runtime scope, including generation UX follow-up from [#32 Improve the generation CLI and runtime UX](https://github.com/t0yohei/Irodori-TTS-MLX/issues/32), real-checkpoint integration coverage from [#44 Add real-checkpoint VoiceDesign integration coverage](https://github.com/t0yohei/Irodori-TTS-MLX/issues/44), and hosted Apple Silicon full-generation coverage from [#46 Add hosted Apple Silicon CI coverage for full VoiceDesign generation](https://github.com/t0yohei/Irodori-TTS-MLX/issues/46).
- **M4 Performance and packaging**: completed for the current prototype scope, including memory residency mitigation from [#29 Investigate and reduce reference-path memory residency in the MLX bridge](https://github.com/t0yohei/Irodori-TTS-MLX/issues/29), repeated benchmark automation from [#30 Extend benchmark automation for warm-cache, repeated runs, and scaling sweeps](https://github.com/t0yohei/Irodori-TTS-MLX/issues/30), and reproducible packaging from [#31 Package the project for reproducible runtime and benchmark environments](https://github.com/t0yohei/Irodori-TTS-MLX/issues/31).

## License notes

This repository's own source code and documentation are licensed under the [MIT License](LICENSE), unless a file explicitly states otherwise.

The MIT License does not cover upstream code, checkpoint files, DACVAE weights, tokenizer assets, reference audio, converted `.npz` archives, generated audio, or other artifacts that are not redistributed here. This repository does **not** redistribute checkpoints or derived model/audio artifacts for v0.1; users must obtain upstream artifacts themselves and follow the relevant upstream repository and model-card terms. See [docs/license_and_distribution.md](docs/license_and_distribution.md) for the full policy.
