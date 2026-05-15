# Hosted MLX weights repository layout

Issue: [#79 Define the hosted MLX weights repository layout](https://github.com/t0yohei/irodori-tts-mlx/issues/79)
Parent: [#78 v0.2: Support pre-converted MLX weights from Hugging Face](https://github.com/t0yohei/irodori-tts-mlx/issues/78)

This page is the v0.2 contract for a repository that hosts **pre-converted MLX RF-DiT weights** for `irodori-tts-mlx`.
It defines the file names, metadata, provenance notes, and path conventions that follow-up loader work can rely on.
It does **not** approve publishing any weights by itself; every hosted artifact still needs a separate upstream-license and redistribution review before upload.

## Goals

- Let the same runtime loader resolve either a local directory or a Hugging Face model repository.
- Keep local conversion as a supported fallback whenever hosted weights are missing, incompatible, or not licensed for redistribution.
- Keep converted weights, model config, tokenizer/conditioning metadata, and conversion provenance together.
- Make family differences explicit enough for validation before `generate_wav.py` starts inference.

## Repository naming

Recommended Hugging Face repository name pattern:

```text
<t0yohei-or-approved-org>/irodori-tts-mlx-<family>-<upstream-version>
```

Examples, subject to license approval before publication:

- `t0yohei/irodori-tts-mlx-v3-500m`
- `t0yohei/irodori-tts-mlx-voicedesign-v2-500m`

The repository name identifies the converted checkpoint family, but loaders must treat `irodori_mlx_manifest.json` as the source of truth.
Do not infer compatibility from the repo name alone.

## Top-level layout

A hosted weights repo and an equivalent local directory use the same layout:

```text
repo-or-local-dir/
├── README.md
├── LICENSE.md                  # terms for metadata in this repo, not a blanket upstream-weight license
├── irodori_mlx_manifest.json   # required loader/validation manifest
├── model_config.json           # required ModelConfig-compatible runtime config
├── tokenizer_config.json       # required tokenizer/conditioning metadata for this family
├── conversion_metadata.json    # required conversion provenance and tool versions
├── weights.npz                 # required converted MLX RF-DiT weights
└── checksums.sha256            # required hashes for all required files including weights.npz
```

Optional files may be added under clearly named directories such as `docs/`, `examples/`, or `validation/`, but required loader inputs must stay at the top level for simple `hf_hub_download` / `snapshot_download` resolution.

## Required file contracts

### `weights.npz`

- MLX-friendly NumPy archive produced by `scripts/convert_weights.py` or the packaged `irodori-tts-convert` command.
- Contains only the converted RF-DiT / text / conditioning weights owned by this MLX runtime path.
- Must not include upstream safetensors files, Hugging Face cache snapshots, reference audio, generated audio, DACVAE weights, or unrelated artifacts.

### `model_config.json`

A JSON object containing fields accepted by `irodori_mlx.config.ModelConfig` for the converted family.
The loader should reject unknown required architecture changes instead of silently falling back to defaults.

Minimum expectations:

- v3 sets `use_duration_predictor: true` and includes the duration-predictor dimensions needed by the runtime.
- base v2 and VoiceDesign v2 keep `use_duration_predictor: false` unless a future validated family says otherwise.
- VoiceDesign sets the caption-conditioning fields required by `--caption` generation.

### `tokenizer_config.json`

A JSON object describing tokenizer and conditioning-token semantics needed to reproduce the upstream-compatible preprocessing path.
It should include, at minimum:

```json
{
  "schema_version": 1,
  "text_tokenizer": {
    "source": "upstream",
    "normalization_contract": "docs/text_preprocessing.md",
    "padding": "right",
    "truncation": "family-defined"
  },
  "caption_tokenizer": null
}
```

For VoiceDesign, `caption_tokenizer` must be an object instead of `null` and must name the caption-tokenizer source/contract used for the converted family.
Tokenizer metadata can point at an upstream tokenizer source when redistribution is not approved; do not mirror tokenizer assets into the hosted repo unless their terms allow it.

### `conversion_metadata.json`

A JSON object with conversion provenance. Required fields:

```json
{
  "schema_version": 1,
  "converter": {
    "repository": "https://github.com/t0yohei/irodori-tts-mlx",
    "version": "git:<commit-sha-or-tag>",
    "command": "irodori-tts-convert /path/to/model.safetensors weights.npz"
  },
  "upstream": {
    "checkpoint_repo": "Aratako/Irodori-TTS-500M-v3",
    "checkpoint_revision": "<hf-revision-or-commit>",
    "source_file": "model.safetensors",
    "model_card": "https://huggingface.co/Aratako/Irodori-TTS-500M-v3"
  },
  "detected_family": "v3",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "license_review": {
    "status": "pending|approved|rejected",
    "notes": "No publication unless approved."
  }
}
```

Use `license_review.status: "pending"` for local/private staging only. Public hosted repos must not be created from pending or rejected metadata.

### `irodori_mlx_manifest.json`

The manifest is the loader source of truth. Required shape:

```json
{
  "schema_version": 1,
  "format": "irodori-tts-mlx-weights",
  "format_version": "0.2",
  "family": "v3",
  "upstream_checkpoint": "Aratako/Irodori-TTS-500M-v3",
  "files": {
    "weights": "weights.npz",
    "model_config": "model_config.json",
    "tokenizer_config": "tokenizer_config.json",
    "conversion_metadata": "conversion_metadata.json",
    "checksums": "checksums.sha256"
  },
  "runtime": {
    "minimum_irodori_tts_mlx_version": "0.2.0",
    "requires_upstream_dacvae_bridge": true,
    "requires_reference_audio": false,
    "supports_no_reference": true,
    "supports_caption": false,
    "supports_predicted_duration": true
  },
  "license_review": {
    "status": "approved",
    "review_reference": "<issue-or-document-url>"
  }
}
```

The manifest deliberately separates loader behavior from the README prose so automated validation can reject incompatible layouts.

### `checksums.sha256`

A plain SHA-256 checksum file covering every required top-level file, including `weights.npz`.
Consumers should verify checksums after download when practical. At minimum, repository validation should ensure the checksum file names every required file.

### `README.md` and `LICENSE.md`

The hosted repo README must state:

- the upstream checkpoint repo and revision used for conversion;
- that this is a converted MLX artifact, not the upstream checkpoint;
- the converter repo/version used;
- the supported runtime version and generation family (`v3`, `voicedesign`, etc.);
- the DACVAE/upstream-code boundary still required by the runtime;
- a link to the license review or publishing decision for the converted weights;
- a fallback local-conversion recipe using the original upstream checkpoint.

`LICENSE.md` must not imply that this repository's MIT license grants rights to upstream checkpoints, tokenizer assets, DACVAE weights, reference audio, or generated audio. Keep code/docs licensing separate from model-artifact licensing.

## Local path versus hosted repository

The loader should accept both forms and normalize them to the same resolved directory contract:

```bash
# Local converted layout, useful for private conversion or redistribution-not-approved families.
irodori-tts-generate \
  --weights-dir /models/irodori-tts-mlx-v3-500m \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav

# Hosted layout after license approval and publication.
irodori-tts-generate \
  --weights-repo t0yohei/irodori-tts-mlx-v3-500m \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav
```

Follow-up implementation may choose the final CLI flag names, but the contract is:

1. resolve a local directory or remote snapshot;
2. read `irodori_mlx_manifest.json`;
3. validate required files, schema version, family, runtime flags, and license-review status;
4. load `model_config.json`, `tokenizer_config.json`, and `weights.npz` through the same internal runtime path.

The existing direct `.npz` path remains supported as a local fallback:

```bash
irodori-tts-generate \
  --weights /path/to/irodori-v3.npz \
  --model-config-json /path/to/v3-model-config.json \
  --text "こんにちは。今日は良い天気です。" \
  --no-reference \
  --output /tmp/irodori-v3.wav
```

## Family-specific notes

| Family | Hosted eligibility for v0.2 | Required differences |
| --- | --- | --- |
| v3 (`Aratako/Irodori-TTS-500M-v3`) | Primary candidate after license approval | `supports_predicted_duration: true`; `use_duration_predictor: true`; no-reference smoke path should be documented. |
| VoiceDesign v2 (`Aratako/Irodori-TTS-500M-v2-VoiceDesign`) | Candidate only after redistribution and tokenizer/metadata review | `supports_caption: true`; caption-tokenizer metadata required; README must document `--caption` usage. |
| base v2 (`Aratako/Irodori-TTS-500M-v2`) | Lower priority because v0.1 generation remains experimental | Usually `requires_reference_audio: true`; no predicted duration; publish only after support status and license review are clear. |
| Future families | Not eligible by default | Add converter/runtime validation and a new manifest family contract before hosting. |

## Versioning policy

- Increment `schema_version` only when JSON structure changes incompatibly.
- Keep `format_version` aligned with the first project minor version that can load the layout; this contract starts at `0.2`.
- Prefer immutable Hugging Face revisions for published weights. If a conversion is regenerated, publish a new commit/revision and update `conversion_metadata.json`.
- Do not replace `weights.npz` in-place without updating checksums, conversion metadata, README provenance, and validation evidence.

## Validation expectations for follow-up issues

Before a hosted repository is published or consumed by default, validation should confirm:

- all required top-level files exist;
- `irodori_mlx_manifest.json` names every required file;
- `model_config.json` is accepted by `ModelConfig` for the family;
- tokenizer metadata matches the documented text/caption preprocessing contracts;
- checksums cover all required files;
- `license_review.status` is `approved` for any public hosted repo;
- local directory and hosted snapshot resolution feed the same runtime loading function;
- local conversion from the upstream checkpoint remains documented as a fallback.

## Redistribution boundary

This layout is only metadata and contract design. It does not upload, publish, or bless redistribution of any converted weights.
Converted `.npz` files are derived artifacts from upstream checkpoints, and the runtime still depends on upstream code/DACVAE boundaries. Follow [docs/license_and_distribution.md](license_and_distribution.md) before publishing anything outside a private local directory.
