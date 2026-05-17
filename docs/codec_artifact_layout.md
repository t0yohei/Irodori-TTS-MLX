# Hosted/local DACVAE codec artifact layout and bridge fallback policy

Issue: [#116](https://github.com/t0yohei/Irodori-TTS-MLX/issues/116),
[#158](https://github.com/t0yohei/Irodori-TTS-MLX/issues/158)
Parent epic: [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123),
[#160](https://github.com/t0yohei/Irodori-TTS-MLX/issues/160)

This page defines the v0.2 artifact story for DACVAE codec weights. It is a
layout and runtime-policy contract only; acoustic parity, real codec conversion,
and full encode/decode validation remain owned by the dedicated DACVAE issues.

## Relationship to hosted RF-DiT weights

Hosted RF-DiT repositories described in
[hosted_weights_layout.md](hosted_weights_layout.md) continue to contain model
weights, model config, tokenizer metadata, conversion provenance, and checksums.
They do not bundle Semantic-DACVAE weights by default.

Codec artifacts are separate because they have a different upstream source,
different conversion tooling, and different redistribution review. A hosted
RF-DiT manifest may point to an optional companion codec artifact, but the
runtime must still work when that pointer is absent by using the documented
PyTorch bridge fallback.

## Local codec artifact file

The local runtime contract is one `.npz` file:

```text
dacvae-codec.npz
|-- sample_rate              # scalar int, expected 48000 for the public codec
|-- hop_length               # scalar int, expected 512 for the public codec
|-- latent_dim               # scalar int, expected 32 for Irodori families
|-- decode_basis             # required by current MLX decode fixture path
|-- decode_bias              # required by current MLX decode fixture path
|-- encode_basis             # required only for experimental full mlx mode
|-- encode_bias              # required only for experimental full mlx mode
|-- semantic_encoder_manifest_json  # required by a future real Semantic-DACVAE encoder artifact
`-- metadata_json            # optional scalar JSON string with provenance
```

The current checked-in MLX artifact format is intentionally a small fixture
contract. It proves runtime selection, local artifact loading, encode/decode
routing, and metadata reporting without claiming Semantic-DACVAE acoustic
parity. A real converted codec artifact must replace the fixture tensors with
the full DACVAE encoder, quantizer projections, decoder, and watermark-bypass
metadata described in [dacvae_architecture.md](dacvae_architecture.md).

`metadata_json.artifact_kind` distinguishes this temporary runtime fixture from
a future real conversion. The current writer/fixtures use
`"linear-fixture"`. A real Semantic-DACVAE artifact must use
`"semantic-dacvae"` and include `semantic_encoder_manifest_json` alongside the
complete encoder, `quantizer.in_proj`, `quantizer.out_proj`, decoder, and
watermark-bypass tensors. `irodori_mlx.runtime.inspect_mlx_codec_artifact()`
reports `artifact_kind`, `is_semantic_dacvae`, and semantic encoder evidence
counts so parity scripts and hosted artifact checks can reject fixture artifacts
when real codec evidence is required.

## Semantic-DACVAE encoder conversion status

Issue [#154](https://github.com/t0yohei/Irodori-TTS-MLX/issues/154) adds the
explicit conversion entrypoint for the real encoder path:

```bash
irodori-tts-convert-dacvae-codec /path/to/weights.pth /path/to/dacvae-codec.npz \
  --inspect-only \
  --report-json /tmp/dacvae-codec-conversion-blocker.json \
  --json
```

The command loads and inspects the local `weights.pth` state_dict, verifies
whether the expected encoder and `quantizer.in_proj` logical groups are present,
and writes a machine-readable blocker report. It intentionally does not write a
`dacvae-codec.npz` while the runtime lacks MLX implementations of the real
DACVAE conv/residual/VAEBottleneck modules. Running without `--inspect-only`
returns a blocked conversion status instead of emitting a misleading artifact.

## Real Semantic-DACVAE decoder artifact

Issue [#151](https://github.com/t0yohei/Irodori-TTS-MLX/issues/151) adds the
first real-weight decoder conversion contract for the public
`Aratako/Semantic-DACVAE-Japanese-32dim` family. The source artifact is the
upstream PyTorch `weights.pth`; it must be obtained locally and must not be
committed here.

Use the dedicated converter when the real weights and exact dependency
revisions are available:

```bash
python scripts/convert_dacvae_decoder.py \
  /path/to/Semantic-DACVAE-Japanese-32dim/weights.pth \
  /tmp/irodori-dacvae-codec/dacvae-codec.npz \
  --source-revision <hf-commit> \
  --dacvae-revision <dacvae-commit> \
  --license-review-status pending \
  --json
```

The converter writes the scalar runtime constants, `metadata_json`, every
decoder-side source tensor under `dacvae_decoder/<state-dict-key>`, and the
runtime-ready MLX decoder tensors under `dacvae_decoder_exec/<module-key>`. The
required source groups are:

- `quantizer.out_proj.*`, including a 32-channel latent decode projection;
- `decoder.*`, including the mono waveform projection;
- provenance fields for source repo, source revision, source file, converter
  commit, `dacvae` revision, watermark-bypass policy, and license-review status.

The executable keys mirror `irodori_mlx.dacvae.SemanticDACVAEDecoder` names, for
example `quantizer_out_proj.weight_g`, `blocks.0.main_upsample.1.weight_v`, and
`conv_out.bias`. PyTorch Conv1d and ConvTranspose1d kernels are transposed into
MLX channel-last layout, and PyTorch parametrized weight norm tensors are mapped
from `parametrizations.weight.original0/original1` to `weight_g/weight_v`. The
final non-watermarked output activation/projection is sourced from
`decoder.wm_model.encoder_block.pre`, matching the upstream Irodori watermark
bypass path where `decoder.alpha == 0`.

This real decoder artifact is deterministic and executable for decode-only MLX
runtime modes. Runtime capability inspection reports
`has_real_dacvae_decode=true`, `has_executable_mlx_decode=true`, and
`has_mlx_decode=true` when the executable tensor layout is present. Acoustic
parity is still a separate validation gate; keep `persistent`/`subprocess`
PyTorch bridge modes available as a fallback until parity reports have been run
against local real weights.

After conversion, validate the transport contract by checking that the converter
report and runtime capability inspection identify the artifact as an executable
real decoder artifact:

```bash
python scripts/convert_dacvae_decoder.py \
  /path/to/Semantic-DACVAE-Japanese-32dim/weights.pth \
  /tmp/irodori-dacvae-codec/dacvae-codec.npz \
  --source-revision <hf-commit> \
  --dacvae-revision <dacvae-commit> \
  --license-review-status pending \
  --json
```

Expected evidence is `artifact_kind=real_semantic_dacvae_decoder`,
`has_real_dacvae_decode=true`, `has_executable_mlx_decode=true`,
`has_mlx_decode=true`, `runtime_status.mlx_decoder_execution=available_unvalidated`,
and a capability message that acoustic parity remains gated by local validation.
Decode parity comparison is the next validation step before publishing converted
weights as a parity-backed hosted artifact.

## Hosted companion metadata

The approved public hosted path for converted DACVAE codec artifacts is a
dedicated Hugging Face model repository, not an RF-DiT weights repository
subdirectory. The recommended name is:

```text
<t0yohei-or-approved-org>/Irodori-DACVAE-Codec-MLX
```

Equivalent local staging directories and public hosted repos use this layout:

```text
repo-or-local-dir/
|-- README.md
|-- LICENSE.md
|-- irodori_dacvae_codec_manifest.json
|-- dacvae-codec.npz
|-- codec_metadata.json
|-- checksums.sha256
```

`irodori_dacvae_codec_manifest.json` is the source of truth. It must contain:

```json
{
  "schema_version": 1,
  "artifact_format": "irodori-tts-mlx-dacvae-codec",
  "artifact_format_version": "0.2",
  "files": {
    "codec": "dacvae-codec.npz",
    "metadata": "codec_metadata.json",
    "checksums": "checksums.sha256"
  },
  "codec": {
    "source_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
    "source_revision": "<hf-commit>",
    "source_file": "weights.pth",
    "artifact_kind": "semantic-dacvae",
    "sample_rate": 48000,
    "hop_length": 512,
    "latent_dim": 32
  },
  "runtime": {
    "minimum_irodori_tts_mlx_version": "0.2.0",
    "supports_mlx_decode": true,
    "supports_mlx_encode": false,
    "requires_pytorch_fallback": true
  },
  "license_review": {
    "status": "approved",
    "review_reference": "<public-review-url>"
  }
}
```

`codec_metadata.json` must repeat the artifact format/version and record
provenance plus validation evidence:

- upstream codec repo id, exact source revision, and source file;
- converter repository and converter commit/tag;
- `dacvae` package revision used for tensor interpretation;
- license review status and review reference;
- decode and encode parity report locations, or an explicit reason that a path
  is not yet executable by the current MLX runtime;
- checksum validation for `dacvae-codec.npz`, `codec_metadata.json`, and the
  manifest.

Public hosted repositories require `license_review.status: "approved"` before
`irodori-tts-generate --codec-artifact-repo` accepts them. Local staging
directories may use `pending` while conversion or parity work is in progress.

Resolve a local staged codec artifact with:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-runtime-mode mlx-decode \
  --codec-artifact-dir /models/Irodori-DACVAE-Codec-MLX \
  --text "こんにちは。" \
  --no-reference \
  --output /tmp/irodori-v3-mlx-codec.wav
```

Resolve a public hosted codec artifact with:

```bash
irodori-tts-generate \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v3 \
  --codec-runtime-mode mlx-decode \
  --codec-artifact-repo t0yohei/Irodori-DACVAE-Codec-MLX \
  --codec-artifact-revision <approved-hf-commit> \
  --text "こんにちは。" \
  --no-reference \
  --output /tmp/irodori-v3-hosted-codec.wav
```

The CLI downloads only manifest-declared files from the codec repo, validates
checksums and approved license status, then passes the resolved
`dacvae-codec.npz` path into the existing `--codec-path` runtime. Use
`--codec-path` directly for private one-file experiments or as the fallback
when hosted resolution is unavailable.


If a hosted RF-DiT repo references a companion codec, add the pointer under the
manifest `codec` key instead of copying codec weights into `weights.npz`:

```json
{
  "codec": {
    "source_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
    "source_revision": "<hf-revision-or-commit>",
    "source_file": "weights.pth",
    "artifact_kind": "separate-local-or-hosted-dacvae-codec",
    "artifact_format": "irodori-tts-mlx-dacvae-codec",
    "artifact_format_version": "0.2",
    "sample_rate": 48000,
    "hop_length": 512,
    "latent_dim": 32,
    "runtime_modes": ["mlx-decode", "mlx"],
    "provenance": {
      "converter_repository": "https://github.com/t0yohei/Irodori-TTS-MLX",
      "converter_version": "git:<commit-sha-or-tag>",
      "dacvae_package_revision": "<dacvae-revision>",
      "license_review": "pending|approved|rejected"
    }
  }
}
```

Public hosted codec artifacts require approved license review for the upstream
codec weights and the converted derivative. Local/private codec artifacts may
use pending provenance for development, but they must not be published or used
as a supported hosted model until review is approved.

## Runtime capability checks

`irodori_mlx.runtime.describe_codec_capabilities()` reports the selected codec
mode, whether an MLX codec artifact is required, whether local MLX decode/encode
is available, and whether PyTorch encode/decode fallback is still required. It
inspects a local `.npz` path without importing PyTorch.

The CLI boundary JSON returned by `--json`, `--metadata-json`, or
`--print-boundaries` includes the same capability report under
`boundaries.codec.capabilities`.

## Bridge fallback policy by mode

| `--codec-runtime-mode` | Decode backend | Reference encode backend | Codec artifact required | Main use |
| --- | --- | --- | --- | --- |
| `persistent` | PyTorch bridge | PyTorch bridge | No | Default production-like local generation. |
| `subprocess` | PyTorch bridge in helper process | PyTorch bridge in helper process | No | Memory/lifecycle experiments. |
| `mlx-decode` | Local MLX codec artifact | PyTorch bridge when reference audio is used | Yes | Decode-port validation and no-reference generation smoke tests. |
| `mlx-decode-subprocess` | Local MLX codec artifact | PyTorch helper process when reference audio is used | Yes | Decode-port validation with isolated fallback encode. |
| `mlx` | Local MLX codec artifact | Local MLX codec artifact | Yes | Full local codec artifact experiments; requires encode tensors. |

When an MLX codec artifact is missing or decode-only, the error message should
tell users which artifact is required and how to fall back: use
`--codec-runtime-mode persistent` or `subprocess` for the upstream PyTorch
bridge, use `mlx-decode` for decode-only artifacts, or use `--no-reference`
when the checkpoint family and generation request do not need reference encode.

## Checkpoint-family UX

- `base_v2`: speaker/reference conditioning is normally enabled. Reference-audio
  requests need codec encode. `mlx-decode` can avoid PyTorch decode, but
  reference encode still uses the PyTorch bridge unless the artifact supports
  full `mlx` encode.
- `voicedesign`: caption conditioning disables the speaker/reference branch for
  the supported family. `mlx-decode` can exercise MLX decode without importing
  PyTorch when the request uses `--no-reference`.
- `v3`: duration prediction is independent from the codec. No-reference v3 smoke
  runs can use `mlx-decode`; reference-audio v3 runs follow the same encode
  fallback rule as base v2.

Unsupported or unaudited checkpoint families remain local-conversion-only and do
not change the codec fallback contract.

## Provenance requirements

Every real codec artifact, local or hosted, must record:

- upstream codec repo id, source file, and exact revision;
- `dacvae` package revision and converter commit/tag;
- runtime constants (`sample_rate`, `hop_length`, `latent_dim`);
- `artifact_kind: "semantic-dacvae"` and `semantic_encoder_manifest_json`;
- whether encode, decode, or both are present;
- parity evidence location for decode and encode when available;
- license-review status and review reference;
- a statement that the artifact is a converted derivative, not the original
  upstream `weights.pth`.

Do not commit codec weights, converted codec artifacts, reference audio,
generated WAVs, or Hugging Face cache snapshots to this repository.
