# Hosted/local DACVAE codec artifact layout and bridge fallback policy

Issue: [#116](https://github.com/t0yohei/Irodori-TTS-MLX/issues/116)
Parent epic: [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

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

The converter writes the scalar runtime constants, `metadata_json`, and every
decoder-side tensor under `dacvae_decoder/<state-dict-key>`. The required groups
are:

- `quantizer.out_proj.*`, including a 32-channel latent decode projection;
- `decoder.*`, including the mono waveform projection;
- provenance fields for source repo, source revision, source file, converter
  commit, `dacvae` revision, watermark-bypass policy, and license-review status.

This real decoder artifact is a deterministic manifest and tensor transport
contract. The current public runtime can inspect it and report that real
Semantic-DACVAE decoder tensors are present, but it cannot yet execute the full
MLX convolutional decoder stack. Until that executor lands, keep
`persistent`/`subprocess` PyTorch bridge modes as the waveform decode fallback.
Do not use `scripts/check_dacvae_decode_parity.py` for artifacts of kind
`real_semantic_dacvae_decoder`: those artifacts intentionally contain
`dacvae_decoder/<state-dict-key>` tensors instead of the fixture
`decode_basis`/`decode_bias` arrays, so `MLXDACVAEBridge` rejects them until the
real MLX decoder executor exists.

After conversion, validate the transport contract by checking that the converter
report and runtime capability inspection identify the artifact as a blocked real
decoder artifact:

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
`has_real_dacvae_decode=true`, `has_mlx_decode=false`, and a capability message
that the MLX DACVAE convolutional decoder executor is not implemented yet.
Decode parity comparison becomes the required validation path only after the
real MLX executor can consume those decoder tensors.

## Hosted companion metadata

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
