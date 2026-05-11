# Checkpoint metadata and weight mapping notes

Issue: [#3 Inspect checkpoint metadata and state_dict layout](https://github.com/t0yohei/irodori-tts-mlx/issues/3)

This document records the observed Irodori-TTS v2 checkpoint structure before implementing MLX weight conversion.

The first conversion target is the base checkpoint:

- `Aratako/Irodori-TTS-500M-v2`

The VoiceDesign checkpoint is inspected for comparison because it is likely to share most RF-DiT and text-path weights while changing the conditioning path:

- `Aratako/Irodori-TTS-500M-v2-VoiceDesign`

Do not commit model weights, Hugging Face cache files, generated audio, or full checkpoint dumps to this repository.

## Inspection method

The observations below were collected from the public `model.safetensors` headers using HTTP range reads. This reads only the safetensors header bytes, not the full tensor payload.

This is enough to inspect:

- safetensors metadata
- tensor names
- tensor dtypes
- tensor shapes
- tensor byte offsets

It does **not** validate numerical values or runtime semantics. The upstream PyTorch implementation remains the source of truth for behavior.

## Checkpoint summary

| Checkpoint | File | Tensor count | Dtype | Tensor payload size | Notes |
| --- | --- | ---: | --- | ---: | --- |
| `Aratako/Irodori-TTS-500M-v2` | `model.safetensors` | 613 | `F32` | ~1.84 GiB | Base v2 checkpoint with text + speaker conditioning. |
| `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | `model.safetensors` | 636 | `F32` | ~1.90 GiB | VoiceDesign variant with text + caption conditioning. |

Both safetensors files include a `__metadata__.config_json` entry.

### Base config fields

Important fields from `Aratako/Irodori-TTS-500M-v2`:

| Field | Value |
| --- | ---: |
| `latent_dim` | 32 |
| `latent_patch_size` | 1 |
| `model_dim` | 1280 |
| `num_layers` | 12 |
| `num_heads` | 20 |
| `mlp_ratio` | 2.875 |
| `text_vocab_size` | 99574 |
| `text_tokenizer_repo` | `llm-jp/llm-jp-3-150m` |
| `text_add_bos` | `true` |
| `text_dim` | 512 |
| `text_layers` | 10 |
| `text_heads` | 8 |
| `text_mlp_ratio` | 2.6 |
| `speaker_dim` | 768 |
| `speaker_layers` | 8 |
| `speaker_heads` | 12 |
| `speaker_mlp_ratio` | 2.6 |
| `speaker_patch_size` | 1 |
| `timestep_embed_dim` | 512 |
| `adaln_rank` | 192 |
| `norm_eps` | `1e-5` |
| `max_text_len` | 256 |
| `fixed_target_latent_steps` | 750 |

### VoiceDesign config differences

The VoiceDesign checkpoint keeps the same core RF-DiT and text dimensions, and adds caption conditioning:

| Field | Value |
| --- | ---: |
| `use_caption_condition` | `true` |
| `caption_vocab_size` | 99574 |
| `caption_tokenizer_repo` | `llm-jp/llm-jp-3-150m` |
| `caption_add_bos` | `true` |
| `caption_dim` | 512 |
| `caption_layers` | 10 |
| `caption_heads` | 8 |
| `caption_mlp_ratio` | 2.6 |
| `max_caption_len` | 512 |

The VoiceDesign checkpoint metadata does not include the base checkpoint's speaker encoder fields.

## Top-level tensor groups

### Base checkpoint

| Group | Tensor count | Purpose |
| --- | ---: | --- |
| `blocks` | 384 | 12 RF-DiT blocks. |
| `text_encoder` | 121 | Text token embedding and 10 text encoder blocks. |
| `speaker_encoder` | 98 | Speaker/reference latent encoder and 8 speaker encoder blocks. |
| `cond_module` | 3 | Timestep/conditioning MLP. |
| `in_proj` | 2 | Input projection from DACVAE latent dimension to model dimension. |
| `out_proj` | 2 | Output projection from model dimension back to DACVAE latent dimension. |
| `text_norm` | 1 | Text encoder output norm. |
| `speaker_norm` | 1 | Speaker encoder output norm. |
| `out_norm` | 1 | Final model norm before output projection. |

### VoiceDesign checkpoint

| Group | Tensor count | Purpose |
| --- | ---: | --- |
| `blocks` | 384 | 12 RF-DiT blocks. |
| `text_encoder` | 121 | Text token embedding and 10 text encoder blocks. |
| `caption_encoder` | 121 | Caption token embedding and 10 caption encoder blocks. |
| `cond_module` | 3 | Timestep/conditioning MLP. |
| `in_proj` | 2 | Input projection from DACVAE latent dimension to model dimension. |
| `out_proj` | 2 | Output projection from model dimension back to DACVAE latent dimension. |
| `text_norm` | 1 | Text encoder output norm. |
| `caption_norm` | 1 | Caption encoder output norm. |
| `out_norm` | 1 | Final model norm before output projection. |

## Naming patterns

### Shared RF-DiT blocks

Both inspected checkpoints have 12 blocks named `blocks.0` through `blocks.11`.

Each block has these shared self-attention tensors:

```text
blocks.{i}.attention.gate.weight       [1280, 1280]
blocks.{i}.attention.k_norm.weight     [20, 64]
blocks.{i}.attention.q_norm.weight     [20, 64]
blocks.{i}.attention.wk.weight         [1280, 1280]
blocks.{i}.attention.wo.weight         [1280, 1280]
blocks.{i}.attention.wq.weight         [1280, 1280]
blocks.{i}.attention.wv.weight         [1280, 1280]
```

The base checkpoint also has speaker cross-attention projections in every RF-DiT block:

```text
blocks.{i}.attention.wk_speaker.weight [1280, 768]
blocks.{i}.attention.wv_speaker.weight [1280, 768]
```

The VoiceDesign checkpoint replaces these with caption projections:

```text
blocks.{i}.attention.wk_caption.weight [1280, 512]
blocks.{i}.attention.wv_caption.weight [1280, 512]
```

Both checkpoints share text cross-attention projections:

```text
blocks.{i}.attention.wk_text.weight    [1280, 512]
blocks.{i}.attention.wv_text.weight    [1280, 512]
```

Each block has an MLP with the observed hidden size `3680`:

```text
blocks.{i}.mlp.w1.weight               [3680, 1280]
blocks.{i}.mlp.w2.weight               [1280, 3680]
blocks.{i}.mlp.w3.weight               [3680, 1280]
```

Each block has low-rank AdaLN modulation tensors for attention and MLP:

```text
blocks.{i}.attention_adaln.{gate,scale,shift}_down.weight [192, 1280]
blocks.{i}.attention_adaln.{gate,scale,shift}_up.weight   [1280, 192]
blocks.{i}.attention_adaln.{gate,scale,shift}_up.bias     [1280]
blocks.{i}.mlp_adaln.{gate,scale,shift}_down.weight       [192, 1280]
blocks.{i}.mlp_adaln.{gate,scale,shift}_up.weight         [1280, 192]
blocks.{i}.mlp_adaln.{gate,scale,shift}_up.bias           [1280]
```

### Text encoder

Both checkpoints have the same text encoder pattern:

```text
text_encoder.text_embedding.weight                  [99574, 512]
text_encoder.blocks.{i}.attention.gate.weight       [512, 512]
text_encoder.blocks.{i}.attention.k_norm.weight     [8, 64]
text_encoder.blocks.{i}.attention.q_norm.weight     [8, 64]
text_encoder.blocks.{i}.attention.wk.weight         [512, 512]
text_encoder.blocks.{i}.attention.wo.weight         [512, 512]
text_encoder.blocks.{i}.attention.wq.weight         [512, 512]
text_encoder.blocks.{i}.attention.wv.weight         [512, 512]
text_encoder.blocks.{i}.attention_norm.weight       [512]
text_encoder.blocks.{i}.mlp.w1.weight               [1331, 512]
text_encoder.blocks.{i}.mlp.w2.weight               [512, 1331]
text_encoder.blocks.{i}.mlp.w3.weight               [1331, 512]
text_encoder.blocks.{i}.mlp_norm.weight             [512]
text_norm.weight                                    [512]
```

The text encoder has 10 blocks named `text_encoder.blocks.0` through `text_encoder.blocks.9`.

### Base speaker encoder

The base checkpoint has an 8-block speaker encoder:

```text
speaker_encoder.in_proj.weight                      [768, 32]
speaker_encoder.in_proj.bias                        [768]
speaker_encoder.blocks.{i}.attention.gate.weight    [768, 768]
speaker_encoder.blocks.{i}.attention.k_norm.weight  [12, 64]
speaker_encoder.blocks.{i}.attention.q_norm.weight  [12, 64]
speaker_encoder.blocks.{i}.attention.wk.weight      [768, 768]
speaker_encoder.blocks.{i}.attention.wo.weight      [768, 768]
speaker_encoder.blocks.{i}.attention.wq.weight      [768, 768]
speaker_encoder.blocks.{i}.attention.wv.weight      [768, 768]
speaker_encoder.blocks.{i}.attention_norm.weight    [768]
speaker_encoder.blocks.{i}.mlp.w1.weight            [1996, 768]
speaker_encoder.blocks.{i}.mlp.w2.weight            [768, 1996]
speaker_encoder.blocks.{i}.mlp.w3.weight            [1996, 768]
speaker_encoder.blocks.{i}.mlp_norm.weight          [768]
speaker_norm.weight                                 [768]
```

### VoiceDesign caption encoder

The VoiceDesign checkpoint has a caption encoder with the same shape pattern as the text encoder:

```text
caption_encoder.text_embedding.weight               [99574, 512]
caption_encoder.blocks.{i}.attention.*              same shapes as text_encoder.blocks.{i}.attention.*
caption_encoder.blocks.{i}.mlp.*                    same shapes as text_encoder.blocks.{i}.mlp.*
caption_norm.weight                                 [512]
```

The caption encoder has 10 blocks named `caption_encoder.blocks.0` through `caption_encoder.blocks.9`.

### Input, output, and conditioning tensors

Both checkpoints share these non-encoder projection tensors:

```text
in_proj.weight       [1280, 32]
in_proj.bias         [1280]
out_norm.weight      [1280]
out_proj.weight      [32, 1280]
out_proj.bias        [32]
cond_module.0.weight [1280, 512]
cond_module.2.weight [1280, 1280]
cond_module.4.weight [3840, 1280]
```

The observed `cond_module` tensors have weights but no biases in the safetensors header.

## Base vs VoiceDesign differences

The common subset includes the core RF-DiT self-attention, text projections, MLPs, AdaLN tensors, input/output projections, `text_encoder`, `text_norm`, `out_norm`, and `cond_module` names.

The main differences are:

- Base-only:
  - `speaker_encoder.*`
  - `speaker_norm.weight`
  - `blocks.{i}.attention.wk_speaker.weight`
  - `blocks.{i}.attention.wv_speaker.weight`
- VoiceDesign-only:
  - `caption_encoder.*`
  - `caption_norm.weight`
  - `blocks.{i}.attention.wk_caption.weight`
  - `blocks.{i}.attention.wv_caption.weight`
  - caption-related config fields

The raw key-set comparison reports 123 base-only tensor names and 146 VoiceDesign-only tensor names. The count difference is expected because the 10-block caption encoder has more tensors than the 8-block speaker encoder it replaces.

## Initial MLX mapping notes

The converter should initially preserve the checkpoint hierarchy closely, because the names already separate the major runtime components:

| PyTorch/safetensors prefix | Initial MLX target |
| --- | --- |
| `text_encoder.*` | MLX text encoder module. |
| `speaker_encoder.*` | Base-checkpoint reference latent encoder. |
| `caption_encoder.*` | VoiceDesign caption encoder; can be staged after base support. |
| `blocks.*` | RF-DiT block stack. |
| `cond_module.*` | Timestep/conditioning MLP. |
| `in_proj.*` | Latent input projection. |
| `out_norm.*`, `out_proj.*` | Final normalization and latent output projection. |
| `*_norm.weight` | RMSNorm or equivalent upstream norm weights; confirm in upstream source before implementation. |

Shape orientation should not be changed blindly. MLX layer conventions may differ from PyTorch, so issue #5 should decide whether to transpose linear weights during conversion or load them as-is and adapt the MLX module definitions.

## Unknown or ambiguous tensors

These items need explicit confirmation in follow-up work:

- Whether `gate.weight` in attention modules is a learned output gate, a linear projection, or another upstream-specific operation.
- Exact upstream norm type for `*_norm.weight`, `attention_norm.weight`, `mlp_norm.weight`, `q_norm.weight`, and `k_norm.weight`.
- Exact `cond_module` activation/function order, since safetensors records only the three weight matrices.
- Whether any linear weights require transposition for MLX, or whether MLX modules should mirror PyTorch weight orientation.
- How upstream maps DACVAE latents into `speaker_encoder.in_proj` and `in_proj`, including expected tensor layout/order.
- Whether VoiceDesign should be supported in the first converter or intentionally staged after base checkpoint parity.
- The DACVAE codec checkpoint `Aratako/Semantic-DACVAE-Japanese-32dim` uses `weights.pth`, not `model.safetensors`; it needs a separate inspection path if DACVAE conversion is ever in scope.

## Follow-up issues

- Issue #4 should turn this ad-hoc header inspection into a reusable checkpoint inspection script.
- Issue #5 should define the PyTorch-to-MLX mapping rules, including transpose policy and unsupported tensor handling.
- Issue #6 should implement the first converter using those mapping rules.
