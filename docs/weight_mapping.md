# Checkpoint metadata and weight mapping notes

Issue: [#3 Inspect checkpoint metadata and state_dict layout](https://github.com/t0yohei/Irodori-TTS-MLX/issues/3)

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

## Deterministic PyTorch-to-MLX mapping

Issue [#5](https://github.com/t0yohei/Irodori-TTS-MLX/issues/5) fixes the first converter contract for the base v2 checkpoint.

### Supported checkpoint family

The first converter should support exactly the base checkpoint layout from `Aratako/Irodori-TTS-500M-v2`:

- expected tensor count: `613`
- expected dtype for all inspected tensors: `F32`
- expected conditioning path: text + speaker/reference-latent conditioning

The VoiceDesign checkpoint should be rejected by the first converter with a clear unsupported-checkpoint message, because it replaces the speaker path with caption tensors. Caption support can be added later after base checkpoint parity is proven.

### MLX naming convention

Use hierarchy-preserving MLX parameter names. For supported base tensors, the PyTorch/safetensors key is also the MLX destination key.

This deliberately keeps converter logic simple and makes missing/unexpected-key diagnostics line up with the source checkpoint:

```text
text_encoder.blocks.0.attention.wq.weight
  -> text_encoder.blocks.0.attention.wq.weight
blocks.11.mlp_adaln.gate_up.bias
  -> blocks.11.mlp_adaln.gate_up.bias
out_proj.bias
  -> out_proj.bias
```

If a future MLX module implementation intentionally renames attributes, that change should update this mapping table and the converter at the same time. Do not add silent aliases in the converter.

### Tensor transform policy

The first converter should not transpose or reshape checkpoint tensors. It should validate the source shape and write the same array under the MLX key.

| Tensor kind | Examples | Transform | Reason |
| --- | --- | --- | --- |
| Linear weights | `*.wq.weight`, `*.mlp.w1.weight`, `in_proj.weight`, `cond_module.0.weight` | Copy as-is | Upstream PyTorch `nn.Linear` stores `[out_features, in_features]`; MLX modules should mirror that parameter storage and perform the runtime matmul accordingly. |
| Linear biases | `in_proj.bias`, `out_proj.bias`, `*_up.bias` | Copy as-is | Bias vectors are already one-dimensional destination features. |
| Embeddings | `text_encoder.text_embedding.weight` | Copy as-is | Shape is `[vocab_size, dim]`. |
| RMSNorm weights | `*_norm.weight`, `q_norm.weight`, `k_norm.weight` | Copy as-is | Upstream uses RMSNorm. `q_norm`/`k_norm` intentionally store `[heads, head_dim]`. |
| SwiGLU weights | `mlp.w1.weight`, `mlp.w2.weight`, `mlp.w3.weight` | Copy as-is | Runtime implements `w2(silu(w1(x)) * w3(x))`; no packing is needed for the first converter. |
| Low-rank AdaLN weights | `*_adaln.*_{down,up}.weight` | Copy as-is | Upstream is two linear projections per shift/scale/gate branch. |

Runtime reshapes are model responsibilities, not converter responsibilities:

- attention projections reshape q/k/v from `[batch, seq, dim]` to `[batch, seq, heads, head_dim]` after projection;
- RF-DiT joint attention applies half-RoPE to latent self-attention q/k only;
- text/speaker encoder self-attention applies full RoPE to q/k;
- speaker reference latents are patched before `speaker_encoder.in_proj` when `speaker_patch_size > 1`.

### Upstream module semantics confirmed

The upstream PyTorch model defines these checkpoint-backed modules:

| Checkpoint tensors | Upstream module behavior | Converter implication |
| --- | --- | --- |
| `*.attention.gate.weight` | `nn.Linear(..., bias=False)` followed by `sigmoid(gate)` and elementwise attention-output gating. | Treat as an ordinary linear weight, copied as-is. |
| `*_norm.weight`, `attention_norm.weight`, `mlp_norm.weight`, `q_norm.weight`, `k_norm.weight` | Custom `RMSNorm`; q/k norms use `RMSNorm((heads, head_dim))`. | Copy norm weights as-is and require exact shapes. |
| `cond_module.0.weight`, `cond_module.2.weight`, `cond_module.4.weight` | `Linear(timestep_embed_dim, model_dim, bias=False)`, `SiLU`, `Linear(model_dim, model_dim, bias=False)`, `SiLU`, `Linear(model_dim, model_dim * 3, bias=False)`. | Expect exactly these three weight tensors and no cond-module biases. |
| `speaker_encoder.in_proj.*` | Linear projection from patched DACVAE reference latents to `speaker_dim`, followed by division by `6.0`. | Copy weights/bias as-is; the `/ 6.0` scale belongs in model forward code, not conversion. |
| `in_proj.*`, `out_proj.*` | Linear projections between patched DACVAE latents and model dimension. | Copy weights/bias as-is; latent patching/unpatching belongs in model code. |

### Base checkpoint mapping summary

The following pattern table covers all `613` tensors in `Aratako/Irodori-TTS-500M-v2`. `{i}` ranges are inclusive and every expanded key maps to the same destination key.

| PyTorch/safetensors key pattern | Range/count | MLX destination | Transform |
| --- | ---: | --- | --- |
| `blocks.{i}.attention.{gate,wk,wk_speaker,wk_text,wo,wq,wv,wv_speaker,wv_text}.weight` | `i=0..11`, 108 tensors | same key | copy |
| `blocks.{i}.attention.{k_norm,q_norm}.weight` | `i=0..11`, 24 tensors | same key | copy |
| `blocks.{i}.mlp.{w1,w2,w3}.weight` | `i=0..11`, 36 tensors | same key | copy |
| `blocks.{i}.{attention_adaln,mlp_adaln}.{gate,scale,shift}_down.weight` | `i=0..11`, 72 tensors | same key | copy |
| `blocks.{i}.{attention_adaln,mlp_adaln}.{gate,scale,shift}_up.weight` | `i=0..11`, 72 tensors | same key | copy |
| `blocks.{i}.{attention_adaln,mlp_adaln}.{gate,scale,shift}_up.bias` | `i=0..11`, 72 tensors | same key | copy |
| `text_encoder.text_embedding.weight` | 1 tensor | same key | copy |
| `text_encoder.blocks.{i}.attention.{gate,wk,wo,wq,wv}.weight` | `i=0..9`, 50 tensors | same key | copy |
| `text_encoder.blocks.{i}.attention.{k_norm,q_norm}.weight` | `i=0..9`, 20 tensors | same key | copy |
| `text_encoder.blocks.{i}.{attention_norm,mlp_norm}.weight` | `i=0..9`, 20 tensors | same key | copy |
| `text_encoder.blocks.{i}.mlp.{w1,w2,w3}.weight` | `i=0..9`, 30 tensors | same key | copy |
| `text_norm.weight` | 1 tensor | same key | copy |
| `speaker_encoder.in_proj.{weight,bias}` | 2 tensors | same key | copy |
| `speaker_encoder.blocks.{i}.attention.{gate,wk,wo,wq,wv}.weight` | `i=0..7`, 40 tensors | same key | copy |
| `speaker_encoder.blocks.{i}.attention.{k_norm,q_norm}.weight` | `i=0..7`, 16 tensors | same key | copy |
| `speaker_encoder.blocks.{i}.{attention_norm,mlp_norm}.weight` | `i=0..7`, 16 tensors | same key | copy |
| `speaker_encoder.blocks.{i}.mlp.{w1,w2,w3}.weight` | `i=0..7`, 24 tensors | same key | copy |
| `speaker_norm.weight` | 1 tensor | same key | copy |
| `cond_module.{0,2,4}.weight` | 3 tensors | same key | copy |
| `in_proj.{weight,bias}` | 2 tensors | same key | copy |
| `out_norm.weight` | 1 tensor | same key | copy |
| `out_proj.{weight,bias}` | 2 tensors | same key | copy |

### Converter validation requirements

The issue #6 converter should fail closed:

1. Inspect the checkpoint header and parse `metadata.config_json` before writing output.
2. Confirm base-checkpoint identity from config:
   - `use_speaker_condition` is true or speaker fields are present;
   - `use_caption_condition` is absent/false;
   - `latent_dim=32`, `model_dim=1280`, `num_layers=12`, `text_layers=10`, `speaker_layers=8`.
3. Expand the base mapping table and require the checkpoint key set to match exactly.
4. Report missing keys, unexpected keys, and shape mismatches separately.
5. Require all supported tensors to be numeric arrays with dtype `F32` initially. Optional dtype conversion can be a later explicit flag.
6. Reject VoiceDesign-only keys such as `caption_encoder.*`, `caption_norm.weight`, `blocks.{i}.attention.wk_caption.weight`, and `blocks.{i}.attention.wv_caption.weight` until caption support is implemented.
7. Write no partial output on validation failure.

## Remaining follow-up notes

These items remain outside issue #5 and should be handled by implementation/parity work:

- The DACVAE codec checkpoint `Aratako/Semantic-DACVAE-Japanese-32dim` uses `weights.pth`, not `model.safetensors`; it needs a separate inspection path if DACVAE conversion is ever in scope.
- Issue #6 should implement the first converter using the mapping rules above.
- Later MLX model parity work should validate numerical behavior for RoPE, RMSNorm, LowRankAdaLN, masking, and speaker latent patching.
