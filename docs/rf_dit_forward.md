# RF-DiT forward path notes

This document records the first MLX RF-DiT forward implementation added for issue #9.

## Implemented MLX path

- `JointAttention` projects latent self-attention Q/K/V and conditioning K/V for text plus either speaker/reference or caption contexts.
- Latent self Q/K use upstream-compatible half-RoPE: the first half of the head axis receives RoPE and the second half passes through unchanged.
- `project_context_kv()` exposes the static text/speaker/caption K/V path so sampling can reuse per-layer conditioning projections.
- `DiffusionBlock` applies timestep-conditioned `LowRankAdaLN` before attention and MLP residuals.
- `TextToLatentRFDiT` wires condition encoders, timestep MLP, latent input/output projections, RF-DiT blocks, output RMSNorm, and latent patch/unpatch handling.

## Weight-loading shape

The module names intentionally mirror upstream checkpoint keys:

- `cond_module.{0,2,4}.weight`
- `in_proj.*`, `out_norm.weight`, `out_proj.*`
- `blocks.{i}.attention.{wq,wk,wv,wo,wk_text,wv_text,wk_speaker,wv_speaker,wk_caption,wv_caption,gate}.weight`
- `blocks.{i}.{attention_adaln,mlp_adaln}.{shift,scale,gate}_{down,up}.*`
- root aliases for `text_encoder`, `text_norm`, `speaker_encoder`, `speaker_norm`, `caption_encoder`, and `caption_norm`

`rf_dit_required_keys(ModelConfig(...))` builds the expected key set for converted checkpoints.

## Numerical comparison status

`tests/test_torch_parity.py` compares a deterministic tiny `TextToLatentRFDiT` forward pass against the upstream PyTorch implementation on fixed inputs. The test is optional in generic CI: it runs when PyTorch, MLX, and an upstream Irodori-TTS checkout are available, and can be enabled with:

```bash
IRODORI_TTS_UPSTREAM_PATH=/path/to/Irodori-TTS python -m pytest tests/test_torch_parity.py -q
```

This PR was validated locally with a temporary Python 3.11 virtualenv containing PyTorch and MLX:

```text
IRODORI_TTS_UPSTREAM_PATH=/Users/kouka/.openclaw/workspace/repos/_scratch/Irodori-TTS-upstream python -m pytest -q
23 passed
```

Known expected differences for future full-checkpoint parity work:

- MLX and PyTorch softmax/attention kernels may differ at normal floating-point tolerance.
- Current parity coverage uses a small synthetic model, not the full 500M checkpoint.
- Dropout parity is only meaningful with `dropout=0.0`; inference should keep dropout disabled.
- Full output parity still requires converted production weights with the documented no-transpose policy.
