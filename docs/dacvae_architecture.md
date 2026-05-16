# DACVAE architecture and checkpoint contract for the MLX port

Issue: [#111](https://github.com/t0yohei/Irodori-TTS-MLX/issues/111)  
Parent epic: [#123](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

This note is the research input for the DACVAE implementation issues #112-#115.
It describes the upstream codec boundary that the MLX port must match before it
is used as a replacement for the current PyTorch bridge.

Implementation consumers:
[#112](https://github.com/t0yohei/Irodori-TTS-MLX/issues/112),
[#113](https://github.com/t0yohei/Irodori-TTS-MLX/issues/113),
[#114](https://github.com/t0yohei/Irodori-TTS-MLX/issues/114), and
[#115](https://github.com/t0yohei/Irodori-TTS-MLX/issues/115).

## Sources inspected

- Upstream Irodori wrapper: `irodori_tts.codec.DACVAECodec`
- Upstream runtime use: `irodori_tts.inference_runtime.RuntimeKey` and reference/decode paths
- Codec implementation: `facebookresearch/dacvae`, class `dacvae.DACVAE`
- Public codec artifact: `Aratako/Semantic-DACVAE-Japanese-32dim/weights.pth`
- Hugging Face model card for `Aratako/Semantic-DACVAE-Japanese-32dim`

The public codec artifact is a PyTorch `.pth` file, not a safetensors file. The
contract below is therefore a runtime and logical state-dict contract. The first
real converter must still load the exact `dacvae` implementation and inspect the
actual `state_dict` keys before writing MLX weights.

## Runtime constants

| Field | Value | Porting implication |
| --- | ---: | --- |
| Codec repo | `Aratako/Semantic-DACVAE-Japanese-32dim` | Default codec for base v2, VoiceDesign v2, and v3 paths. |
| Artifact | `weights.pth` | Converter needs PyTorch checkpoint loading, not safetensors header parsing. |
| Sample rate | `48000` | Reference audio is resampled before encode; decoded WAVs are emitted at 48 kHz. |
| Hop length | `512` | One latent frame covers 512 samples, about 93.75 frames/sec. |
| Runtime latent dim | `32` | RF-DiT checkpoints use `ModelConfig.latent_dim=32`. |
| Runtime latent layout | `(batch, latent_steps, 32)` | Irodori transposes the DACVAE layout before/after codec calls. |
| DACVAE latent layout | `(batch, 32, latent_steps)` | Native `dacvae.DACVAE.encode/decode` layout. |

`hop_length` is the product of the encoder rates `[2, 4, 8, 8]`. Manual duration
resolution in Irodori computes `ceil(seconds * sample_rate / hop_length)`.

## Architecture summary

The public 32-dim Japanese codec is derived from `facebook/dacvae-watermarked`
with WavLM semantic distillation and a compressed continuous latent size. The
core shape is:

```text
mono waveform (B,1,S)
  -> optional resample/loudness normalize/peak safety
  -> reflect pad to hop multiple
  -> DACVAE encoder: stem Conv1d + four downsample EncoderBlocks
  -> VAEBottleneck.in_proj: mean[32] + scale[32]
  -> Irodori deterministic encode: use mean only
  -> RF-DiT latent sequence in (B,T,32)
  -> DACVAE decode input transposed to (B,32,T)
  -> VAEBottleneck.out_proj
  -> DACVAE decoder: upsample stack + residual units
  -> watermark branch bypassed for Irodori
  -> mono waveform (B,1,S')
```

The upstream `DACVAECodec.load` wrapper performs the Irodori-specific codec
setup:

- downloads `weights.pth` from the codec repo when needed;
- loads `dacvae.DACVAE` and moves it to the requested device/dtype;
- sets `decoder.alpha = 0.0`;
- replaces `decoder.watermark` with `decoder.wm_model.encoder_block.forward_no_conv`
  when the watermark module exists;
- infers `sample_rate` from `model.sample_rate`;
- infers runtime `latent_dim` by encoding a dummy waveform.

## Required tensors and shapes

The exact serialized PyTorch keys depend on the `dacvae` version because the
codec uses parametrized weight normalization. A converter should treat these as
the required logical tensors and accept the corresponding PyTorch
`parametrizations.weight.original*` storage keys.

| Group | Logical key pattern | Required for | Shape contract |
| --- | --- | --- | --- |
| Encoder stem/stages | `encoder.block.*` | encode | 1-D conv weights from mono audio through downsample rates `[2,4,8,8]`; final encoder channel count feeds `quantizer.in_proj`. |
| VAE mean/scale projection | `quantizer.in_proj.*` | encode | Logical Conv1d to 64 channels; split into `mean` and `scale`, each 32 channels. Deterministic Irodori encode uses `mean`. |
| VAE decoder projection | `quantizer.out_proj.*` | decode | Logical Conv1d from 32 latent channels back to the decoder channel space. |
| Decoder upsample/residual stack | `decoder.*` | decode | Transposed-conv/residual stack using decoder rates `[8,8,4,2]`; output is mono waveform. |
| Watermark modules | `decoder.wm_model.*` when present | decode setup | Present in the watermarked base architecture, but bypassed for Irodori by the wrapper. |

For parametrized Conv1d/ConvTranspose1d modules, expect state-dict storage like:

```text
<module>.bias
<module>.parametrizations.weight.original0
<module>.parametrizations.weight.original1
```

The converter should validate the loaded model rather than guessing all nested
module widths from filenames. Minimum acceptance for the first real converter:

- `model.sample_rate == 48000`
- `model.hop_length == 512`
- deterministic encode returns `(B, 32, T)` before Irodori transposes to `(B,T,32)`
- decode accepts `(B, 32, T)` and returns mono `(B,1,S)`
- `quantizer.in_proj` output channel count is `64`
- `quantizer.out_proj` input channel count is `32`

## Preprocessing and postprocessing

Encode input behavior:

- accept `(B,C,S)` or `(C,S)` waveforms;
- downmix multi-channel audio to mono;
- resample to the codec sample rate;
- optionally normalize each item to `-16.0` dB using `audiotools.AudioSignal`;
- when normalization is disabled, optionally scale down peaks above 1.0;
- reflect-pad to a hop-length multiple before the encoder.

Decode behavior:

- accept runtime latents as `(B,T,32)`;
- transpose to `(B,32,T)` before calling `model.decode`;
- return mono waveform at 48 kHz;
- caller trims to the requested sample count when duration is known.

## Shared use across checkpoint families

The same codec repo is used across the current supported Irodori families:

- `base_v2`: reference audio is encoded for speaker/style conditioning and
  generated latents are decoded to waveform.
- `voicedesign_v2`: caption conditioning replaces the speaker/reference branch,
  but generated 32-dim latents still decode through this codec.
- `v3`: speaker/reference conditioning remains compatible with this codec; the
  RF-DiT checkpoint adds duration-predictor weights.

This means the MLX codec port should be family-neutral. Family-specific behavior
belongs in RF-DiT config/runtime code, not in the codec artifact.

## Hosted and existing MLX artifacts

The existing `--codec-runtime-mode mlx` path and its `.npz` fixture contract are
useful as runtime plumbing tests only. They prove that the app can route encode
and decode through an MLX-owned object with `sample_rate`, `hop_length`, and
`latent_dim` metadata, but they are linear toy projections. They are not a
converted Semantic-DACVAE checkpoint and they are not acoustic parity evidence.

Hosted/pre-converted RF-DiT artifacts do not contain DACVAE weights. A future
hosted codec artifact should therefore be versioned separately and include:

- codec source repo and revision;
- exact `dacvae` package revision used for conversion;
- `sample_rate`, `hop_length`, and `latent_dim`;
- complete converted encoder, `quantizer.in_proj`, `quantizer.out_proj`, decoder,
  and watermark-bypass metadata;
- license/distribution review status.

## Unknowns and blockers for #112-#115

- Exact serialized key names and nested tensor shapes must be inspected from
  `weights.pth` with the exact `dacvae` version before implementing conversion.
- Encode parity must compare the deterministic mean latent path, not the
  stochastic VAE sample path.
- Decode parity must include the Irodori watermark bypass; enabling the default
  watermark path is expected to change output.
- Padding and tail trimming need fixtures because encode pads to hop multiples
  while generation trims decoded audio to requested samples.
- Redistributing converted codec weights requires the existing license and
  distribution policy review; tests may use synthetic fixtures, but hosted codec
  weights need explicit provenance.

The machine-readable summary used by tests is
[`docs/dacvae_codec_contract.json`](dacvae_codec_contract.json).
