# MLX RF Euler sampler

`irodori_mlx.sampling.sample_euler_rf_cfg` implements the first MLX rectified-flow Euler sampler for `TextToLatentRFDiT`.

## API

The public arguments intentionally follow upstream Irodori-TTS names where practical:

- `sequence_length`: number of patched latent steps to generate.
- `num_steps`: Euler solver steps. The schedule uses `num_steps + 1` values from `0.999` to `0.0`, matching upstream's `init_scale = 0.999` behavior.
- `t_schedule_mode`: timestep schedule, either `linear` or `sway`. `linear` is the default and preserves the previous behavior.
- `sway_coeff`: Sway Sampling coefficient used when `t_schedule_mode="sway"`. The default `-1.0` matches upstream's low-step recipe knob.
- `rescale_k`, `rescale_sigma`: optional temporal score rescaling parameters. Set both together or leave both unset.
- `speaker_kv_scale`: optional multiplier for speaker context K/V projections. Requires a speaker-conditioned checkpoint.
- `speaker_kv_min_t`: timestep threshold for speaker K/V scaling. When `speaker_kv_scale` is set and this is omitted, the runtime uses the upstream default `0.9`.
- `speaker_kv_max_layers`: optional limit for applying speaker K/V scaling only to the first N diffusion layers.
- `seed`: fixed MLX RNG key for deterministic noise initialization.
- `cfg_scale_text`, `cfg_scale_speaker`, `cfg_scale_caption`: per-condition CFG scales.
- `cfg_min_t`, `cfg_max_t`: CFG active window.
- `cfg_guidance_mode`: `independent`, `joint`, `alternating`, or `reduced`.
- `truncation_factor`: optional multiplier applied to the initial noise.
- `use_context_kv_cache`: pre-project condition K/V tensors once per condition bundle.

The return value is in patched latent space:

```text
(batch, sequence_length, model.cfg.patched_latent_dim)
```

Call `unpatch_latents` before codec decoding when `latent_patch_size > 1`.

## CFG modes

### `independent`

Runs the conditional path and one unconditioned path per enabled condition in a single batched forward pass:

```text
v = v_cond
v += cfg_scale_text    * (v_cond - v_text_uncond)
v += cfg_scale_speaker * (v_cond - v_speaker_uncond)
v += cfg_scale_caption * (v_cond - v_caption_uncond)
```

### `joint`

Runs `v_cond` and a fully unconditioned path. Enabled guidance scales must be equal, matching upstream's joint-mode guard.

### `alternating`

Runs `v_cond` and one single-condition unconditioned path on CFG-active steps. It cycles by absolute diffusion step through enabled CFG conditions in upstream order: text, speaker, caption. Disabled conditions are skipped, and each selected condition uses its own scale:

```text
diffusion step 0: v = v_cond + cfg_scale_text    * (v_cond - v_text_uncond)
diffusion step 1: v = v_cond + cfg_scale_speaker * (v_cond - v_speaker_uncond)
diffusion step 2: v = v_cond + cfg_scale_caption * (v_cond - v_caption_uncond)
diffusion step 3: repeat from the first enabled condition
```

### `reduced`

A v0 lightweight mode for MLX runtime bring-up. It also uses the two-forward conditional/full-unconditional path, but permits different per-condition scales and uses the maximum enabled scale as the joint scale. This avoids the larger independent CFG batch while still exposing a guided path.

## Deviations from upstream

Implemented now:

- fixed-seed noise initialization
- upstream-style Euler timestep schedules: default `linear` and optional F5-TTS-style `sway`
- temporal score rescaling with `rescale_k` and `rescale_sigma`
- speaker K/V scaling with `speaker_kv_scale`, `speaker_kv_min_t`, and `speaker_kv_max_layers`
- text / speaker / caption CFG paths
- optional context K/V cache
- `independent`, upstream-compatible `joint` / `alternating`, and MLX-specific `reduced` CFG modes

The `sway` schedule mirrors upstream Irodori-TTS schedule construction:

```text
u = linspace(0, 1, num_steps + 1)
u = clamp(u + sway_coeff * (cos(pi / 2 * u) + u - 1), 0, 1)
t = (1 - u) * 0.999
```

Negative `sway_coeff` values allocate more schedule resolution to the noise side
of the trajectory. This control exists so MLX can carry upstream-validated
low-step recipes, such as `--t-schedule-mode sway --sway-coeff -1.0`, without
changing the default linear behavior. Matching the schedule does not imply exact
audio parity by itself because MLX and upstream still differ in runtime details
such as codec artifacts, execution dtype, and unsupported sampler options.

Temporal score rescaling follows the upstream formula after CFG velocity
calculation and before the Euler update. `rescale_k` and `rescale_sigma` must
both be positive and must be set together.

Speaker K/V scaling operates on projected context K/V caches, so enabling
`speaker_kv_scale` forces context cache use even when `use_context_kv_cache` is
otherwise disabled. The control is accepted only for speaker-conditioned
checkpoints. It applies while `t >= speaker_kv_min_t`; `speaker_kv_max_layers`
limits scaling to the first N RF-DiT blocks.

No known upstream sampler parity knobs covered by this document remain
intentionally unsupported in this v0.

`reduced` mode remains MLX-specific and is not an upstream parity mode.
