# MLX RF Euler sampler

`irodori_mlx.sampling.sample_euler_rf_cfg` implements the first MLX rectified-flow Euler sampler for `TextToLatentRFDiT`.

## API

The public arguments intentionally follow upstream Irodori-TTS names where practical:

- `sequence_length`: number of patched latent steps to generate.
- `num_steps`: Euler solver steps. The schedule uses `num_steps + 1` values from `0.999` to `0.0`, matching upstream's `init_scale = 0.999` behavior.
- `t_schedule_mode`: timestep schedule, either `linear` or `sway`. `linear` is the default and preserves the previous behavior.
- `sway_coeff`: Sway Sampling coefficient used when `t_schedule_mode="sway"`. The default `-1.0` matches upstream's low-step recipe knob.
- `seed`: fixed MLX RNG key for deterministic noise initialization.
- `cfg_scale_text`, `cfg_scale_speaker`, `cfg_scale_caption`: per-condition CFG scales.
- `cfg_min_t`, `cfg_max_t`: CFG active window.
- `cfg_guidance_mode`: `independent`, `joint`, or `reduced`.
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

### `reduced`

A v0 lightweight mode for MLX runtime bring-up. It also uses the two-forward conditional/full-unconditional path, but permits different per-condition scales and uses the maximum enabled scale as the joint scale. This avoids the larger independent CFG batch while still exposing a guided path.

## Deviations from upstream

Implemented now:

- fixed-seed noise initialization
- upstream-style Euler timestep schedules: default `linear` and optional F5-TTS-style `sway`
- text / speaker / caption CFG paths
- optional context K/V cache
- `independent`, `joint`, and MLX-specific `reduced` CFG modes

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

Not implemented in this v0:

- upstream `alternating` CFG mode
- temporal score rescale (`rescale_k`, `rescale_sigma`)
- speaker K/V force scaling (`speaker_kv_scale`, `speaker_kv_max_layers`, `speaker_kv_min_t`)

These are intentionally left for follow-up runtime parity work so this PR stays focused on the sampler loop and CFG mechanics.
