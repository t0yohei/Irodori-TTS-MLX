# v0.2 pre-converted MLX weights redistribution audit

Issue: [#80 v0.2 redistribution/licensing audit](https://github.com/t0yohei/Irodori-TTS-MLX/issues/80)  
Parent: [#78 v0.2 hosted pre-converted MLX weights](https://github.com/t0yohei/Irodori-TTS-MLX/issues/78)  
Audit date: 2026-05-15

This is an engineering due-diligence artifact for deciding whether #81-#85 may rely on hosted pre-converted MLX weights. It is **not legal advice** and it does not approve publishing artifacts whose upstream terms change after this audit.

## Decision summary

- **Go, with conditions:** pre-converted MLX weights derived only from the three audited Irodori-TTS checkpoints may be prepared for hosted publication in v0.2 if the target repository includes the attribution/provenance text below and does not bundle codec weights, reference audio, generated audio, Hugging Face cache contents, or unaudited tokenizer/model artifacts.
- **Local-conversion-only:** any historical, third-party, fine-tuned, quantized, LoRA, renamed, architecture-modified, or otherwise unaudited checkpoint remains local-conversion-only until separately audited.
- **Do not publish as part of this issue:** this issue records eligibility only. It must not upload model weights or create a hosted weights repository.

## Candidate-family status

| Candidate family | Upstream source | Evidence observed | Converted MLX redistribution status | v0.2 eligibility |
| --- | --- | --- | --- | --- |
| Base v2 speaker-conditioned | [`Aratako/Irodori-TTS-500M-v2`](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) | Hugging Face model page and API list `License: mit`; model card says the model is released under MIT and adds ethical restrictions around impersonation/misinformation; API snapshot `8fd631cafb911dde466bc30dd558a0dc55e8ccae`. | Eligible as a derived/pre-converted MLX artifact when attribution, provenance, MIT license notice, and ethical-use restrictions are preserved. Do not bundle reference audio or generated samples. | **Go, conditional**. This repo currently marks base v2 generation as experimental, so it is not the first recommended hosted artifact despite redistribution eligibility. |
| VoiceDesign v2 caption-conditioned | [`Aratako/Irodori-TTS-500M-v2-VoiceDesign`](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) | Hugging Face model page and API list `License: mit`; model card says the model is released under MIT, is derived from base v2, and adds ethical restrictions for caption-based impersonation/misinformation; API snapshot `456e55708e7183f5c7faa1448209d54aa8991451`. | Eligible as a derived/pre-converted MLX artifact when attribution, provenance, MIT license notice, and ethical-use restrictions are preserved. Do not bundle generated samples. | **Go, recommended first candidate** because it is supported by the v0.1 contract and does not require user reference audio for the standard no-reference workflow. |
| v3 speaker-conditioned / duration-predictor | [`Aratako/Irodori-TTS-500M-v3`](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | Hugging Face model page and API list `License: mit`; model card says the model is released under MIT, is based on v2, adds duration prediction, references SilentCipher watermarking in generated outputs, and includes ethical restrictions; API snapshot `236c1e56591279fc24e3c1bf6609fc06e48dde28`. | Eligible as a derived/pre-converted MLX artifact when attribution, provenance, MIT license notice, and ethical-use restrictions are preserved. The hosted MLX weights repo should state that this project does not implement or redistribute SilentCipher itself unless added separately. | **Go, conditional**. Suitable when the hosted manifest is approved and the no-reference predicted-duration smoke path is validated for the published revision. |
| Semantic-DACVAE Japanese codec weights | [`Aratako/Semantic-DACVAE-Japanese-32dim`](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) | Hugging Face model page and API list `License: mit`; model card says the weights are derived from `facebook/dacvae-watermarked` and lists MIT; API snapshot `47376ee24834d7a05a48ebabfe3cde29b3c5e214`. | Out of scope for the first hosted MLX RF-DiT weights artifact. Runtime users should continue obtaining codec weights from the original upstream source unless a separate codec redistribution audit is opened. | **No-go for bundling in #83**. Link to the upstream codec instead. |
| Other Irodori-TTS-derived or third-party checkpoints | Any checkpoint outside the three explicit candidates above | No audit evidence in this issue. | Ambiguous. Converted weights may inherit upstream model, dataset, tokenizer, audio, or fine-tune terms not reviewed here. | **No-go / local-conversion-only** until separately audited and approved. |

## Required attribution and provenance text

Any hosted MLX weights README should include a short notice equivalent to:

> These MLX `.npz` weights were converted from the upstream Irodori-TTS checkpoint `<UPSTREAM_REPO_ID>` by the unofficial `t0yohei/Irodori-TTS-MLX` converter. The upstream model card lists the source checkpoint under the MIT license and includes ethical-use restrictions, including no impersonation and no misleading synthetic speech. This repository republishes only converted RF-DiT/text/condition/duration-predictor weights needed by the MLX runtime; it does not bundle Semantic-DACVAE codec weights, reference audio, generated samples, Hugging Face cache snapshots, or upstream source code. Users remain responsible for following the upstream model card, applicable law, and any rights attached to prompts, reference audio, and generated audio.

Also include these machine-checkable fields where practical:

- `upstream_repo_id`: one of the audited Hugging Face repository IDs above.
- `upstream_revision`: the exact Hugging Face commit used for conversion.
- `converted_with`: `t0yohei/Irodori-TTS-MLX` commit SHA and `scripts/convert_weights.py` invocation.
- `source_license`: `MIT`, with a link to the upstream model card.
- `derived_artifact`: `true`.
- `bundled_artifacts`: explicitly list the files present; do not include codec weights or samples.

## Publication checklist for #83

Before publishing a hosted converted weights repository:

1. Select one audited checkpoint family. For v0.2, the recommended first candidate is `Aratako/Irodori-TTS-500M-v2-VoiceDesign`.
2. Convert from a pinned upstream revision and record the converter commit SHA.
3. Publish only the converted MLX weights and lightweight metadata needed by this runtime.
4. Add the attribution/provenance notice above to the hosted weights README or model card.
5. Preserve the upstream MIT license reference and ethical-use restrictions.
6. Link users to the original Semantic-DACVAE codec model instead of bundling codec weights.
7. Do not include generated audio, reference audio, sample WAVs, Hugging Face cache directories, or unaudited tokenizer/model files.
8. If the upstream model card/license changes before publishing, pause and re-run this audit.

## Blockers and escalation rules

Treat any of the following as a blocker requiring human review before publication:

- upstream license metadata is missing, non-MIT, conflicting, or changed since this audit;
- the artifact includes codec weights, tokenizer/model files, generated audio, reference audio, or training data not covered here;
- the candidate is a fine-tune, quantization, LoRA merge, or third-party derivative not listed in the table;
- the hosted repository wants to claim legal approval rather than describing observed upstream evidence;
- the hosted repository cannot preserve attribution, provenance, and ethical-use restrictions.

## Sources checked

- Upstream code repository: [`Aratako/Irodori-TTS`](https://github.com/Aratako/Irodori-TTS), GitHub API reported MIT license on default branch `main`, commit `07dfa74d19e961faa499d8d365f36914fd85a97e`.
- Base v2 model card/API: [`Aratako/Irodori-TTS-500M-v2`](https://huggingface.co/Aratako/Irodori-TTS-500M-v2), API snapshot `8fd631cafb911dde466bc30dd558a0dc55e8ccae`.
- VoiceDesign v2 model card/API: [`Aratako/Irodori-TTS-500M-v2-VoiceDesign`](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign), API snapshot `456e55708e7183f5c7faa1448209d54aa8991451`.
- v3 model card/API: [`Aratako/Irodori-TTS-500M-v3`](https://huggingface.co/Aratako/Irodori-TTS-500M-v3), API snapshot `236c1e56591279fc24e3c1bf6609fc06e48dde28`.
- Codec model card/API: [`Aratako/Semantic-DACVAE-Japanese-32dim`](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), API snapshot `47376ee24834d7a05a48ebabfe3cde29b3c5e214`.
