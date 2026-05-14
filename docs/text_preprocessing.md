# Text normalization and tokenization compatibility

This document is the v0.1 contract for prompt text preprocessing in `irodori-tts-mlx`.

## Boundary

`irodori-tts-mlx` owns the runtime input that reaches the MLX text and caption encoders:

1. normalize prompt text with the same small policy used by upstream `irodori_tts.inference_runtime`;
2. tokenize prompt text with the configured Hugging Face tokenizer;
3. prepend the configured BOS token when `ModelConfig.text_add_bos` is true;
4. right-pad or truncate to `MLXRuntimeConfig.text_max_length`;
5. build the boolean text mask consumed by MLX encoders and the duration predictor.

The repository still does **not** own upstream checkpoint assets, tokenizer assets, or the PyTorch DACVAE codec. Those remain external runtime dependencies. The text tokenizer defaults to `ModelConfig.text_tokenizer_repo` (`sbintuitions/sarashina2.2-0.5b` unless checkpoint config overrides it), and caption-conditioned checkpoints use `caption_tokenizer_repo_resolved`.

## Prompt normalization policy

Prompt `--text` is normalized before tokenization and before duration-feature extraction. The policy mirrors upstream `irodori_tts.text_normalization.normalize_text`:

- remove tabs, `[n]`, escaped `\[n\]`, ASCII/full-width spaces, and a small set of decorative symbols;
- map Japanese full-width question/exclamation marks to ASCII `?` / `!`;
- map `♥`, `●`, `◯`, and `〇` to their upstream canonical forms;
- strip one or more outer bracket pairs only when they wrap the whole text;
- apply Unicode NFKC normalization;
- replace ASCII dot runs with the same sequential `...` / `..` substitutions used upstream.

If the prompt becomes empty after normalization and trimming, generation fails early instead of silently producing an unconditional text path.

Representative examples:

| Input | Runtime prompt |
| --- | --- |
| `（今日は　いい天気ですね！）` | `今日はいい天気ですね!` |
| `〖今日は いい 天気ですね〗` | `今日はいい天気ですね` |
| `１２３円です...` | `123円です…` |
| `あ～～～！` | `あーーー!` |
| `[n]\t　` | rejected as empty |

## Tokenization contract

`PretrainedTextTokenizer` intentionally follows the upstream batch-tokenizer semantics at the fixed-length runtime boundary:

- call the Hugging Face tokenizer with `add_special_tokens=False`;
- prepend BOS explicitly when enabled by model config;
- force right-padding for stable positional behavior;
- use the tokenizer pad token, falling back to EOS as pad when necessary;
- create a `True` mask for emitted tokens and `False` for padded positions;
- truncate after BOS insertion when token count exceeds `max_length`.

This means v0.1 compatibility is at the **normalized-text + tokenizer-id + mask** boundary, not at a Japanese linguistic frontend boundary. There is no separate kana/phoneme/G2P frontend in this repository.

## Caption text

Caption text is a separate conditioning stream for VoiceDesign-style checkpoints. It uses the resolved caption tokenizer and caption max length. Empty or whitespace-only captions are encoded but then forced to an all-false caption mask, matching the existing unconditional-caption path. Prompt normalization does not alter caption strings beyond the current runtime whitespace handling.

## Known limitations

- Tokenizer output depends on the exact tokenizer revision available in the user's Hugging Face cache or download path. Pin tokenizer/checkpoint revisions externally when reproducibility matters.
- The project does not yet expose a CLI flag to disable prompt normalization for experiments.
- Only token/runtime-input behavior is covered by lightweight local tests; acoustic quality still depends on the checkpoint, DACVAE bridge, sampler settings, and runtime environment.

## Validation

Local contract tests cover representative Japanese prompt cleanup, BOS/right-padding/truncation, EOS pad fallback, normalized text reaching both tokenizer input and duration features, and empty-normalized-prompt rejection:

```bash
python -m pytest tests/test_runtime_bridge.py
```
