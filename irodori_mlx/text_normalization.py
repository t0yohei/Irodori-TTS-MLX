from __future__ import annotations

import re
import unicodedata

# Keep this intentionally small and source-compatible with the upstream
# irodori_tts.text_normalization.normalize_text helper used by the PyTorch
# inference runtime. The MLX runtime applies it before tokenizer input so the
# token stream matches upstream for common Japanese prompt cleanup cases.
SIMPLE_REPLACE_MAP: dict[str, str] = {
    "\t": "",
    "[n]": "",
    r"\[n\]": "",
    "　": "",
    "？": "?",
    "！": "!",
    "♥": "♡",
    "●": "○",
    "◯": "○",
    "〇": "○",
}

REGEX_REPLACE_MAP: dict[re.Pattern[str], str] = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]"): "",
    re.compile(r"[\uff5e\u301C]"): "ー",
    re.compile(r"…{3,}"): "……",
}


def strip_outer_brackets(text: str) -> str:
    """Remove one or more bracket pairs only when they enclose the full string."""

    pairs = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}

    while True:
        if len(text) < 2:
            break

        start_char = text[0]
        end_char = text[-1]

        if start_char in pairs and pairs[start_char] == end_char:
            depth = 0
            is_enclosing_all = True

            for i, char in enumerate(text):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1

                if depth == 0 and i < len(text) - 1:
                    is_enclosing_all = False
                    break

            if is_enclosing_all and depth == 0:
                text = text[1:-1]
                continue

        break

    return text


def normalize_text(text: str) -> str:
    """Normalize prompt text with the upstream Irodori-TTS inference policy."""

    text = str(text)
    for old, new in SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)

    for pattern, replacement in REGEX_REPLACE_MAP.items():
        text = pattern.sub(replacement, text)

    text = strip_outer_brackets(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\.{3,}", lambda match: "……" if len(match.group(0)) >= 6 else "…", text)
    text = text.replace("..", "…")
    return re.sub(r"…{3,}", "……", text)
