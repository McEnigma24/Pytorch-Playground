"""Extract and filter text from PDF for TTS (deletions only, plus dehyphenation)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from tts.infer_Qwen_3_tts import strip_delimited_segments

_BRACKET_PAIRS: list[tuple[str, str]] = [("[", "]")]


def _dehyphenate(text: str) -> str:
    """Join words split across lines with a trailing hyphen."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def filter_text_for_tts(raw: str) -> str:
    """PDF text with bracketed segments removed and whitespace normalized."""
    text = _dehyphenate(raw)
    text = strip_delimited_segments(text, _BRACKET_PAIRS)
    return _normalize_whitespace(text)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        raw = "\n".join(parts)
    finally:
        doc.close()
    return filter_text_for_tts(raw)
