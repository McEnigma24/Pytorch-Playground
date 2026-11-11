"""Extract and filter text from PDF for TTS (deletions only, plus dehyphenation)."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from tts.infer_Qwen_3_tts import strip_delimited_segments

REF_HEADER_RE = re.compile(
    r"^\s*(References|Bibliography|Bibliografia|Literatura)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
SECTION_AFTER_AUTHORS_RE = re.compile(
    r"^\s*((\d+\s+)?(Abstract|ABSTRACT|Streszczenie|Summary|Introduction|Wstęp))\s*$",
    re.IGNORECASE | re.MULTILINE,
)
AUTHOR_AFFIL_RE = re.compile(
    r"(?i)(university|universit|institut|laborator|department|faculty|"
    r"school of|college|academy|mila|arxiv|preprint|"
    r"\binc\.|\bltd\.|\bcorp\.|@|\.edu\b|\.ac\.)"
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
PAGE_NUM_LINE_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_RULE_CHARS = set("-_=·.•‒–—―_＿\u2010\u2011\u2012\u2013\u2014\u2015\u2500\u2501")
TABLE_CAPTION_RE = re.compile(
    r"(?i)^\s*(table|tabela)\s*\d+\s*[.:]?\s*.+$"
)


@dataclass(frozen=True)
class FilterSpec:
    id: str
    label: str
    default: bool


FILTERS: list[FilterSpec] = [
    FilterSpec("square_brackets", "Usuń tekst w [ nawiasach kwadratowych ]", True),
    FilterSpec("authors", "Usuń autorów i afiliacje (zostaw tytuł; do Abstract/Wstęp)", True),
    FilterSpec("round_brackets", "Usuń tekst w ( nawiasach okrągłych )", False),
    FilterSpec("references", "Usuń sekcję References / Bibliography", True),
    FilterSpec("urls_emails", "Usuń adresy URL i e-mail", True),
    FilterSpec("page_numbers", "Usuń linie z samym numerem strony", True),
    FilterSpec("repeated_lines", "Usuń powtarzające się linie (nagłówki/stopki)", True),
    FilterSpec(
        "page_bottom_rule",
        "Usuń dopiski pod linią na dole strony (footnotes / preprint)",
        True,
    ),
    FilterSpec("dehyphenate", "Sklej wyrazy przeniesione myślnikiem na końcu linii", True),
    FilterSpec(
        "tables_label",
        "Tabele: wstaw etykietę (Tabela N / podpis), zamiast komórek z liczbami",
        True,
    ),
    FilterSpec("tables_skip", "Tabele: usuń bez etykiety (nadpisuje etykietę)", False),
]

_DEFAULT_FILTER_IDS = {f.id for f in FILTERS if f.default}


def filters_catalog() -> list[dict[str, object]]:
    return [{"id": f.id, "label": f.label, "default": f.default} for f in FILTERS]


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _apply_nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _apply_references(text: str) -> str:
    match = REF_HEADER_RE.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def _looks_like_author_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if EMAIL_RE.search(s) or AUTHOR_AFFIL_RE.search(s):
        return True
    if re.search(r"(?i)\b(and|or)\b", s) and re.search(r"[A-Z][a-z]+", s):
        return True
    words = s.split()
    if len(words) <= 8 and all(re.match(r"^[\w\-'.]+$", w) for w in words):
        caps = sum(1 for w in words if w[:1].isupper())
        if caps >= 2 and len(s) < 120:
            return True
    return False


def _strip_author_block(text: str) -> str:
    """Drop author/affiliation lines after title, up to Abstract/Introduction."""
    section = SECTION_AFTER_AUTHORS_RE.search(text)
    if section:
        head = text[: section.start()].rstrip()
        tail = text[section.start() :].lstrip()
    else:
        head = text
        tail = ""

    lines = head.splitlines()
    if not lines:
        return text if not tail else tail

    title_lines: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            if title_lines:
                i += 1
                break
            i += 1
            continue
        if title_lines and _looks_like_author_line(ln):
            break
        if title_lines and len(title_lines) >= 3 and _looks_like_author_line(ln):
            break
        title_lines.append(lines[i])
        i += 1
        if len(title_lines) >= 4:
            break

    while i < len(lines):
        if not _looks_like_author_line(lines[i]) and len(lines[i].strip()) > 80:
            title_lines.append(lines[i])
        i += 1

    for j in range(i, len(lines)):
        if _looks_like_author_line(lines[j]):
            continue
        if lines[j].strip():
            title_lines.append(lines[j])

    kept_head = "\n".join(title_lines).strip()
    if kept_head and tail:
        return f"{kept_head}\n\n{tail}"
    if tail:
        return tail
    return kept_head or text


def _is_horizontal_rule_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    rule_count = sum(1 for ch in stripped if ch in _RULE_CHARS or ch.isspace())
    if rule_count < len(stripped) * 0.7:
        return False
    non_space = [ch for ch in stripped if not ch.isspace()]
    return len(non_space) >= 4 and all(ch in _RULE_CHARS for ch in non_space)


def _strip_text_below_page_rule(page_text: str) -> str:
    lines = page_text.splitlines()
    last_rule: int | None = None
    for i, line in enumerate(lines):
        if _is_horizontal_rule_line(line):
            last_rule = i
    if last_rule is None:
        return page_text
    below = [ln for ln in lines[last_rule + 1 :] if ln.strip()]
    if not below:
        return page_text
    return "\n".join(lines[:last_rule]).rstrip()


def _find_bottom_rule_y(page: pymupdf.Page) -> float | None:
    page_h = float(page.rect.height)
    page_w = float(page.rect.width)
    min_y = page_h * 0.45
    min_width = page_w * 0.4
    best_y: float | None = None

    try:
        drawings = page.get_drawings()
    except (RuntimeError, ValueError):
        drawings = []

    for path in drawings:
        for item in path.get("items", []):
            op = item[0]
            if op != "l":
                continue
            p0, p1 = item[1], item[2]
            if abs(float(p0.y) - float(p1.y)) > 2.0:
                continue
            y = (float(p0.y) + float(p1.y)) / 2.0
            width = abs(float(p1.x) - float(p0.x))
            if y < min_y or width < min_width:
                continue
            if best_y is None or y > best_y:
                best_y = y
    return best_y


def _extract_page_text_below_rule(page: pymupdf.Page, y_rule: float) -> str:
    margin = 4.0
    cutoff = y_rule - margin
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    blocks.sort(key=lambda b: (b[1], b[0]))
    parts: list[str] = []
    for b in blocks:
        if float(b[3]) <= cutoff:
            text = (b[4] or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _apply_page_bottom_rule(page: pymupdf.Page, page_text: str) -> str:
    stripped = _strip_text_below_page_rule(page_text)
    if stripped != page_text:
        return stripped
    y_rule = _find_bottom_rule_y(page)
    if y_rule is None:
        return page_text
    return _extract_page_text_below_rule(page, y_rule)


def _filter_repeated_lines_on_pages(page_texts: list[str]) -> list[str]:
    """Drop lines that appear on >= 3 pages (headers/footers) — per page, before in-text filters."""
    line_counts: dict[str, int] = defaultdict(int)
    for page in page_texts:
        seen_on_page: set[str] = set()
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) < 4:
                continue
            if stripped not in seen_on_page:
                seen_on_page.add(stripped)
                line_counts[stripped] += 1

    drop = {line for line, count in line_counts.items() if count >= 3}
    if not drop:
        return page_texts

    out_pages: list[str] = []
    for page in page_texts:
        kept = [ln for ln in page.splitlines() if ln.strip() not in drop]
        out_pages.append("\n".join(kept))
    return out_pages


def _strip_square_brackets(text: str) -> str:
    pairs = [("[", "]"), ("\uff3b", "\uff3d")]  # ASCII + fullwidth
    text = strip_delimited_segments(text, pairs)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"\uff3b[^\uff3d]*\uff3d", "", text)
    return text


def _strip_round_brackets(text: str) -> str:
    text = strip_delimited_segments(text, [("(", ")")])
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\([^()]*\)", "", text)
    return text


def apply_filters(raw: str, *, filter_ids: set[str]) -> str:
    text = _apply_nfkc(raw)

    if "dehyphenate" in filter_ids:
        text = _dehyphenate(text)
    if "authors" in filter_ids:
        text = _strip_author_block(text)
    if "square_brackets" in filter_ids:
        text = _strip_square_brackets(text)
    if "round_brackets" in filter_ids:
        text = _strip_round_brackets(text)
    if "urls_emails" in filter_ids:
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
    if "page_numbers" in filter_ids:
        text = PAGE_NUM_LINE_RE.sub("", text)
    if "references" in filter_ids:
        text = _apply_references(text)

    return _normalize_whitespace(text)


def _table_mode(filter_ids: set[str]) -> str | None:
    if "tables_skip" in filter_ids:
        return "skip"
    if "tables_label" in filter_ids:
        return "label"
    return None


def _intersect_area(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _block_mostly_inside(block: pymupdf.Rect, table: pymupdf.Rect, ratio: float = 0.55) -> bool:
    area = block.get_area()
    if area <= 0:
        return False
    return _intersect_area(block, table) / area >= ratio


def _table_label(n: int, language: str, caption: str | None) -> str:
    if caption and caption.strip():
        return caption.strip()
    word = "Table" if language == "English" else "Tabela"
    return f"{word} {n}"


def _caption_near_table(block: pymupdf.Rect, text: str, table: pymupdf.Rect) -> bool:
    if not TABLE_CAPTION_RE.match(text.strip()):
        return False
    if block.y1 > table.y0 + 4:
        return False
    gap = table.y0 - block.y1
    return 0 <= gap < 72


def _extract_page_text(page: pymupdf.Page, table_mode: str | None, counter: list[int], language: str) -> str:
    if table_mode is None:
        return page.get_text("text")

    table_rects: list[pymupdf.Rect] = []
    try:
        finder = page.find_tables()
        for tab in finder.tables:
            table_rects.append(pymupdf.Rect(tab.bbox))
    except (RuntimeError, ValueError):
        table_rects = []

    if not table_rects:
        return page.get_text("text")

    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    blocks.sort(key=lambda b: (b[1], b[0]))

    labels_emitted: set[int] = set()
    captions: dict[int, str] = {}

    for b in blocks:
        text = (b[4] or "").strip()
        if not text:
            continue
        rect = pymupdf.Rect(b[0], b[1], b[2], b[3])
        for ti, trect in enumerate(table_rects):
            if _caption_near_table(rect, text, trect):
                captions[ti] = text

    parts: list[str] = []
    for b in blocks:
        text = (b[4] or "").strip()
        if not text:
            continue
        rect = pymupdf.Rect(b[0], b[1], b[2], b[3])

        in_table: int | None = None
        for ti, trect in enumerate(table_rects):
            if _block_mostly_inside(rect, trect):
                in_table = ti
                break

        if in_table is None:
            parts.append(text)
            continue

        if table_mode == "skip":
            continue

        if in_table in labels_emitted:
            continue

        counter[0] += 1
        labels_emitted.add(in_table)
        parts.append(_table_label(counter[0], language, captions.get(in_table)))

    return "\n\n".join(parts)


def extract_text_from_pdf(
    pdf_path: Path,
    *,
    page_from: int = 1,
    page_to: int | None = None,
    filter_ids: set[str] | None = None,
    language: str = "English",
) -> tuple[str, int]:
    """Return (filtered text, total page count in PDF)."""
    active = filter_ids if filter_ids is not None else _DEFAULT_FILTER_IDS
    table_mode = _table_mode(active)

    doc = pymupdf.open(pdf_path)
    try:
        num_pages = doc.page_count
        last = page_to if page_to is not None else num_pages
        page_from = max(1, min(page_from, num_pages))
        last = max(page_from, min(last, num_pages))

        table_counter = [0]
        page_texts: list[str] = []
        strip_bottom = "page_bottom_rule" in active
        for page_idx in range(page_from - 1, last):
            page = doc[page_idx]
            text = _extract_page_text(page, table_mode, table_counter, language)
            if strip_bottom:
                text = _apply_page_bottom_rule(page, text)
            page_texts.append(text)

        if "repeated_lines" in active:
            page_texts = _filter_repeated_lines_on_pages(page_texts)

        raw = "\n".join(page_texts)
        text = apply_filters(raw, filter_ids=active)
        return text, num_pages
    finally:
        doc.close()


def pdf_page_count(pdf_path: Path) -> int:
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def is_subsequence_of(candidate: str, source: str) -> bool:
    """True if candidate is a subsequence of source (deletions only)."""
    if not candidate:
        return True
    i = 0
    for ch in source:
        if ch == candidate[i]:
            i += 1
            if i == len(candidate):
                return True
    return False
