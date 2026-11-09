"""Zapis wyników inferencji (CSV + raport TXT)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

INFERENCE_COLUMNS = (
    "timestamp",
    "prompt_idx",
    "prompt",
    "length",
    "architecture",
    "checkpoint",
    "epoch",
    "eval_perplexity",
    "device",
    "temperature",
    "prompt_tokens",
    "generated_tokens",
    "prefill_seconds",
    "generate_seconds",
    "inference_seconds",
    "tokens_per_second",
    "prompt_perplexity",
    "generate_perplexity",
    "sampling_perplexity",
    "params",
    "model_bytes",
    "model_description",
    "response",
)

INFERENCE_FILENAME = "inference.csv"
REPORT_FILENAME = "report.txt"


def append_inference_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INFERENCE_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_float(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "nan"
    return f"{v:.4f}"


def write_inference_report(path: Path, rows: list[dict[str, Any]], *, header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header:
        lines.append(header)
        lines.append("")

    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(int(row["prompt_idx"]), []).append(row)

    prompt_idxs = sorted(by_prompt)
    for pi, prompt_idx in enumerate(prompt_idxs):
        group = by_prompt[prompt_idx]
        first = group[0]
        prompt = first.get("prompt", "")
        length = first.get("length", "")
        lines.append(f"=== Prompt {pi + 1}/{len(prompt_idxs)} (length={length}): {prompt!r} ===")
        lines.append("")
        for row in group:
            arch = row.get("architecture", "")
            ckpt = row.get("checkpoint", "")
            epoch = row.get("epoch", "")
            eval_ppl = row.get("eval_perplexity", "")
            temp = row.get("temperature", "")
            lines.append(f"[{arch}] {ckpt} (epoka {epoch}, eval_ppl={eval_ppl})")
            lines.append(
                "czas: "
                f"{_fmt_float(row.get('inference_seconds'))}s "
                f"(prefill {_fmt_float(row.get('prefill_seconds'))}s + "
                f"generate {_fmt_float(row.get('generate_seconds'))}s, "
                f"{_fmt_float(row.get('tokens_per_second'))} tok/s)"
            )
            lines.append(
                "perplexity: "
                f"prompt={_fmt_float(row.get('prompt_perplexity'))}, "
                f"generate={_fmt_float(row.get('generate_perplexity'))}, "
                f"sampling(T={temp})={_fmt_float(row.get('sampling_perplexity'))}"
            )
            if row.get("params") is not None:
                lines.append(
                    f"model: {row.get('params')} param, {row.get('model_bytes')} B — "
                    f"{row.get('model_description')}"
                )
            lines.append(str(row.get("response", "")))
            lines.append("")
        if pi < len(prompt_idxs) - 1:
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
