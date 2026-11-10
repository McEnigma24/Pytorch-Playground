"""Zapis wyników inferencji (CSV + raport TXT)."""

from __future__ import annotations

import csv
import json
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
    "bench_repeats",
    "bench_tps_n",
    "bench_tps_std",
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
REPORT_SIMPLIFIED_FILENAME = "report_simplified.txt"

GEN_HEAD_CHARS = 900
GEN_TAIL_CHARS = 400


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


def _generated_text(prompt: str, response: str) -> str:
    if not response:
        return ""
    if prompt and response.startswith(prompt):
        return response[len(prompt) :]
    return response


def _gen_block_lines(
    prompt: str,
    generated: str,
    *,
    head: int = GEN_HEAD_CHARS,
    tail: int = GEN_TAIL_CHARS,
) -> list[str]:
    generated = generated.replace("\r\n", "\n").replace("\n", " ").strip()
    seed = json.dumps(prompt, ensure_ascii=False)
    lines = ["<GEN>"]
    if not generated:
        lines.append(seed)
    elif len(generated) <= head + tail + 5:
        lines.append(f"{seed}{generated}")
    else:
        lines.append(f"{seed}{generated[:head].rstrip()}")
        lines.append("...")
        lines.append(generated[-tail:].lstrip())
    lines.append("<~GEN>")
    return lines


def _group_rows_by_prompt(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(int(row["prompt_idx"]), []).append(row)
    return by_prompt


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

    by_prompt = _group_rows_by_prompt(rows)
    prompt_idxs = sorted(by_prompt)
    for pi, prompt_idx in enumerate(prompt_idxs):
        group = by_prompt[prompt_idx]
        first = group[0]
        prompt = str(first.get("prompt", ""))
        length = first.get("length", "")
        lines.append(f"=== Prompt {pi + 1}/{len(prompt_idxs)} (length={length}): {prompt!r} ===")
        lines.append("")
        for row in group:
            arch = row.get("architecture", "")
            ckpt = row.get("checkpoint", "")
            epoch = row.get("epoch", "")
            eval_ppl = row.get("eval_perplexity", "")
            temp = row.get("temperature", "")
            response = str(row.get("response", ""))
            generated = _generated_text(prompt, response)
            lines.append(f"[{arch}] {ckpt} (epoka {epoch}, eval_ppl={eval_ppl})")
            bench_n = row.get("bench_tps_n", "")
            bench_r = row.get("bench_repeats", "")
            bench_std = row.get("bench_tps_std", "")
            tps_note = _fmt_float(row.get("tokens_per_second")) + " tok/s"
            if bench_r not in (None, ""):
                tps_note += f" (śr. {bench_n}/{bench_r}, σ={_fmt_float(bench_std)})"
            lines.append(
                "czas: "
                f"{_fmt_float(row.get('inference_seconds'))}s "
                f"(prefill {_fmt_float(row.get('prefill_seconds'))}s + "
                f"generate {_fmt_float(row.get('generate_seconds'))}s, "
                f"{tps_note})"
            )
            bench_r = row.get("bench_repeats", "")
            ppl_note = (
                f"prompt={_fmt_float(row.get('prompt_perplexity'))}, "
                f"generate={_fmt_float(row.get('generate_perplexity'))}, "
                f"sampling(T={temp})={_fmt_float(row.get('sampling_perplexity'))}"
            )
            if bench_r not in (None, ""):
                ppl_note += f" (śr. z {row.get('bench_tps_n', '?')}/{bench_r} runów)"
            lines.append(f"perplexity: {ppl_note}")
            if row.get("params") is not None:
                lines.append(
                    f"model: {row.get('params')} param, {row.get('model_bytes')} B — "
                    f"{row.get('model_description')}"
                )
            lines.extend(_gen_block_lines(prompt, generated))
            lines.append("")
        if pi < len(prompt_idxs) - 1:
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_inference_report_simplified(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    header: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header:
        lines.append(header)
        lines.append("")

    by_prompt = _group_rows_by_prompt(rows)
    prompt_idxs = sorted(by_prompt)
    for pi, prompt_idx in enumerate(prompt_idxs):
        group = by_prompt[prompt_idx]
        first = group[0]
        prompt = str(first.get("prompt", ""))
        length = first.get("length", "")
        lines.append(f"=== Prompt {pi + 1}/{len(prompt_idxs)} (length={length}): {prompt!r} ===")
        lines.append("")
        for row in group:
            arch = row.get("architecture", "")
            tps = _fmt_float(row.get("tokens_per_second"))
            bench_n = row.get("bench_tps_n", "")
            bench_r = row.get("bench_repeats", "")
            if bench_r not in (None, ""):
                tps += f" tok/s (śr. {bench_n}/{bench_r})"
            else:
                tps += " tok/s"
            lines.append(f"[{arch}] {tps}")
            generated = _generated_text(prompt, str(row.get("response", "")))
            lines.extend(_gen_block_lines(prompt, generated))
            lines.append("")
        if pi < len(prompt_idxs) - 1:
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_inference_reports(
    report_dir: Path,
    rows: list[dict[str, Any]],
    *,
    header: str = "",
) -> tuple[Path, Path]:
    """Pełny report.txt + report_simplified.txt (GEN + tok/s)."""
    report_dir = Path(report_dir)
    full = report_dir / REPORT_FILENAME
    simple = report_dir / REPORT_SIMPLIFIED_FILENAME
    write_inference_report(full, rows, header=header)
    write_inference_report_simplified(simple, rows, header=header)
    return full, simple
