"""Wykresy porównania inferencji (model_evaluations/inference.csv)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from log_print import log_print

PROMPT_LABEL_WORDS = 4
PROMPT_LABEL_MAX_CHARS = 32


def prompt_short_label(prompt: str, *, n_words: int = PROMPT_LABEL_WORDS) -> str:
    parts = prompt.strip().split()
    if not parts:
        return "?"
    short = " ".join(parts[:n_words])
    if len(parts) > n_words:
        short += "…"
    if len(short) > PROMPT_LABEL_MAX_CHARS:
        short = short[: PROMPT_LABEL_MAX_CHARS - 1] + "…"
    return short


def load_inference_rows(csv_path: Path) -> list[dict[str, Any]]:
    path = Path(csv_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        row["prompt_idx"] = int(float(row["prompt_idx"]))
        for key in (
            "prompt_perplexity",
            "generate_perplexity",
            "sampling_perplexity",
            "tokens_per_second",
            "inference_seconds",
        ):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        records.append(
            {
                "architecture": str(row.get("architecture", "")),
                "prompt_idx": int(row["prompt_idx"]),
                "prompt_label": prompt_short_label(prompt),
                "prompt_perplexity": float(row.get("prompt_perplexity", float("nan"))),
                "tokens_per_second": float(row.get("tokens_per_second", float("nan"))),
            }
        )
    return pd.DataFrame(records)


def plot_inference_eval(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 120,
) -> list[Path]:
    """Linie = modele, oś X = prompty (pierwsze słowa seeda)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _rows_to_frame(rows)
    if df.empty:
        return []

    order = df.drop_duplicates("prompt_idx").sort_values("prompt_idx")
    idx_to_label = dict(zip(order["prompt_idx"], order["prompt_label"]))
    x_labels = [idx_to_label[i] for i in order["prompt_idx"]]
    n_prompts = len(order)
    fig_w = max(9.0, min(24.0, 0.55 * n_prompts + 4))

    saved: list[Path] = []
    sns.set_theme(style="whitegrid")

    specs = (
        ("prompt_perplexity", "Perplexity (prefill / prompt)", True),
        ("tokens_per_second", "Przepustowość generacji (tok/s)", False),
    )
    for col, title, log_y in specs:
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        for arch in sorted(df["architecture"].unique()):
            sub = df[df["architecture"] == arch].sort_values("prompt_idx")
            ax.plot(
                sub["prompt_idx"],
                sub[col],
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=arch,
            )
        ax.set_xticks(order["prompt_idx"])
        ax.set_xticklabels(x_labels, rotation=40, ha="right")
        ax.set_xlabel("prompt (pierwsze słowa seeda)")
        ax.set_ylabel(col)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out = output_dir / f"{col}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        saved.append(out)
        log_print(f"Wykres inferencji → {out}")

    return saved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Wykresy z model_evaluations/.../inference.csv")
    parser.add_argument(
        "eval_dir",
        type=Path,
        help="Katalog runu (z inference.csv)",
    )
    args = parser.parse_args()
    csv_path = args.eval_dir / "inference.csv"
    rows = load_inference_rows(csv_path)
    plot_inference_eval(rows, args.eval_dir)


if __name__ == "__main__":
    main()
