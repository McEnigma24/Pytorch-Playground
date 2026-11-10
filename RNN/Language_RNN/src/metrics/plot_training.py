"""Wykresy pojedynczego runu treningu (saved_models/*/metrics.csv)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from log_print import log_print
from metrics.run_metrics import META_FILENAME, METRICS_FILENAME, load_run_metrics


def _load_epoch_frame(save_dir: Path) -> pd.DataFrame:
    rows = load_run_metrics(save_dir)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    for col in (
        "train_loss",
        "train_perplexity",
        "eval_loss",
        "eval_perplexity",
        "train_seconds",
        "eval_seconds",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("epoch")


def _load_train_total_seconds(save_dir: Path) -> float | None:
    meta_path = save_dir / META_FILENAME
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = meta.get("train_total_seconds")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def plot_training_run(save_dir: Path, *, dpi: int = 120) -> list[Path]:
    """Loss, perplexity (train/eval), czasy epoki + podsumowanie całego treningu."""
    save_dir = Path(save_dir)
    if not (save_dir / METRICS_FILENAME).is_file():
        log_print(f"Brak {METRICS_FILENAME} w {save_dir} — pomijam wykresy treningu")
        return []

    df = _load_epoch_frame(save_dir)
    if df.empty:
        return []

    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid")
    arch = save_dir.name.split("-", 1)[0]

    # Loss #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    ax.plot(df["epoch"], df["eval_loss"], marker="s", label="eval_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Loss — {arch}")
    ax.legend()
    fig.tight_layout()
    out = plots_dir / "loss.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    # Perplexity #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["train_perplexity"], marker="o", label="train_perplexity")
    ax.plot(df["epoch"], df["eval_perplexity"], marker="s", label="eval_perplexity")
    ax.set_xlabel("epoch")
    ax.set_ylabel("perplexity")
    ax.set_title(f"Perplexity — {arch}")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    out = plots_dir / "perplexity.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    # Czas epoki #
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    epochs = df["epoch"].tolist()
    x = list(range(len(epochs)))
    ax.bar([i - width / 2 for i in x], df["train_seconds"], width=width, label="train_seconds")
    ax.bar([i + width / 2 for i in x], df["eval_seconds"], width=width, label="eval_seconds")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(e)) for e in epochs])
    ax.set_xlabel("epoch")
    ax.set_ylabel("sekundy")
    ax.set_title(f"Czas epoki (train + eval) — {arch}")
    ax.legend()
    fig.tight_layout()
    out = plots_dir / "epoch_timing.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    # Cały trening #
    total_wall = _load_train_total_seconds(save_dir)
    sum_train = float(df["train_seconds"].sum()) if "train_seconds" in df else float("nan")
    sum_eval = float(df["eval_seconds"].sum()) if "eval_seconds" in df else float("nan")
    sum_logged = sum_train + sum_eval

    fig, ax = plt.subplots(figsize=(7, 4))
    labels: list[str] = []
    values: list[float] = []
    if total_wall is not None:
        labels.append("cały run (wall)")
        values.append(total_wall)
    if sum_logged == sum_logged:
        labels.append("suma train+eval (log)")
        values.append(sum_logged)
    if labels:
        ax.barh(labels, values, color=["#4c72b0", "#55a868"][: len(labels)])
        ax.set_xlabel("sekundy")
        ax.set_title(f"Czas treningu — {arch}")
        for i, v in enumerate(values):
            ax.text(v, i, f"  {v:.1f}s", va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "brak danych czasu", ha="center", va="center")
    fig.tight_layout()
    out = plots_dir / "training_total_timing.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    for p in saved:
        log_print(f"Wykres treningu → {p}")
    return saved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Wykresy treningu z saved_models/<run>/metrics.csv")
    parser.add_argument("save_dir", type=Path, help="Katalog runu treningu")
    args = parser.parse_args()
    plot_training_run(args.save_dir)


if __name__ == "__main__":
    main()
