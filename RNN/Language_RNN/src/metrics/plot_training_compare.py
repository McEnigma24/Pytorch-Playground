"""Porównanie metryk treningu wielu modeli (model_evaluations)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from log_print import log_print
from metrics.run_metrics import summarize_training_run

TRAINING_COMPARE_FILENAME = "training_compare.csv"
TRAINING_COMPARE_PLOT = "training_compare.png"


def _model_label(architecture: str, summary: dict[str, Any]) -> str:
    run_id = str(summary.get("run_id") or "")
    if len(run_id) > 28:
        run_id = run_id[:25] + "…"
    if run_id and run_id not in architecture:
        return f"{architecture}\n({run_id})"
    return architecture


def load_training_summaries(
    models: list[tuple[str, Path | str]],
) -> list[dict[str, Any]]:
    """models: (architecture, save_dir runu treningu)."""
    out: list[dict[str, Any]] = []
    for architecture, save_dir in models:
        summary = summarize_training_run(Path(save_dir))
        summary["compare_architecture"] = architecture
        out.append(summary)
    return out


def write_training_compare_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "compare_architecture",
        "architecture",
        "run_id",
        "save_dir",
        "epochs",
        "train_time_logged_seconds",
        "train_time_wall_seconds",
        "eval_perplexity_best",
        "eval_perplexity_final",
        "train_perplexity_final",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k) for k in fieldnames}
            writer.writerow(row)


def plot_training_model_compare(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 120,
) -> Path | None:
    """Jeden wykres: czas treningu (s) + eval perplexity z treningu (log), modele obok siebie."""
    if not summaries:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    labels = [
        _model_label(str(s.get("compare_architecture") or s.get("architecture", "?")), s)
        for s in summaries
    ]
    times = [float(s.get("train_time_logged_seconds") or 0.0) for s in summaries]
    if all(t <= 0 for t in times):
        for i, s in enumerate(summaries):
            wall = s.get("train_time_wall_seconds")
            if wall is not None:
                times[i] = float(wall)

    eval_ppl = [float(s.get("eval_perplexity_best", float("nan"))) for s in summaries]
    n = len(summaries)
    fig_w = max(8.0, min(16.0, 2.8 * n + 2))
    fig, (ax_time, ax_ppl) = plt.subplots(1, 2, figsize=(fig_w, 5))
    x = list(range(n))
    colors = sns.color_palette("husl", n)

    bars = ax_time.bar(x, times, color=colors)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax_time.set_ylabel("sekundy")
    ax_time.set_title("Całkowity czas treningu (suma train+eval z logów)")
    for bar, val in zip(bars, times):
        if val > 0:
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.0f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax_ppl.bar(x, eval_ppl, color=colors)
    ax_ppl.set_xticks(x)
    ax_ppl.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax_ppl.set_ylabel("eval_perplexity (najlepsza z epok)")
    ax_ppl.set_title("Perplexity walidacyjna z treningu")
    ax_ppl.set_yscale("log")
    for i, val in enumerate(eval_ppl):
        if val == val and val > 0:
            ax_ppl.text(i, val, f"  {val:.1f}", va="center", fontsize=9)

    fig.suptitle("Porównanie treningu modeli", fontsize=12, y=1.02)
    fig.tight_layout()
    out = output_dir / TRAINING_COMPARE_PLOT
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log_print(f"Wykres porównania treningu → {out}")
    return out


def format_training_compare_section(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return ""
    lines = ["=== Metryki treningu (runy checkpointów) ===", ""]
    for s in summaries:
        arch = s.get("compare_architecture") or s.get("architecture")
        logged = float(s.get("train_time_logged_seconds") or 0.0)
        wall = s.get("train_time_wall_seconds")
        lines.append(f"[{arch}] run={s.get('run_id')}")
        lines.append(f"  katalog: {s.get('save_dir')}")
        lines.append(f"  epoki (max w CSV): {s.get('epochs')}")
        if logged > 0:
            lines.append(f"  czas treningu (suma train+eval): {logged:.1f}s")
        if wall is not None:
            lines.append(f"  wall clock ostatniej sesji train.main: {float(wall):.1f}s")
        lines.append(
            f"  perplexity: best_eval={_float_fmt(s.get('eval_perplexity_best'))}, "
            f"final_eval={_float_fmt(s.get('eval_perplexity_final'))}, "
            f"final_train={_float_fmt(s.get('train_perplexity_final'))}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n\n"


def _float_fmt(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if v != v:
        return "nan"
    return f"{v:.2f}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Wykres porównania treningu z katalogów saved_models (run dirs)"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Katalogi runów (metrics.csv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Gdzie zapisać wykres i CSV",
    )
    args = parser.parse_args()
    pairs = [(p.parent.name, p) for p in args.run_dirs]
    summaries = load_training_summaries(pairs)
    write_training_compare_csv(args.output_dir / TRAINING_COMPARE_FILENAME, summaries)
    plot_training_model_compare(summaries, args.output_dir)


if __name__ == "__main__":
    main()
