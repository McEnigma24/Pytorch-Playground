"""Multi-line wykresy metryk z wielu runów (saved_models/*/metrics.csv)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from metrics.run_metrics import METRICS_COLUMNS, discover_metric_runs, load_run_metrics, project_root
from log_print import log_print


def build_long_dataframe(run_dirs: list[Path]):
    frames = []
    value_cols = [c for c in METRICS_COLUMNS if c != "epoch"]
    for run_dir in run_dirs:
        rows = load_run_metrics(run_dir)
        if not rows:
            continue
        meta_path = run_dir / "run_meta.json"
        architecture = run_dir.name.split("-", 1)[0]
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            architecture = meta.get("architecture", architecture)
        run_label = f"{architecture} ({run_dir.name})"
        df = pd.DataFrame(rows)
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["run_label"] = run_label
        df["architecture"] = architecture
        df["run_dir"] = str(run_dir)
        frames.append(df)
    if not frames:
        return None
    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(
        id_vars=["epoch", "run_label", "architecture", "run_dir"],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    return wide, long


def plot_metrics(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    export_comparison_csv: bool = True,
) -> list[Path]:
    built = build_long_dataframe(run_dirs)
    if built is None:
        raise FileNotFoundError(
            "Brak metrics.csv w podanych runach — najpierw wytrenuj (train.main)"
        )
    wide, long = built
    output_dir.mkdir(parents=True, exist_ok=True)

    if export_comparison_csv:
        cmp_path = output_dir / "comparison_long.csv"
        long.to_csv(cmp_path, index=False)
        log_print(f"Eksport → {cmp_path}")

    saved: list[Path] = []
    sns.set_theme(style="whitegrid")
    for metric in [c for c in METRICS_COLUMNS if c != "epoch"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        subset = long[long["metric"] == metric]
        sns.lineplot(data=subset, x="epoch", y="value", hue="run_label", marker="o", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{metric}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        saved.append(out)
        log_print(f"Wykres → {out}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Porównanie metryk treningu (multi-line)")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Katalog saved_models (domyślnie project saved_models/)",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=None,
        help="Konkretne katalogi runów (zamiast auto-discovery)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Katalog na PNG (domyślnie runs/plots/)",
    )
    args = parser.parse_args()

    models_root = args.models_root or project_root() / "saved_models"
    if args.runs:
        run_dirs = [p for p in args.runs if (Path(p) / "metrics.csv").is_file()]
    else:
        run_dirs = discover_metric_runs(models_root)

    if not run_dirs:
        raise SystemExit(f"Brak runów z metrics.csv w {models_root}")

    log_root = os.environ.get("TENSORBOARD_LOGDIR", str(project_root() / "runs"))
    output_dir = args.output or Path(log_root) / "plots"
    plot_metrics(run_dirs, output_dir)


if __name__ == "__main__":
    main()
