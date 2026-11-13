"""Zapis metryk epok do CSV + metadanych runu (stdlib csv)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS_COLUMNS = (
    "epoch",
    "train_loss",
    "train_perplexity",
    "eval_loss",
    "eval_perplexity",
    "train_seconds",
    "eval_seconds",
)

METRICS_FILENAME = "metrics.csv"
META_FILENAME = "run_meta.json"
SUMMARY_FILENAME = "model_summary.txt"


def project_root() -> Path:
    from paths import project_root as repo_root

    return repo_root()


class RunMetricsLogger:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.save_dir / METRICS_FILENAME
        self.meta_path = self.save_dir / META_FILENAME

    def write_meta(
        self,
        *,
        architecture: str,
        run_id: str,
        tensorboard_dir: str | None = None,
        hparams: dict[str, Any] | None = None,
        resumed_from: dict[str, Any] | None = None,
    ) -> None:
        if self.meta_path.is_file() and resumed_from is None:
            return
        payload = self.load_meta() if self.meta_path.is_file() else {}
        payload.update(
            {
                "architecture": architecture,
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "tensorboard_dir": tensorboard_dir,
                "hparams": hparams or payload.get("hparams") or {},
            }
        )
        if resumed_from is not None:
            payload["resumed_from"] = resumed_from
        self.meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_perplexity: float,
        eval_loss: float,
        eval_perplexity: float,
        train_seconds: float,
        eval_seconds: float,
    ) -> None:
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": train_perplexity,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        }
        write_header = not self.metrics_path.is_file()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load_meta(self) -> dict[str, Any]:
        if not self.meta_path.is_file():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def write_timing(
        self,
        *,
        train_total_seconds: float,
        epochs: int,
        start_epoch: int,
    ) -> None:
        payload = self.load_meta()
        if not payload:
            payload = {"run_id": self.save_dir.name}
        payload["finished_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        payload["train_total_seconds"] = train_total_seconds
        payload["epochs"] = epochs
        payload["start_epoch"] = start_epoch
        self.meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_model_report(self, report: dict[str, Any], module_tree: str) -> None:
        payload = self.load_meta()
        if not payload:
            payload = {"run_id": self.save_dir.name}
        payload["model"] = report
        self.meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary_path = self.save_dir / SUMMARY_FILENAME
        lines = [
            f"architecture: {report.get('architecture')}",
            f"description: {report.get('description')}",
            f"params: {report.get('params'):,}",
            f"model_bytes: {report.get('model_bytes'):,}",
            f"dtype: {report.get('dtype')}",
            f"weight_tying: {report.get('weight_tying')}",
            "",
            json.dumps(report, indent=2, ensure_ascii=False),
            "",
            "--- module tree ---",
            module_tree,
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_run_metrics(save_dir: Path) -> list[dict[str, Any]]:
    path = Path(save_dir) / METRICS_FILENAME
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_nan(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def summarize_training_run(save_dir: Path) -> dict[str, Any]:
    """Podsumowanie treningu z metrics.csv + run_meta.json (do porównań modeli)."""
    save_dir = Path(save_dir)
    rows = load_run_metrics(save_dir)
    meta = {}
    if (save_dir / META_FILENAME).is_file():
        meta = json.loads((save_dir / META_FILENAME).read_text(encoding="utf-8"))

    architecture = str(meta.get("architecture") or save_dir.parent.name)
    run_id = str(meta.get("run_id") or save_dir.name)

    total_logged = 0.0
    best_eval_ppl = float("inf")
    final_eval_ppl = float("nan")
    final_train_ppl = float("nan")
    max_epoch = 0

    for row in rows:
        epoch = int(_float_or_nan(row.get("epoch")))
        if epoch <= 0:
            continue
        max_epoch = max(max_epoch, epoch)
        total_logged += _float_or_nan(row.get("train_seconds")) + _float_or_nan(row.get("eval_seconds"))
        ep = _float_or_nan(row.get("eval_perplexity"))
        if ep == ep and ep < best_eval_ppl:
            best_eval_ppl = ep

    if rows:
        last = max(rows, key=lambda r: int(_float_or_nan(r.get("epoch"))))
        final_eval_ppl = _float_or_nan(last.get("eval_perplexity"))
        final_train_ppl = _float_or_nan(last.get("train_perplexity"))

    wall_raw = meta.get("train_total_seconds")
    wall: float | None
    try:
        wall = float(wall_raw) if wall_raw is not None else None
    except (TypeError, ValueError):
        wall = None

    return {
        "save_dir": str(save_dir.resolve()),
        "run_id": run_id,
        "architecture": architecture,
        "epochs": max_epoch,
        "train_time_logged_seconds": total_logged,
        "train_time_wall_seconds": wall,
        "eval_perplexity_best": best_eval_ppl if best_eval_ppl != float("inf") else float("nan"),
        "eval_perplexity_final": final_eval_ppl,
        "train_perplexity_final": final_train_ppl,
    }


def discover_metric_runs(models_root: Path) -> list[Path]:
    root = Path(models_root)
    if not root.is_dir():
        return []
    out: list[Path] = []

    def add_run_dir(path: Path) -> None:
        if path.is_dir() and (path / METRICS_FILENAME).is_file():
            out.append(path)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / METRICS_FILENAME).is_file():
            add_run_dir(child)
            continue
        for grand in sorted(child.iterdir()):
            add_run_dir(grand)
    return out
