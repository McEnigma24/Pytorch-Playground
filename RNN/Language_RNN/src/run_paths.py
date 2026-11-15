"""Nazewnictwo runów: odwrócony unix + saved_models/{arch}/…"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from log_print import inverted_unix_prefix, log_print
from metrics.run_metrics import (
    META_FILENAME,
    METRICS_FILENAME,
    RESUME_LIST_FILENAME,
    SUMMARY_FILENAME,
    load_run_metrics,
)

_RUN_DIR_SEP = "___"
_RESUME_HEADER = (
    "# started_at\tsegment_wall_seconds\tprevious_run_dir\n"
    "# Każdy resume dopisuje zamkniętą próbę; suma segmentów = czas przed bieżącą sesją.\n"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize_run_tag(tag: str) -> str:
    cleaned = tag.strip().replace(" ", "-")
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in cleaned).strip("-")


def run_dir_name(run_tag: str | None = None, when: datetime | None = None) -> str:
    when = when or datetime.now()
    prefix = f"{inverted_unix_prefix(when):010d}"
    date_part = when.strftime("%Y-%m-%d_%H-%M")
    if run_tag and str(run_tag).strip():
        safe = _sanitize_run_tag(str(run_tag))
        if safe:
            return f"{prefix}{_RUN_DIR_SEP}{safe}{_RUN_DIR_SEP}{date_part}"
    return f"{prefix}{_RUN_DIR_SEP}{date_part}"


def run_tag_from_dir_name(dir_name: str) -> str | None:
    parts = dir_name.split(_RUN_DIR_SEP)
    if len(parts) >= 3:
        return parts[1]
    return None


def models_root() -> Path:
    return project_root() / "saved_models"


def evaluations_root() -> Path:
    return project_root() / "model_evaluations"


def resolve_save_dir(
    architecture: str,
    when: datetime | None = None,
    run_tag: str | None = None,
) -> Path:
    env = os.environ.get("LANGUAGE_RNN_SAVE_DIR", "").strip()
    if env:
        root = Path(env)
    else:
        when = when or datetime.now()
        root = models_root() / architecture / run_dir_name(run_tag, when)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return root


def resolve_evaluation_dir(run_tag: str | None = None, when: datetime | None = None) -> Path:
    root = evaluations_root() / run_dir_name(run_tag, when)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_has_artifacts(run_dir: Path, architecture: str) -> bool:
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir() and any(ckpt_dir.glob("epoch_*.pt")):
        return True
    if any(run_dir.glob(f"{architecture}_best_eval.pt")):
        return True
    return (run_dir / METRICS_FILENAME).is_file()


def _legacy_flat_run_dirs(architecture: str) -> list[Path]:
    """Stary układ: saved_models/{arch}-YYYY.MM.DD-HHMMSS[-tag]."""
    root = models_root()
    if not root.is_dir():
        return []
    prefix = f"{architecture}-"
    return [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]


def find_latest_run_dir(architecture: str, *, root: Path | None = None) -> Path:
    arch_root = (root or models_root()) / architecture
    nested: list[Path] = []
    if arch_root.is_dir():
        for child in arch_root.iterdir():
            if child.is_dir() and _run_has_artifacts(child, architecture):
                nested.append(child)

    legacy = [p for p in _legacy_flat_run_dirs(architecture) if _run_has_artifacts(p, architecture)]
    candidates = nested + legacy
    if not candidates:
        raise FileNotFoundError(
            f"Brak runów dla {architecture!r} w {models_root()} — najpierw wytrenuj (train.main …)"
        )

    def sort_key(p: Path) -> tuple[int, str, float]:
        if p.parent.name == architecture:
            return (0, p.name, 0.0)
        return (1, p.name, -p.stat().st_mtime)

    return min(candidates, key=sort_key)


def latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Brak checkpoints/ w {run_dir}")
    paths = list(ckpt_dir.glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"Brak epoch_*.pt w {ckpt_dir}")

    def epoch_num(path: Path) -> int:
        stem = path.stem  # epoch_0005
        try:
            return int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(paths, key=epoch_num)


def _float_or_zero(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v else 0.0


def _run_started_at(run_dir: Path) -> str:
    meta_path = run_dir / META_FILENAME
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        started = meta.get("started_at")
        if started:
            return str(started)
    when = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).astimezone()
    return when.isoformat(timespec="seconds")


def _run_segment_wall_seconds(run_dir: Path) -> float:
    """Czas zamykanej próby: suma train+eval z CSV albo ostatni segment z meta."""
    meta_path = run_dir / META_FILENAME
    list_path = run_dir / RESUME_LIST_FILENAME
    rows = load_run_metrics(run_dir)
    if rows:
        wall = sum(
            _float_or_zero(r.get("train_seconds")) + _float_or_zero(r.get("eval_seconds")) for r in rows
        )
        if wall > 0:
            return wall
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("train_segment_seconds")
        if raw is not None:
            return float(raw)
        raw_total = meta.get("train_total_seconds")
        prior = meta.get("train_prior_seconds")
        if raw_total is not None and prior is not None:
            return max(0.0, float(raw_total) - float(prior))
        if raw_total is not None and not list_path.is_file():
            return float(raw_total)
    return 0.0


def _read_resume_list(path: Path) -> tuple[list[str], float]:
    if not path.is_file():
        return [], 0.0
    body_lines: list[str] = []
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        body_lines.append(line.rstrip("\n"))
        total += _float_or_zero(parts[1])
    return body_lines, total


def _write_resume_list(path: Path, lines: list[str]) -> None:
    path.write_text(_RESUME_HEADER + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def resume_run_relocate(
    source_run_dir: Path,
    architecture: str,
    run_tag: str | None = None,
    *,
    relocate: bool = True,
) -> tuple[Path, float, Path]:
    """Dopisz resume_list; opcjonalnie przenieś run na świeżą datę w nazwie."""
    _ = architecture
    source_run_dir = Path(source_run_dir).resolve()
    tag = run_tag or run_tag_from_dir_name(source_run_dir.name)
    dest = (
        source_run_dir.parent / run_dir_name(tag, datetime.now())
        if relocate
        else source_run_dir
    )

    started_at = _run_started_at(source_run_dir)
    segment_wall = _run_segment_wall_seconds(source_run_dir)
    archived_dir_name = source_run_dir.name

    list_path = source_run_dir / RESUME_LIST_FILENAME
    resume_lines, old_prior = _read_resume_list(list_path)
    resume_lines.append(f"{started_at}\t{segment_wall:.3f}\t{archived_dir_name}")
    prior_cumulative = old_prior + segment_wall

    if relocate and dest != source_run_dir:
        if dest.exists():
            raise FileExistsError(f"Resume: katalog docelowy już istnieje: {dest}")
        shutil.move(str(source_run_dir), str(dest))
        log_print(f"Resume: przeniesiono run → {dest.name}")
        save_dir = dest
    else:
        save_dir = source_run_dir
        if relocate:
            log_print("Resume: kontynuacja w tym samym katalogu (bez zmiany nazwy)")
        else:
            log_print("Resume: LANGUAGE_RNN_SAVE_DIR — bez zmiany nazwy katalogu")

    _write_resume_list(save_dir / RESUME_LIST_FILENAME, resume_lines)
    log_print(
        f"Resume: dopisano próbę ({started_at}, {segment_wall:.1f}s wall) — "
        f"łącznie wcześniej {prior_cumulative:.1f}s"
    )

    ckpt = latest_checkpoint(save_dir)
    log_print(f"Resume: checkpoint {ckpt.name}")
    return save_dir, prior_cumulative, ckpt


def resolve_source_run_dir(architecture: str, resume: str) -> Path:
    resume = resume.strip()
    if resume == "latest":
        return find_latest_run_dir(architecture)
    path = Path(resume)
    if not path.is_dir():
        alt = project_root() / resume
        if alt.is_dir():
            path = alt
        else:
            raise FileNotFoundError(f"Nie znaleziono katalogu runu: {resume}")
    return path.resolve()


def prior_train_seconds_from_resume_list(run_dir: Path) -> float:
    _, total = _read_resume_list(run_dir / RESUME_LIST_FILENAME)
    return total


def resolve_resume_from_env_or_arg(resume: str | None) -> str | None:
    if resume is not None and str(resume).strip():
        return str(resume).strip()
    env = os.environ.get("LANGUAGE_RNN_RESUME", "").strip()
    return env or None