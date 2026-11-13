"""Nazewnictwo runów: odwrócony unix + saved_models/{arch}/…"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from log_print import inverted_unix_prefix, log_print
from metrics.run_metrics import META_FILENAME, METRICS_FILENAME, SUMMARY_FILENAME

_RUN_DIR_SEP = "___"


def project_root() -> Path:
    # .../AE/src/paths/run_paths.py → repo root
    return Path(__file__).resolve().parents[2]


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


def models_root() -> Path:
    return project_root() / "saved_models"


def evaluations_root() -> Path:
    return project_root() / "model_evaluations"


def resolve_save_dir(
    architecture: str,
    when: datetime | None = None,
    run_tag: str | None = None,
) -> Path:
    env = os.environ.get("AE_SAVE_DIR", "").strip()
    if not env:
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


def prepare_resumed_run(source_run_dir: Path, new_save_dir: Path, architecture: str) -> Path:
    source_run_dir = Path(source_run_dir).resolve()
    new_save_dir = Path(new_save_dir).resolve()
    if source_run_dir == new_save_dir:
        raise ValueError("Źródłowy i docelowy katalog runu muszą się różnić")

    new_save_dir.mkdir(parents=True, exist_ok=True)
    (new_save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for name in (METRICS_FILENAME, META_FILENAME, SUMMARY_FILENAME):
        src = source_run_dir / name
        if src.is_file():
            shutil.copy2(src, new_save_dir / name)

    for pt in source_run_dir.glob(f"{architecture}_best_eval.pt"):
        shutil.copy2(pt, new_save_dir / pt.name)

    ckpt = latest_checkpoint(source_run_dir)
    dest_ckpt = new_save_dir / "checkpoints" / ckpt.name
    shutil.copy2(ckpt, dest_ckpt)
    log_print(f"Resume: skopiowano {ckpt.name} + artefakty runu z {source_run_dir}")
    return dest_ckpt


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


def resolve_resume_from_env_or_arg(resume: str | None) -> str | None:
    if resume is not None and str(resume).strip():
        return str(resume).strip()
    env = os.environ.get("AE_RESUME", "").strip()
    if not env:
        env = os.environ.get("LANGUAGE_RNN_RESUME", "").strip()
    return env or None
