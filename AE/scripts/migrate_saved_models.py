#!/usr/bin/env python3
"""Migracja saved_models/ i model_evaluations/ do nowego schematu nazw (dry-run domyślnie)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from log_print import log_print  # noqa: E402
from metrics.run_metrics import META_FILENAME  # noqa: E402
from run_paths import evaluations_root, models_root, project_root, run_dir_name  # noqa: E402

OLD_SAVE_RE = re.compile(
    r"^(?P<arch>[a-zA-Z0-9_]+)-(?P<date>\d{4}\.\d{2}\.\d{2}-\d{6})(?:-(?P<tag>.+))?$"
)
OLD_EVAL_RE = re.compile(r"^(?P<date>\d{4}\.\d{2}\.\d{2}-\d{6})(?:-(?P<tag>.+))?$")
RUN_DIR_NEW_RE = re.compile(r"^\d{10}___")


def _parse_old_datetime(date_part: str) -> datetime:
    return datetime.strptime(date_part, "%Y.%m.%d-%H%M%S")


def plan_save_migration() -> list[tuple[Path, Path, str]]:
    root = models_root()
    if not root.is_dir():
        return []
    moves: list[tuple[Path, Path, str]] = []
    legacy = root / "_legacy"
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "_legacy":
            continue
        if RUN_DIR_NEW_RE.match(child.name):
            continue
        m = OLD_SAVE_RE.match(child.name)
        if not m:
            if any(child.glob(f"*/{META_FILENAME}")) or any(child.glob("*/metrics.csv")):
                continue
            dest = legacy / child.name
            moves.append((child, dest, "legacy (nieparsowalna nazwa saved_models)"))
            continue
        arch = m.group("arch")
        when = _parse_old_datetime(m.group("date"))
        tag = m.group("tag")
        new_name = run_dir_name(tag, when)
        dest = root / arch / new_name
        moves.append((child, dest, f"saved_models → {arch}/{new_name}"))
    return moves


def plan_eval_migration() -> list[tuple[Path, Path, str]]:
    root = evaluations_root()
    if not root.is_dir():
        return []
    moves: list[tuple[Path, Path, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if RUN_DIR_NEW_RE.match(child.name):
            continue
        m = OLD_EVAL_RE.match(child.name)
        if not m:
            moves.append((child, root / "_legacy" / child.name, "legacy (nieparsowalna nazwa eval)"))
            continue
        when = _parse_old_datetime(m.group("date"))
        tag = m.group("tag")
        new_name = run_dir_name(tag, when)
        dest = root / new_name
        empty = not any(child.iterdir())
        note = " (pusty)" if empty else ""
        moves.append((child, dest, f"model_evaluations → {new_name}{note}"))
    return moves


def _update_run_meta(run_dir: Path, new_run_id: str) -> None:
    meta_path = run_dir / META_FILENAME
    if not meta_path.is_file():
        return
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["run_id"] = new_run_id
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply_moves(moves: list[tuple[Path, Path, str]], *, apply: bool) -> None:
    for src, dest, label in moves:
        if dest.exists():
            log_print(f"KOLIZJA: {src} → {dest} ({label}) — pomijam")
            continue
        log_print(f"{'MOVE' if apply else 'PLAN'}: {src.name} → {dest}  [{label}]")
        if not apply:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        if (dest / META_FILENAME).is_file():
            _update_run_meta(dest, dest.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migracja nazw katalogów runów")
    parser.add_argument("--apply", action="store_true", help="Wykonaj przeniesienia (domyślnie dry-run)")
    args = parser.parse_args()

    log_print(f"Katalog projektu: {project_root()}")
    save_moves = plan_save_migration()
    eval_moves = plan_eval_migration()
    all_moves = save_moves + eval_moves
    if not all_moves:
        log_print("Brak katalogów do migracji.")
        return
    apply_moves(all_moves, apply=args.apply)
    log_print(f"{'Zastosowano' if args.apply else 'Zaplanowano'} {len(all_moves)} przeniesień.")


if __name__ == "__main__":
    main()
