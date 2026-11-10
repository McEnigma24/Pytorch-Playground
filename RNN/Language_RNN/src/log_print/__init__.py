"""Print z prefiksem Δt od poprzedniego logu (synchroniczny, jeden licznik na proces)."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, TextIO

# Odwrócony unix: nowszy run = mniejsza liczba = wyżej przy sortowaniu nazw A→Z.
# Zakładka Scalars w TensorBoard ignoruje nazwę (sortuje po czasie pierwszego eventu).
_UNIX_INV_CEILING = 10**10
_UNIX_INV_WIDTH = 10

_last_log_at: float | None = None


def _format_wall_time(when: datetime | None = None) -> str:
    when = when or datetime.now()
    return when.strftime("%H:%M")


def _format_diff(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:2d}m {secs:2d}s"


def inverted_unix_prefix(when: datetime | None = None) -> int:
    when = when or datetime.now()
    return _UNIX_INV_CEILING - int(when.timestamp())


def tensorboard_run_dir(log_root: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    suffix = when.strftime("%Y%m%d-%H%M%S")
    inv = inverted_unix_prefix(when)
    return os.path.join(log_root, f"{inv:0{_UNIX_INV_WIDTH}d}_{suffix}")


def reset_log_timer() -> None:
    global _last_log_at
    _last_log_at = None


def log_print(
    *args: Any,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
    prefix: bool = True,
) -> None:
    """Domyślnie: prefiks czasu + Δt. prefix=False — bez prefiksu i bez resetu timera (domyślnie flush=False)."""
    if not prefix:
        print(*args, sep=sep, end=end, file=file, flush=flush)
        return

    global _last_log_at
    now_mono = time.perf_counter()
    wall = _format_wall_time()

    if _last_log_at is None:
        diff_part = _format_diff(0)
    else:
        diff_part = _format_diff(now_mono - _last_log_at)

    _last_log_at = now_mono
    stamp = f"{wall} - {diff_part} -"
    print(stamp, *args, sep=sep, end=end, file=file, flush=flush)
