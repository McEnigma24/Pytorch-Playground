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

# Szerokość jak "99m 99s " — kolumny m/s wyrównane (2 cyfry przed m i przed s).
_DIFF_PREFIX_WIDTH = 8
_DIFF_PREFIX_EMPTY = " " * _DIFF_PREFIX_WIDTH

_last_log_at: float | None = None


def _format_diff(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:2d}m {secs:2d}s "


def tensorboard_run_dir(log_root: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    suffix = when.strftime("%Y%m%d-%H%M%S")
    inv = _UNIX_INV_CEILING - int(when.timestamp())
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
) -> None:
    global _last_log_at
    now = time.perf_counter()

    if _last_log_at is None:
        prefix = _DIFF_PREFIX_EMPTY
    else:
        prefix = _format_diff(now - _last_log_at)

    _last_log_at = now
    print(prefix, *args, sep=sep, end=end, file=file, flush=flush)
