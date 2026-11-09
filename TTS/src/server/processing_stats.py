"""Global EMA of GPU seconds per text character for ETA estimates."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from server.materials import MATERIALS_DIR

STATS_PATH = MATERIALS_DIR / "_stats.json"
EMA_ALPHA = 0.25
_lock = threading.Lock()


def _read_unlocked() -> dict:
    if not STATS_PATH.is_file():
        return {}
    try:
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_global_stats() -> dict:
    with _lock:
        return _read_unlocked()


def sec_per_char_prior() -> float | None:
    val = read_global_stats().get("sec_per_char_ema")
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return None


def record_chunk_timing(elapsed_sec: float, chunk_chars: int) -> float:
    """Update EMA; return new per-material sec/char for this chunk."""
    if chunk_chars <= 0 or elapsed_sec <= 0:
        return elapsed_sec / max(chunk_chars, 1)
    chunk_rate = elapsed_sec / chunk_chars
    with _lock:
        data = _read_unlocked()
        prev = data.get("sec_per_char_ema")
        if isinstance(prev, (int, float)) and prev > 0:
            ema = EMA_ALPHA * chunk_rate + (1.0 - EMA_ALPHA) * float(prev)
        else:
            ema = chunk_rate
        data["sec_per_char_ema"] = ema
        data["sample_count"] = int(data.get("sample_count", 0)) + 1
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return chunk_rate
