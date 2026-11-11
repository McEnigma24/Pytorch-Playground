"""Background queue: one thread, lazy-loaded TTS + aligner models."""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue

from qwen_asr import Qwen3ForcedAligner
from qwen_tts import Qwen3TTSModel

from server import materials as mat
from server import processing_stats as pstats
from tts.document_pipeline import (
    DEFAULT_MODEL,
    StopRequested,
    TTS_TEMPERATURE,
    _chunk_cache_paths,
    _load_cached_chunk,
    chunk_text,
    instruct_for_language,
    process_document,
)

log = logging.getLogger(__name__)

_queue: Queue[str] = Queue()
_thread: threading.Thread | None = None
_models_lock = threading.Lock()
_active_lock = threading.Lock()
_models: tuple[Qwen3TTSModel, Qwen3ForcedAligner] | None = None
_active_id: str | None = None
_abort_event = threading.Event()


def is_active(material_id: str) -> bool:
    with _active_lock:
        return _active_id == material_id


def request_abort(material_id: str) -> bool:
    """Signal in-flight GPU generation to stop; only when this material is active."""
    with _active_lock:
        if _active_id != material_id:
            return False
    _abort_event.set()
    return True


def _get_models(model_key: str) -> tuple[Qwen3TTSModel, Qwen3ForcedAligner]:
    global _models
    with _models_lock:
        if _models is not None:
            return _models
    from tts.infer_Qwen_3_tts import load_forced_aligner, load_qwen_tts_model

    log.info("Loading TTS model %s and forced aligner…", model_key)
    tts = load_qwen_tts_model(model_key)
    aligner = load_forced_aligner()
    with _models_lock:
        if _models is None:
            _models = (tts, aligner)
        return _models


def _chars_done_from_cache(cache_dir, chunks: list[str]) -> int:
    done = 0
    for idx, chunk in enumerate(chunks):
        wav_path, meta_path = _chunk_cache_paths(cache_dir, idx)
        if _load_cached_chunk(wav_path, meta_path, chunk) is not None:
            done += len(chunk)
    return done


def _eta_fields(chars_total: int, chars_done: int, sec_per_char: float | None) -> dict:
    if not sec_per_char or chars_total <= 0:
        return {}
    remaining = max(0, chars_total - chars_done)
    eta_sec = round(remaining * sec_per_char, 2)
    now = datetime.now(timezone.utc)
    finish = now + timedelta(seconds=eta_sec)
    return {
        "eta_sec": eta_sec,
        "eta_at": now.isoformat(),
        "eta_finish_at": finish.isoformat(),
    }


def _chunk_eta_fields(chunk_chars: int, sec_per_char: float | None) -> dict:
    if not sec_per_char or chunk_chars <= 0:
        return {
            "chunk_eta_sec": None,
            "chunk_eta_at": None,
            "chunk_finish_at": None,
            "current_chunk_chars": chunk_chars if chunk_chars > 0 else None,
        }
    eta_sec = round(chunk_chars * sec_per_char, 2)
    now = datetime.now(timezone.utc)
    finish = now + timedelta(seconds=eta_sec)
    return {
        "chunk_eta_sec": eta_sec,
        "chunk_eta_at": now.isoformat(),
        "chunk_finish_at": finish.isoformat(),
        "current_chunk_chars": chunk_chars,
    }


def recover_orphans() -> list[str]:
    """Fix stale queued/processing materials; return ids still queued for enqueue."""
    to_enqueue: list[str] = []
    if not mat.MATERIALS_DIR.is_dir():
        return to_enqueue
    for entry in mat.MATERIALS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        meta = mat.read_meta(entry.name)
        if meta is None:
            continue
        status = meta.get("status")
        if status not in ("queued", "processing"):
            continue
        stop = meta.get("stop_requested")
        material_id = entry.name
        if stop == "delete":
            mat.delete_material(material_id)
            log.info("Orphan %s deleted (stop_requested=delete)", material_id)
            continue
        if stop == "abort":
            mat.update_meta(material_id, status="paused", stop_requested=None)
            log.info("Orphan %s -> paused (abort)", material_id)
            continue
        if status == "processing" or stop == "pause":
            mat.update_meta(material_id, status="paused", stop_requested=None)
            log.info("Orphan %s -> paused", material_id)
            continue
        if status == "queued":
            to_enqueue.append(material_id)
    return to_enqueue


def _measured_sec_per_char(meta: dict) -> float | None:
    proc_chars = int(meta.get("proc_chars") or 0)
    if proc_chars <= 0:
        return None
    proc_elapsed = float(meta.get("proc_elapsed_sec") or 0.0)
    if proc_elapsed <= 0:
        return None
    return proc_elapsed / proc_chars


def _should_stop(material_id: str) -> bool:
    meta = mat.read_meta(material_id)
    if meta is None:
        return True
    if meta.get("status") != "processing":
        return True
    return bool(meta.get("stop_requested"))


def _handle_stop(material_id: str) -> None:
    meta = mat.read_meta(material_id)
    if meta is None:
        return
    if meta.get("status") == "draft":
        mat.update_meta(material_id, stop_requested=None)
        return
    action = meta.get("stop_requested")
    if action == "delete":
        mat.delete_material(material_id)
        log.info("Material %s deleted (stop during processing)", material_id)
        return
    if action == "abort":
        mat.update_meta(
            material_id,
            status="paused",
            stop_requested=None,
            chunk_eta_sec=None,
            chunk_eta_at=None,
            chunk_finish_at=None,
            current_chunk_chars=None,
        )
        log.info("Material %s stopped immediately -> paused", material_id)
        return
    mat.update_meta(material_id, status="paused", stop_requested=None)
    log.info("Material %s paused", material_id)


def _process_one(material_id: str) -> None:
    global _active_id
    meta = mat.read_meta(material_id)
    if meta is None:
        log.warning("No meta for %s", material_id)
        return
    if meta.get("status") != "queued":
        log.info("Skip %s (status=%s)", material_id, meta.get("status"))
        return

    model_key = meta.get("model_key") or DEFAULT_MODEL
    speaker = meta.get("speaker") or "ryan"
    language = meta.get("language") or "English"

    with _active_lock:
        _active_id = material_id
    _abort_event.clear()
    try:
        text_path = mat.material_dir(material_id) / mat.TEXT_NAME
        if not text_path.is_file():
            raise FileNotFoundError(f"Missing {mat.TEXT_NAME} — run generate first")
        text = text_path.read_text(encoding="utf-8").strip()

        chunks = chunk_text(text)
        chars_total = sum(len(c) for c in chunks)
        cache_dir = mat.chunks_dir(material_id)
        chars_done = _chars_done_from_cache(cache_dir, chunks)

        start_fields: dict = {
            "status": "processing",
            "progress": "0/?",
            "error": None,
            "stop_requested": None,
            "chars_total": chars_total,
            "chars_done": chars_done,
            "chunk_count": len(chunks),
            "instruct": instruct_for_language(language),
            "temperature": TTS_TEMPERATURE,
            "eta_sec": None,
            "eta_at": None,
            "eta_finish_at": None,
            "chunk_eta_sec": None,
            "chunk_eta_at": None,
            "chunk_finish_at": None,
            "current_chunk_chars": None,
        }
        mat.update_meta(material_id, **start_fields)

        qwen, aligner = _get_models(model_key)
        if _should_stop(material_id):
            raise StopRequested()

        out_dir = mat.material_dir(material_id)
        wav_path = out_dir / mat.WAV_NAME
        vtt_path = out_dir / mat.VTT_NAME

        def on_progress(done: int, total: int) -> None:
            meta_now = mat.read_meta(material_id)
            if meta_now is None or meta_now.get("status") != "processing":
                return
            sec_rate = _measured_sec_per_char(meta_now) if meta_now else None
            chunk_idx = done - 1
            chunk_len = len(chunks[chunk_idx]) if 0 <= chunk_idx < len(chunks) else 0
            fields: dict = {"progress": f"{done}/{total}"}
            fields.update(_chunk_eta_fields(chunk_len, sec_rate))
            mat.update_meta(material_id, **fields)

        def on_chunk_done(
            done: int,
            total: int,
            chunk_chars: int,
            elapsed_sec: float,
            from_cache: bool,
        ) -> None:
            meta = mat.read_meta(material_id)
            if meta is None or meta.get("status") != "processing":
                return
            chars_done_now = int(meta.get("chars_done") or 0) + chunk_chars
            proc_elapsed = float(meta.get("proc_elapsed_sec") or 0.0)
            proc_chars = int(meta.get("proc_chars") or 0)

            if not from_cache and elapsed_sec > 0 and chunk_chars > 0:
                proc_elapsed += elapsed_sec
                proc_chars += chunk_chars
                pstats.record_chunk_timing(elapsed_sec, chunk_chars)

            sec_rate = _measured_sec_per_char(
                {
                    "proc_elapsed_sec": proc_elapsed,
                    "proc_chars": proc_chars,
                }
            )

            fields: dict = {
                "chars_done": chars_done_now,
                "progress": f"{done}/{total}",
                "proc_elapsed_sec": proc_elapsed,
                "proc_chars": proc_chars,
            }
            if sec_rate is not None:
                fields["sec_per_char"] = sec_rate
                fields.update(
                    _eta_fields(
                        int(meta.get("chars_total") or chars_total),
                        chars_done_now,
                        sec_rate,
                    )
                )
            mat.update_meta(material_id, **fields)

        result = process_document(
            text,
            wav_path,
            vtt_path,
            model_key=model_key,
            speaker=speaker,
            language=language,
            qwen_model=qwen,
            forced_aligner=aligner,
            on_progress=on_progress,
            on_chunk_done=on_chunk_done,
            should_stop=lambda: _should_stop(material_id),
            abort_now=lambda: _abort_event.is_set(),
            cache_dir=cache_dir,
        )

        meta_after = mat.read_meta(material_id)
        if meta_after is None:
            return
        if meta_after.get("status") != "processing":
            return

        mat.update_meta(
            material_id,
            status="ready",
            progress=f"{result.chunk_count}/{result.chunk_count}",
            duration_sec=round(result.duration_sec, 3),
            chars_done=chars_total,
            chars_total=chars_total,
            eta_sec=0,
            eta_at=datetime.now(timezone.utc).isoformat(),
            eta_finish_at=datetime.now(timezone.utc).isoformat(),
            chunk_eta_sec=None,
            chunk_eta_at=None,
            chunk_finish_at=None,
            current_chunk_chars=None,
            error=None,
            stop_requested=None,
        )
        log.info("Material %s ready (%.1f s)", material_id, result.duration_sec)

    except StopRequested:
        _handle_stop(material_id)
    except Exception as exc:
        if mat.read_meta(material_id) is None:
            return
        log.exception("Material %s failed", material_id)
        mat.update_meta(
            material_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            stop_requested=None,
        )
    finally:
        _abort_event.clear()
        with _active_lock:
            if _active_id == material_id:
                _active_id = None


def _worker_loop() -> None:
    while True:
        try:
            material_id = _queue.get(timeout=1.0)
        except Empty:
            continue
        try:
            _process_one(material_id)
        finally:
            _queue.task_done()


def start_worker() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    to_enqueue = recover_orphans()
    _thread = threading.Thread(target=_worker_loop, name="tts-material-worker", daemon=True)
    _thread.start()
    for material_id in to_enqueue:
        enqueue(material_id)


def enqueue(material_id: str) -> None:
    _queue.put(material_id)
