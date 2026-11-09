"""Background queue: one thread, lazy-loaded TTS + aligner models."""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timezone
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
_lock = threading.Lock()
_models: tuple[Qwen3TTSModel, Qwen3ForcedAligner] | None = None


def _get_models(model_key: str) -> tuple[Qwen3TTSModel, Qwen3ForcedAligner]:
    global _models
    with _lock:
        if _models is None:
            from tts.infer_Qwen_3_tts import load_forced_aligner, load_qwen_tts_model

            log.info("Loading TTS model %s and forced aligner…", model_key)
            tts = load_qwen_tts_model(model_key)
            aligner = load_forced_aligner()
            _models = (tts, aligner)
        return _models


def _chars_done_from_cache(cache_dir, chunks: list[str]) -> int:
    done = 0
    for idx, chunk in enumerate(chunks):
        wav_path, meta_path = _chunk_cache_paths(cache_dir, idx)
        if _load_cached_chunk(wav_path, meta_path) is not None:
            done += len(chunk)
    return done


def _eta_fields(chars_total: int, chars_done: int, sec_per_char: float | None) -> dict:
    if not sec_per_char or chars_total <= 0:
        return {}
    remaining = max(0, chars_total - chars_done)
    return {
        "eta_sec": round(remaining * sec_per_char, 2),
        "eta_at": datetime.now(timezone.utc).isoformat(),
    }


def _should_stop(material_id: str) -> bool:
    meta = mat.read_meta(material_id)
    if meta is None:
        return True
    return bool(meta.get("stop_requested"))


def _handle_stop(material_id: str) -> None:
    meta = mat.read_meta(material_id)
    if meta is None:
        return
    action = meta.get("stop_requested")
    if action == "delete":
        mat.delete_material(material_id)
        log.info("Material %s deleted (stop during processing)", material_id)
        return
    mat.update_meta(material_id, status="paused", stop_requested=None)
    log.info("Material %s paused", material_id)


def _process_one(material_id: str) -> None:
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

    try:
        text_path = mat.material_dir(material_id) / mat.TEXT_NAME
        if not text_path.is_file():
            raise FileNotFoundError(f"Missing {mat.TEXT_NAME} — run generate first")
        text = text_path.read_text(encoding="utf-8").strip()

        chunks = chunk_text(text)
        chars_total = sum(len(c) for c in chunks)
        cache_dir = mat.chunks_dir(material_id)
        chars_done = _chars_done_from_cache(cache_dir, chunks)
        prior = pstats.sec_per_char_prior()
        sec_per_char = float(prior) if prior else None

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
        }
        start_fields.update(_eta_fields(chars_total, chars_done, sec_per_char))
        mat.update_meta(material_id, **start_fields)

        qwen, aligner = _get_models(model_key)
        out_dir = mat.material_dir(material_id)
        wav_path = out_dir / mat.WAV_NAME
        vtt_path = out_dir / mat.VTT_NAME

        def on_progress(done: int, total: int) -> None:
            mat.update_meta(material_id, progress=f"{done}/{total}")

        def on_chunk_done(
            done: int,
            total: int,
            chunk_chars: int,
            elapsed_sec: float,
            from_cache: bool,
        ) -> None:
            meta = mat.read_meta(material_id)
            if meta is None:
                return
            chars_done_now = int(meta.get("chars_done") or 0) + chunk_chars
            proc_elapsed = float(meta.get("proc_elapsed_sec") or 0.0)
            proc_chars = int(meta.get("proc_chars") or 0)
            sec_rate = meta.get("sec_per_char")

            if not from_cache and elapsed_sec > 0 and chunk_chars > 0:
                proc_elapsed += elapsed_sec
                proc_chars += chunk_chars
                pstats.record_chunk_timing(elapsed_sec, chunk_chars)
                sec_rate = proc_elapsed / proc_chars if proc_chars > 0 else sec_rate

            if sec_rate is None:
                prior_now = pstats.sec_per_char_prior()
                sec_rate = float(prior_now) if prior_now else None

            fields: dict = {
                "chars_done": chars_done_now,
                "progress": f"{done}/{total}",
                "proc_elapsed_sec": proc_elapsed,
                "proc_chars": proc_chars,
            }
            if sec_rate is not None:
                fields["sec_per_char"] = sec_rate
            fields.update(
                _eta_fields(int(meta.get("chars_total") or chars_total), chars_done_now, sec_rate)
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
            cache_dir=cache_dir,
        )

        if mat.read_meta(material_id) is None:
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
    _thread = threading.Thread(target=_worker_loop, name="tts-material-worker", daemon=True)
    _thread.start()
    for material_id in mat.materials_needing_resume():
        enqueue(material_id)


def enqueue(material_id: str) -> None:
    _queue.put(material_id)
