"""Background queue: one thread, lazy-loaded TTS + aligner models."""

from __future__ import annotations

import logging
import threading
import traceback
from queue import Empty, Queue

from qwen_asr import Qwen3ForcedAligner
from qwen_tts import Qwen3TTSModel

from server import materials as mat
from server.pdf_text import extract_text_from_pdf
from tts.document_pipeline import DEFAULT_MODEL, process_document

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


def _resolve_text(material_id: str) -> str:
    text_path = mat.material_dir(material_id) / mat.TEXT_NAME
    if text_path.is_file():
        return text_path.read_text(encoding="utf-8").strip()
    pdf_path = mat.material_dir(material_id) / mat.PDF_NAME
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing PDF for material {material_id}")
    return extract_text_from_pdf(pdf_path)


def _process_one(material_id: str) -> None:
    meta = mat.read_meta(material_id)
    if meta is None:
        log.warning("No meta for %s", material_id)
        return

    model_key = meta.get("model_key") or DEFAULT_MODEL
    speaker = meta.get("speaker") or "ryan"
    language = meta.get("language") or "English"

    mat.update_meta(material_id, status="processing", progress="0/?", error=None)

    try:
        text = _resolve_text(material_id)
        text_path = mat.material_dir(material_id) / mat.TEXT_NAME
        text_path.write_text(text + "\n", encoding="utf-8")

        qwen, aligner = _get_models(model_key)
        out_dir = mat.material_dir(material_id)
        wav_path = out_dir / mat.WAV_NAME
        vtt_path = out_dir / mat.VTT_NAME

        def on_progress(done: int, total: int) -> None:
            mat.update_meta(material_id, progress=f"{done}/{total}")

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
        )

        mat.update_meta(
            material_id,
            status="ready",
            progress=f"{result.chunk_count}/{result.chunk_count}",
            duration_sec=round(result.duration_sec, 3),
            error=None,
        )
        
        log.info("Material %s ready (%.1f s)", material_id, result.duration_sec)

    except Exception as exc:
        log.exception("Material %s failed", material_id)
        mat.update_meta(
            material_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
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
