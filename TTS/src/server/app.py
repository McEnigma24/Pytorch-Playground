"""FastAPI: materials API + static reader UI."""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import materials as mat
from server.pdf_text import (
    FILTERS,
    extract_text_from_pdf,
    filters_catalog,
    is_subsequence_of,
    pdf_page_count,
)
from server.pdf_fetch import PdfFetchError, fetch_pdf_bytes
from server import worker as wk
from server.worker import enqueue, start_worker
from tts.document_pipeline import DEFAULT_MODEL
from tts.infer_Qwen_3_tts import PROJECT_DIR

WEB_DIR = PROJECT_DIR / "web"

SPEAKERS_DEFAULT = "ryan"
LANGUAGE_DEFAULT = "English"

_DEFAULT_FILTER_IDS = [f.id for f in FILTERS if f.default]


class ExtractBody(BaseModel):
    filters: list[str] = Field(default_factory=lambda: list(_DEFAULT_FILTER_IDS))
    page_from: int = 1
    page_to: int | None = None


class GenerateBody(BaseModel):
    text: str
    speaker: str = SPEAKERS_DEFAULT
    language: str = LANGUAGE_DEFAULT


@asynccontextmanager
async def lifespan(app: FastAPI):
    mat.MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    start_worker()
    yield


app = FastAPI(title="TTS PDF Reader", lifespan=lifespan)


@app.get("/api/filters")
def api_filters() -> list[dict]:
    return filters_catalog()


@app.get("/api/stats")
def api_stats() -> dict:
    from server import processing_stats as pstats

    prior = pstats.sec_per_char_prior()
    return {"sec_per_char_prior": prior, **pstats.read_global_stats()}


@app.get("/api/materials")
def api_list_materials() -> list[dict]:
    items = mat.list_materials()
    for m in items:
        m["active"] = wk.is_active(m["id"])
    return items


@app.get("/api/materials/{material_id}")
def api_get_material(material_id: str):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")
    return meta


@app.get("/api/materials/{material_id}/{kind}")
def api_get_artifact(material_id: str, kind: str):
    if kind not in mat._KIND_TO_FILE:
        raise HTTPException(status_code=404, detail="Unknown kind")

    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")

    if kind in ("wav", "vtt") and meta.get("status") != "ready":
        raise HTTPException(status_code=409, detail=f"Material status: {meta.get('status')}")

    path = mat.artifact_path(material_id, kind)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, media_type=mat.media_type_for_kind(kind), filename=path.name)


@app.post("/api/materials", status_code=201)
async def api_create_material(
    pdf: UploadFile | None = File(None),
    pdf_url: str = Form(default=""),
    title: str = Form(default=""),
):
    url_given = bool(pdf_url.strip())
    file_given = pdf is not None and bool(pdf.filename)
    if url_given == file_given:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: pdf file or pdf_url",
        )

    if file_given:
        if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="pdf file must have .pdf extension")
        display_title = title.strip() or Path(pdf.filename).stem
        source_url: str | None = None
    else:
        display_title = title.strip()
        if not display_title:
            parsed = urlparse(pdf_url.strip())
            path_name = Path(parsed.path).stem
            display_title = path_name or "untitled"
        source_url = pdf_url.strip()

    material_id = mat.sanitize_title(display_title)
    if mat.material_id_exists(material_id):
        raise HTTPException(status_code=409, detail=f"Material id already exists: {material_id}")

    out_dir = mat.material_dir(material_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / mat.PDF_NAME
    try:
        if file_given:
            with pdf_path.open("wb") as f:
                shutil.copyfileobj(pdf.file, f)
        else:
            try:
                data = fetch_pdf_bytes(source_url or "")
            except PdfFetchError as exc:
                mat.delete_material(material_id)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            pdf_path.write_bytes(data)

        num_pages = pdf_page_count(pdf_path)
    except OSError as exc:
        mat.delete_material(material_id)
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc

    mat.create_uploaded_record(
        material_id=material_id,
        title=display_title,
        num_pages=num_pages,
        model_key=DEFAULT_MODEL,
        source_url=source_url,
    )
    return {"id": material_id, "num_pages": num_pages, "status": "uploaded"}


@app.post("/api/materials/{material_id}/extract")
def api_extract(material_id: str, body: ExtractBody):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")
    if meta.get("status") not in ("uploaded", "draft", "paused", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot extract in status {meta.get('status')}",
        )

    pdf_path = mat.material_dir(material_id) / mat.PDF_NAME
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF missing")

    num_pages = int(meta.get("num_pages") or pdf_page_count(pdf_path))
    page_from = max(1, body.page_from)
    page_to = body.page_to if body.page_to is not None else num_pages
    page_to = min(page_to, num_pages)
    if page_from > page_to:
        raise HTTPException(status_code=400, detail="page_from > page_to")

    filter_ids = set(body.filters)
    text, _ = extract_text_from_pdf(
        pdf_path,
        page_from=page_from,
        page_to=page_to,
        filter_ids=filter_ids,
        language=str(meta.get("language") or LANGUAGE_DEFAULT),
    )
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text after extraction/filters")

    out_dir = mat.material_dir(material_id)
    extracted_path = out_dir / mat.EXTRACTED_NAME
    extracted_path.write_text(text + "\n", encoding="utf-8")

    text_path = out_dir / mat.TEXT_NAME
    if text_path.is_file():
        text_path.unlink()

    mat.update_meta(
        material_id,
        status="draft",
        filters=list(filter_ids),
        page_from=page_from,
        page_to=page_to,
        error=None,
        stop_requested=None,
    )

    words = len(text.split())
    return {"text": text, "chars": len(text), "words": words}


@app.post("/api/materials/{material_id}/generate", status_code=202)
def api_generate(material_id: str, body: GenerateBody):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")
    if meta.get("status") not in ("draft", "paused", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot generate in status {meta.get('status')}",
        )

    extracted_path = mat.material_dir(material_id) / mat.EXTRACTED_NAME
    if not extracted_path.is_file():
        raise HTTPException(status_code=409, detail="Run extract first")

    extracted = extracted_path.read_text(encoding="utf-8")
    candidate = body.text.strip()
    if not candidate:
        raise HTTPException(status_code=422, detail="text is empty")
    if not is_subsequence_of(candidate, extracted):
        raise HTTPException(
            status_code=422,
            detail="text must be a subsequence of extracted text (deletions only)",
        )

    text_path = mat.material_dir(material_id) / mat.TEXT_NAME
    text_path.write_text(candidate + "\n", encoding="utf-8")

    from tts.document_pipeline import chunk_text

    chunks = chunk_text(candidate)
    chars_total = sum(len(c) for c in chunks)

    mat.update_meta(
        material_id,
        status="queued",
        progress=None,
        speaker=body.speaker,
        language=body.language,
        error=None,
        stop_requested=None,
        duration_sec=None,
        chars_total=chars_total,
        chars_done=0,
        chunk_count=len(chunks),
        eta_sec=None,
        eta_at=None,
        eta_finish_at=None,
        sec_per_char=None,
        proc_elapsed_sec=None,
        proc_chars=None,
    )
    enqueue(material_id)
    return {"id": material_id, "status": "queued"}


def _clear_chunk_eta_fields() -> dict:
    return {
        "chunk_eta_sec": None,
        "chunk_eta_at": None,
        "chunk_finish_at": None,
        "current_chunk_chars": None,
    }


@app.post("/api/materials/{material_id}/abort")
def api_abort(material_id: str):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")

    status = meta.get("status")
    if status == "paused":
        return {"id": material_id, "status": "paused", "active": False}
    if status not in ("queued", "processing"):
        raise HTTPException(status_code=409, detail=f"Cannot abort status {status}")

    if wk.is_active(material_id):
        mat.update_meta(material_id, stop_requested="abort")
        wk.request_abort(material_id)
        return {
            "id": material_id,
            "status": "processing",
            "stop_requested": "abort",
            "active": True,
        }

    mat.update_meta(
        material_id,
        status="paused",
        stop_requested=None,
        **_clear_chunk_eta_fields(),
    )
    return {"id": material_id, "status": "paused", "active": False}


@app.post("/api/materials/{material_id}/pause")
def api_pause(material_id: str):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")

    status = meta.get("status")
    if status == "queued":
        mat.update_meta(material_id, status="paused", stop_requested=None)
        return {"id": material_id, "status": "paused"}
    if status == "processing":
        if wk.is_active(material_id):
            mat.update_meta(material_id, stop_requested="pause")
            return {
                "id": material_id,
                "status": "processing",
                "stop_requested": "pause",
                "active": True,
            }
        mat.update_meta(material_id, status="paused", stop_requested=None)
        return {"id": material_id, "status": "paused", "active": False}
    raise HTTPException(status_code=409, detail=f"Cannot pause status {status}")


@app.post("/api/materials/{material_id}/resume")
def api_resume(material_id: str):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")

    status = meta.get("status")
    stop = meta.get("stop_requested")

    if status == "processing" and wk.is_active(material_id) and stop == "pause":
        mat.update_meta(material_id, stop_requested=None)
        return {"id": material_id, "status": "processing", "stop_requested": None}

    if status == "processing" and not wk.is_active(material_id):
        text_path = mat.material_dir(material_id) / mat.TEXT_NAME
        if not text_path.is_file():
            raise HTTPException(status_code=409, detail="Missing text.txt — generate first")
        mat.update_meta(
            material_id,
            status="queued",
            error=None,
            stop_requested=None,
        )
        enqueue(material_id)
        return {"id": material_id, "status": "queued"}

    if status not in ("paused", "error"):
        raise HTTPException(status_code=409, detail=f"Cannot resume status {status}")

    text_path = mat.material_dir(material_id) / mat.TEXT_NAME
    if not text_path.is_file():
        raise HTTPException(status_code=409, detail="Missing text.txt — generate first")

    mat.update_meta(
        material_id,
        status="queued",
        error=None,
        stop_requested=None,
    )
    enqueue(material_id)
    return {"id": material_id, "status": "queued"}


@app.delete("/api/materials/{material_id}")
def api_delete(material_id: str):
    meta = mat.read_meta(material_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Material not found")

    status = meta.get("status")
    if status in ("queued", "processing"):
        if wk.is_active(material_id):
            mat.update_meta(material_id, stop_requested="delete")
            return {
                "id": material_id,
                "status": status,
                "stop_requested": "delete",
                "active": True,
            }
        mat.delete_material(material_id)
        return {"id": material_id, "deleted": True, "active": False}

    mat.delete_material(material_id)
    return {"id": material_id, "deleted": True}


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
