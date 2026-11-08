"""FastAPI: materials API + static reader UI."""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server import materials as mat
from server.worker import enqueue, start_worker
from tts.document_pipeline import DEFAULT_MODEL
from tts.infer_Qwen_3_tts import PROJECT_DIR

WEB_DIR = PROJECT_DIR / "web"

SPEAKERS_DEFAULT = "ryan"
LANGUAGE_DEFAULT = "English"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mat.MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    start_worker()
    yield


app = FastAPI(title="TTS PDF Reader", lifespan=lifespan)


@app.get("/api/materials")
def api_list_materials() -> list[dict]:
    return mat.list_materials()


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


@app.post("/api/materials", status_code=202)
async def api_create_material(
    pdf: UploadFile = File(...),
    text: str | None = Form(default=None),
    title: str | None = Form(default=None),
    speaker: str = Form(default=SPEAKERS_DEFAULT),
    language: str = Form(default=LANGUAGE_DEFAULT),
    model_key: str = Form(default=DEFAULT_MODEL),
):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="pdf file required")

    display_title = (title or pdf.filename or "Untitled").removesuffix(".pdf")
    material_id = mat.new_material_id(display_title)
    out_dir = mat.material_dir(material_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / mat.PDF_NAME
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    if text is not None and text.strip():
        (out_dir / mat.TEXT_NAME).write_text(text.strip() + "\n", encoding="utf-8")

    mat.create_material_record(
        material_id=material_id,
        title=display_title,
        speaker=speaker,
        language=language,
        model_key=model_key,
    )
    enqueue(material_id)
    return {"id": material_id, "status": "queued"}


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
