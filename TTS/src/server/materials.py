"""Material storage: materials/<id>/ with meta.json and artifacts."""

from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tts.infer_Qwen_3_tts import PROJECT_DIR

MATERIALS_DIR = PROJECT_DIR / "materials"

PDF_NAME = "document.pdf"
WAV_NAME = "audio.wav"
VTT_NAME = "subtitles.vtt"
TEXT_NAME = "text.txt"
EXTRACTED_NAME = "extracted.txt"
META_NAME = "meta.json"

_KIND_TO_FILE = {
    "pdf": PDF_NAME,
    "wav": WAV_NAME,
    "vtt": VTT_NAME,
    "extracted": EXTRACTED_NAME,
    "text": TEXT_NAME,
}

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "wav": "audio/wav",
    "vtt": "text/vtt",
    "extracted": "text/plain; charset=utf-8",
    "text": "text/plain; charset=utf-8",
}

_meta_lock = threading.Lock()


def media_type_for_kind(kind: str) -> str:
    return _MEDIA_TYPES[kind]


def artifact_path(material_id: str, kind: str) -> Path:
    filename = _KIND_TO_FILE[kind]
    return MATERIALS_DIR / material_id / filename


def material_dir(material_id: str) -> Path:
    return MATERIALS_DIR / material_id


def chunks_dir(material_id: str) -> Path:
    return material_dir(material_id) / "chunks"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_title(title: str) -> str:
    t = title.strip()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^\w.\-]", "", t, flags=re.ASCII)
    return t.strip("_") or "untitled"


def material_id_exists(material_id: str) -> bool:
    return material_dir(material_id).is_dir() and (material_dir(material_id) / META_NAME).is_file()


def read_meta(material_id: str) -> dict[str, Any] | None:
    path = material_dir(material_id) / META_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(material_id: str, meta: dict[str, Any]) -> None:
    path = material_dir(material_id) / META_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_meta(material_id: str, **fields: Any) -> dict[str, Any]:
    with _meta_lock:
        meta = read_meta(material_id) or {"id": material_id}
        meta.update(fields)
        write_meta(material_id, meta)
        return meta


def delete_material(material_id: str) -> None:
    path = material_dir(material_id)
    if path.is_dir():
        shutil.rmtree(path)


def list_materials() -> list[dict[str, Any]]:
    if not MATERIALS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for entry in sorted(MATERIALS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        meta = read_meta(entry.name)
        if meta is None:
            continue
        out.append(
            {
                "id": meta.get("id", entry.name),
                "title": meta.get("title", entry.name),
                "status": meta.get("status", "unknown"),
                "progress": meta.get("progress"),
                "duration_sec": meta.get("duration_sec"),
                "created_at": meta.get("created_at"),
                "error": meta.get("error"),
                "stop_requested": meta.get("stop_requested"),
                "num_pages": meta.get("num_pages"),
                "chars_total": meta.get("chars_total"),
                "chars_done": meta.get("chars_done"),
                "eta_sec": meta.get("eta_sec"),
                "eta_at": meta.get("eta_at"),
                "sec_per_char": meta.get("sec_per_char"),
            }
        )
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out


def create_uploaded_record(
    *,
    material_id: str,
    title: str,
    num_pages: int,
    model_key: str,
) -> dict[str, Any]:
    meta = {
        "id": material_id,
        "title": title,
        "status": "uploaded",
        "progress": None,
        "speaker": "ryan",
        "language": "English",
        "model_key": model_key,
        "duration_sec": None,
        "error": None,
        "created_at": _now_iso(),
        "num_pages": num_pages,
        "page_from": 1,
        "page_to": num_pages,
        "filters": None,
        "stop_requested": None,
    }
    write_meta(material_id, meta)
    return meta


def materials_needing_resume() -> list[str]:
    if not MATERIALS_DIR.is_dir():
        return []
    ids: list[str] = []
    for entry in MATERIALS_DIR.iterdir():
        if not entry.is_dir():
            continue
        meta = read_meta(entry.name)
        if meta and meta.get("status") in ("queued", "processing"):
            ids.append(entry.name)
    return ids
