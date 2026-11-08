"""Material storage: materials/<id>/ with meta.json and artifacts."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tts.infer_Qwen_3_tts import PROJECT_DIR

MATERIALS_DIR = PROJECT_DIR / "materials"

PDF_NAME = "document.pdf"
WAV_NAME = "audio.wav"
VTT_NAME = "subtitles.vtt"
TEXT_NAME = "text.txt"
META_NAME = "meta.json"

_KIND_TO_FILE = {
    "pdf": PDF_NAME,
    "wav": WAV_NAME,
    "vtt": VTT_NAME,
}

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "wav": "audio/wav",
    "vtt": "text/vtt",
}


def media_type_for_kind(kind: str) -> str:
    return _MEDIA_TYPES[kind]


def artifact_path(material_id: str, kind: str) -> Path:
    filename = _KIND_TO_FILE[kind]
    return MATERIALS_DIR / material_id / filename


def material_dir(material_id: str) -> Path:
    return MATERIALS_DIR / material_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_material_id(title: str | None = None) -> str:
    slug = ""
    if title:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    suffix = uuid.uuid4().hex[:8]
    if slug:
        return f"{slug}-{suffix}"
    return suffix


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
    meta = read_meta(material_id) or {"id": material_id}
    meta.update(fields)
    write_meta(material_id, meta)
    return meta


def list_materials() -> list[dict[str, Any]]:
    if not MATERIALS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for entry in sorted(MATERIALS_DIR.iterdir()):
        if not entry.is_dir():
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
            }
        )
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out


def create_material_record(
    *,
    material_id: str,
    title: str,
    speaker: str,
    language: str,
    model_key: str,
) -> dict[str, Any]:
    meta = {
        "id": material_id,
        "title": title,
        "status": "queued",
        "progress": None,
        "speaker": speaker,
        "language": language,
        "model_key": model_key,
        "duration_sec": None,
        "error": None,
        "created_at": _now_iso(),
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
