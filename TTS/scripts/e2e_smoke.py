#!/usr/bin/env python3
"""Smoke test: PDF upload API + optional short TTS processing."""

from __future__ import annotations

import os
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from server import materials as mat  # noqa: E402
from server.app import app  # noqa: E402

TRANSFORMERS_SNIPPET = (
    "Self attention mechanisms, represented by Transformers, have driven the advancement "
    "of various machine learning problems, including language understanding and computer vision."
)


def make_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 50, 500, 750)
    page.insert_textbox(rect, text, fontsize=11)
    bio = BytesIO()
    doc.save(bio)
    doc.close()
    return bio.getvalue()


def test_api_ready_fixture() -> None:
    """Serve existing WAV/VTT without running GPU."""
    mat.MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    mid = "e2e-fixture-ready"
    out = mat.material_dir(mid)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)
    (out / mat.PDF_NAME).write_bytes(pdf)
    (out / mat.TEXT_NAME).write_text(TRANSFORMERS_SNIPPET + "\n", encoding="utf-8")

    latest_wav = ROOT / "output" / "_latest_.wav"
    latest_vtt = ROOT / "output" / "_latest_.vtt"
    if not latest_wav.is_file() or not latest_vtt.is_file():
        raise SystemExit("Missing output/_latest_.wav or .vtt for fixture test")

    shutil.copy2(latest_wav, out / mat.WAV_NAME)
    shutil.copy2(latest_vtt, out / mat.VTT_NAME)
    mat.create_material_record(
        material_id=mid,
        title="E2E fixture",
        speaker="ryan",
        language="English",
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(mid, status="ready", progress="1/1", duration_sec=30.0)

    with TestClient(app) as client:
        r = client.get("/api/materials")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()}
        assert mid in ids

        for kind in ("pdf", "wav", "vtt"):
            g = client.get(f"/api/materials/{mid}/{kind}")
            assert g.status_code == 200, (kind, g.status_code, g.text[:200])
            assert g.content, kind

    print("OK fixture:", mid)


def test_upload_queued() -> None:
    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)
    files = {"pdf": ("transformers.pdf", pdf, "application/pdf")}
    data = {"title": "Transformers snippet", "text": TRANSFORMERS_SNIPPET, "speaker": "ryan"}

    with TestClient(app) as client:
        r = client.post("/api/materials", files=files, data=data)
        assert r.status_code == 202, r.text
        body = r.json()
        mid = body["id"]
        print("OK upload queued:", mid)

        if os.environ.get("RUN_TTS_E2E") != "1":
            print("Skip GPU wait (set RUN_TTS_E2E=1 to process)")
            return

        deadline = time.time() + 900
        while time.time() < deadline:
            meta = mat.read_meta(mid)
            assert meta
            st = meta.get("status")
            if st == "ready":
                client.get(f"/api/materials/{mid}/wav").raise_for_status()
                client.get(f"/api/materials/{mid}/vtt").raise_for_status()
                print("OK TTS ready:", mid, meta.get("duration_sec"))
                return
            if st == "error":
                raise AssertionError(meta.get("error"))
            time.sleep(5)
        raise TimeoutError(mid)


if __name__ == "__main__":
    test_api_ready_fixture()
    test_upload_queued()
