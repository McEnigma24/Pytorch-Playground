#!/usr/bin/env python3
"""Smoke test: upload → extract → generate (optional GPU TTS + pause/resume)."""

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
from server.pdf_text import is_subsequence_of  # noqa: E402

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


def test_subsequence_validator() -> None:
    assert is_subsequence_of("hello world", "x hello y world z")
    assert not is_subsequence_of("hellox", "hello")


def test_api_ready_fixture() -> None:
    mat.MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    mid = "e2e-fixture-ready"
    if mat.material_dir(mid).exists():
        mat.delete_material(mid)
    out = mat.material_dir(mid)
    out.mkdir(parents=True)

    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)
    (out / mat.PDF_NAME).write_bytes(pdf)
    (out / mat.EXTRACTED_NAME).write_text(TRANSFORMERS_SNIPPET + "\n", encoding="utf-8")
    (out / mat.TEXT_NAME).write_text(TRANSFORMERS_SNIPPET + "\n", encoding="utf-8")

    latest_wav = ROOT / "output" / "_latest_.wav"
    latest_vtt = ROOT / "output" / "_latest_.vtt"
    if not latest_wav.is_file() or not latest_vtt.is_file():
        raise SystemExit("Missing output/_latest_.wav or .vtt for fixture test")

    shutil.copy2(latest_wav, out / mat.WAV_NAME)
    shutil.copy2(latest_vtt, out / mat.VTT_NAME)
    mat.create_uploaded_record(
        material_id=mid,
        title="E2E fixture",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(
        mid,
        status="ready",
        progress="1/1",
        duration_sec=30.0,
        page_from=1,
        page_to=1,
    )

    with TestClient(app) as client:
        r = client.get("/api/materials")
        assert r.status_code == 200
        assert mid in {m["id"] for m in r.json()}

        for kind in ("pdf", "wav", "vtt", "extracted"):
            g = client.get(f"/api/materials/{mid}/{kind}")
            assert g.status_code == 200, (kind, g.status_code)

        client.delete(f"/api/materials/{mid}").raise_for_status()

    print("OK fixture:", mid)


def test_v2_flow() -> None:
    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)

    with TestClient(app) as client:
        r = client.get("/api/filters")
        assert r.status_code == 200
        assert any(f["id"] == "square_brackets" for f in r.json())

        rs = client.get("/api/stats")
        assert rs.status_code == 200
        assert "sec_per_char_prior" in rs.json()

        title = f"Transformers Snippet {int(time.time())}"
        r = client.post(
            "/api/materials",
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
            data={"title": title},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        mid = body["id"]
        assert " " not in mid
        assert body["status"] == "uploaded"

        r = client.post(
            f"/api/materials/{mid}/extract",
            json={"filters": ["square_brackets", "dehyphenate"], "page_from": 1, "page_to": 1},
        )
        assert r.status_code == 200, r.text
        extracted = r.json()["text"]
        assert "Transformers" in extracted

        trimmed = extracted.replace("computer vision.", "").strip()

        bad = trimmed + "EXTRA"
        r = client.post(
            f"/api/materials/{mid}/generate",
            json={"text": bad, "speaker": "ryan"},
        )
        assert r.status_code == 422

        r = client.post(
            f"/api/materials/{mid}/generate",
            json={"text": trimmed, "speaker": "ryan", "language": "English"},
        )
        assert r.status_code == 202, r.text

        if os.environ.get("RUN_TTS_E2E") != "1":
            client.post(f"/api/materials/{mid}/pause")
            client.delete(f"/api/materials/{mid}")
            print("OK v2 flow (no GPU):", mid)
            return

        time.sleep(2)
        client.post(f"/api/materials/{mid}/pause").raise_for_status()
        meta = mat.read_meta(mid)
        assert meta and meta.get("status") in ("paused", "processing", "queued")

        deadline = time.time() + 120
        while time.time() < deadline:
            meta = mat.read_meta(mid)
            if meta is None:
                break
            if meta.get("status") == "paused":
                break
            time.sleep(0.5)

        client.post(f"/api/materials/{mid}/resume").raise_for_status()

        deadline = time.time() + 900
        while time.time() < deadline:
            meta = mat.read_meta(mid)
            assert meta
            st = meta.get("status")
            if st == "ready":
                cache = mat.chunks_dir(mid)
                had_cache = cache.is_dir() and any(cache.iterdir())
                client.get(f"/api/materials/{mid}/wav").raise_for_status()
                assert meta.get("chars_total")
                assert meta.get("chars_done") == meta.get("chars_total")
                assert mat.MATERIALS_DIR.joinpath("_stats.json").is_file()
                print("OK TTS ready:", mid, "cache_before_resume=", had_cache)
                client.delete(f"/api/materials/{mid}")
                return
            if st == "error":
                raise AssertionError(meta.get("error"))
            time.sleep(5)
        raise TimeoutError(mid)


if __name__ == "__main__":
    test_subsequence_validator()
    test_api_ready_fixture()
    test_v2_flow()
