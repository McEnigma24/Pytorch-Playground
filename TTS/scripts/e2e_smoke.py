#!/usr/bin/env python3
"""Smoke test: upload → extract → generate (optional GPU TTS + pause/resume)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

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


def make_pdf_with_footer(main: str, footer: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 500, 620), main, fontsize=11)
    page.insert_textbox(pymupdf.Rect(50, 650, 500, 670), "------------------------", fontsize=11)
    page.insert_textbox(pymupdf.Rect(50, 680, 500, 720), footer, fontsize=10)
    bio = BytesIO()
    doc.save(bio)
    doc.close()
    return bio.getvalue()


def test_page_bottom_rule_helpers() -> None:
    from server.pdf_text import _is_horizontal_rule_line, _strip_text_below_page_rule

    assert _is_horizontal_rule_line("----------------")
    assert not _is_horizontal_rule_line("Introduction")
    page = "Main body line.\n\n----------\nPreprint. Under review."
    out = _strip_text_below_page_rule(page)
    assert "Preprint" not in out
    assert "Main body" in out


def test_extract_strips_page_footer() -> None:
    from server.pdf_text import extract_text_from_pdf

    pdf = make_pdf_with_footer(TRANSFORMERS_SNIPPET, "Preprint. Under review.")
    fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    path = Path(tmp_name)
    try:
        path.write_bytes(pdf)
        text, _ = extract_text_from_pdf(
            path,
            page_from=1,
            page_to=1,
            filter_ids={"page_bottom_rule", "dehyphenate"},
        )
        assert "Preprint" not in text
        assert "Transformers" in text
    finally:
        path.unlink(missing_ok=True)


def test_eta_fields_finish_at() -> None:
    from server.worker import _eta_fields

    fields = _eta_fields(100, 25, 0.4)
    assert fields["eta_sec"] == 30.0
    assert fields.get("eta_finish_at")
    assert fields.get("eta_at")


def test_inactive_processing_delete_and_pause() -> None:
    mid = f"e2e_inactive_{int(time.time())}"
    if mat.material_dir(mid).exists():
        mat.delete_material(mid)
    out = mat.material_dir(mid)
    out.mkdir(parents=True)
    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)
    (out / mat.PDF_NAME).write_bytes(pdf)
    (out / mat.TEXT_NAME).write_text(TRANSFORMERS_SNIPPET + "\n", encoding="utf-8")
    mat.create_uploaded_record(
        material_id=mid,
        title="inactive",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(mid, status="processing", chars_total=50)

    with TestClient(app) as client:
        r = client.delete(f"/api/materials/{mid}")
        assert r.status_code == 200, r.text
        assert r.json().get("deleted") is True
        assert not mat.material_dir(mid).exists()

    mid2 = f"e2e_inactive2_{int(time.time())}"
    out2 = mat.material_dir(mid2)
    out2.mkdir(parents=True)
    (out2 / mat.PDF_NAME).write_bytes(pdf)
    (out2 / mat.TEXT_NAME).write_text(TRANSFORMERS_SNIPPET + "\n", encoding="utf-8")
    mat.create_uploaded_record(
        material_id=mid2,
        title="inactive2",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(mid2, status="processing")

    with TestClient(app) as client:
        r = client.post(f"/api/materials/{mid2}/pause")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "paused"
    meta = mat.read_meta(mid2)
    assert meta and meta["status"] == "paused"
    mat.delete_material(mid2)


def test_chunk_cache_validates_text_sha1() -> None:
    import numpy as np
    from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem

    from tts.document_pipeline import _chunk_cache_paths, _load_cached_chunk, _save_cached_chunk

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        wav_path, meta_path = _chunk_cache_paths(cache, 0)
        wav_arr = np.zeros(1600, dtype=np.float32)
        items = [ForcedAlignItem(text="hi", start_time=0.0, end_time=0.1)]
        text_a = "chunk A text"
        _save_cached_chunk(wav_path, meta_path, wav_arr, 16000, items, text_a)
        assert _load_cached_chunk(wav_path, meta_path, text_a) is not None
        assert _load_cached_chunk(wav_path, meta_path, "other text") is None


def test_inactive_abort_keeps_chunk_cache() -> None:
    mid = f"e2e_abort_{int(time.time())}"
    if mat.material_dir(mid).exists():
        mat.delete_material(mid)
    out = mat.material_dir(mid)
    out.mkdir(parents=True)
    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)
    (out / mat.PDF_NAME).write_bytes(pdf)
    (out / mat.TEXT_NAME).write_text(TRANSFORMERS_SNIPPET + "\n", encoding="utf-8")
    mat.create_uploaded_record(
        material_id=mid,
        title="abort inactive",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    cache = mat.chunks_dir(mid)
    cache.mkdir(parents=True)
    (cache / "0000.json").write_text('{"text_sha1":"x"}', encoding="utf-8")
    mat.update_meta(mid, status="processing")

    with TestClient(app) as client:
        r = client.post(f"/api/materials/{mid}/abort")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "paused"
        assert body.get("active") is False

    assert cache.is_dir()
    meta = mat.read_meta(mid)
    assert meta and meta["status"] == "paused"
    mat.delete_material(mid)


def test_abort_forward_pre_hook_raises() -> None:
    import threading

    import torch
    import torch.nn as nn

    from tts.document_pipeline import StopRequested

    abort = threading.Event()
    mod = nn.Linear(4, 4)

    def hook(_module, _inputs) -> None:
        if abort.is_set():
            raise StopRequested()

    handle = mod.register_forward_pre_hook(hook)
    try:
        x = torch.zeros(1, 4)
        mod(x)
        abort.set()
        try:
            mod(x)
            raise AssertionError("expected StopRequested")
        except StopRequested:
            pass
    finally:
        handle.remove()


def test_recover_orphans() -> None:
    from server.worker import recover_orphans

    mid_del = f"e2e_orphan_del_{int(time.time())}"
    mat.material_dir(mid_del).mkdir(parents=True)
    mat.create_uploaded_record(
        material_id=mid_del,
        title="orphan_del",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(mid_del, status="processing", stop_requested="delete")
    recover_orphans()
    assert not mat.material_dir(mid_del).exists()

    mid = f"e2e_orphan_{int(time.time())}"
    mat.material_dir(mid).mkdir(parents=True)
    mat.create_uploaded_record(
        material_id=mid,
        title="orphan",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(mid, status="processing")
    recover_orphans()
    meta = mat.read_meta(mid)
    assert meta and meta["status"] == "paused"
    mat.delete_material(mid)

    mid_abort = f"e2e_orphan_abort_{int(time.time())}"
    mat.material_dir(mid_abort).mkdir(parents=True)
    mat.chunks_dir(mid_abort).mkdir(parents=True)
    mat.create_uploaded_record(
        material_id=mid_abort,
        title="orphan_abort",
        num_pages=1,
        model_key="1.7b-custom-voice",
    )
    mat.update_meta(mid_abort, status="processing", stop_requested="abort")
    recover_orphans()
    meta_abort = mat.read_meta(mid_abort)
    assert meta_abort and meta_abort["status"] == "paused"
    assert mat.chunks_dir(mid_abort).is_dir()
    mat.delete_material(mid_abort)


def test_create_material_from_url() -> None:
    pdf = make_pdf_bytes(TRANSFORMERS_SNIPPET)
    title = f"URL Snippet {int(time.time())}"

    with patch("server.app.fetch_pdf_bytes", return_value=pdf):
        with TestClient(app) as client:
            r = client.post(
                "/api/materials",
                data={"title": title, "pdf_url": "https://example.org/papers/sample.pdf"},
            )
            assert r.status_code == 201, r.text
            body = r.json()
            mid = body["id"]
            meta = mat.read_meta(mid)
            assert meta
            assert meta.get("source_url") == "https://example.org/papers/sample.pdf"

            r = client.post(
                f"/api/materials/{mid}/extract",
                json={"filters": ["page_bottom_rule", "dehyphenate"], "page_from": 1, "page_to": 1},
            )
            assert r.status_code == 200, r.text
            assert "Transformers" in r.json()["text"]

            client.delete(f"/api/materials/{mid}")

    print("OK url create:", title)


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
    test_page_bottom_rule_helpers()
    test_extract_strips_page_footer()
    test_eta_fields_finish_at()
    test_chunk_cache_validates_text_sha1()
    test_inactive_abort_keeps_chunk_cache()
    test_abort_forward_pre_hook_raises()
    test_inactive_processing_delete_and_pause()
    test_recover_orphans()
    test_create_material_from_url()
    test_api_ready_fixture()
    test_v2_flow()
