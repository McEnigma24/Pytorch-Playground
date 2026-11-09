#!/usr/bin/env python3
"""Print chars/sec per cached chunk JSON (timing from wav generation if present)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from server import materials as mat  # noqa: E402
from tts.document_pipeline import chunk_text  # noqa: E402


def measure_material(material_id: str) -> None:
    meta = mat.read_meta(material_id)
    if meta is None:
        raise SystemExit(f"No meta for {material_id}")

    text_path = mat.material_dir(material_id) / mat.TEXT_NAME
    if not text_path.is_file():
        raise SystemExit(f"Missing {mat.TEXT_NAME}")

    text = text_path.read_text(encoding="utf-8").strip()
    chunks = chunk_text(text)
    cache_dir = mat.chunks_dir(material_id)

    print(f"{material_id}: {len(chunks)} chunks, {sum(len(c) for c in chunks)} chars")
    if meta.get("sec_per_char"):
        print(f"  meta sec_per_char: {meta['sec_per_char']:.4f}")

    for idx, chunk in enumerate(chunks):
        meta_path = cache_dir / f"{idx:04d}.json"
        wav_path = cache_dir / f"{idx:04d}.wav"
        if not wav_path.is_file() and (mat.material_dir(material_id) / mat.WAV_NAME).is_file():
            # cache cleared on success — use duration from final wav is not per-chunk
            pass
        if not meta_path.is_file():
            print(f"  chunk {idx:04d}: (no cache meta)")
            continue
        try:
            import soundfile as sf

            wav, sr = sf.read(cache_dir / f"{idx:04d}.wav", dtype="float32")
            dur = len(wav) / float(sr)
        except Exception:
            dur = None
        if dur and len(chunk) > 0:
            print(f"  chunk {idx:04d}: {len(chunk)} chars, {dur:.2f} s audio, {len(chunk) / dur:.2f} chars/s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("material_id", nargs="?", default="Attention_Free_Transformer")
    args = p.parse_args()
    measure_material(args.material_id)


if __name__ == "__main__":
    main()
