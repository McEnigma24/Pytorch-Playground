"""Synthesize long documents in chunks; merge WAV + word-aligned VTT."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner
from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem
from qwen_tts import Qwen3TTSModel

from tts.infer_Qwen_3_tts import (
    _format_vtt_timestamp,
    load_forced_aligner,
    load_qwen_tts_model,
)

CHUNK_TARGET_CHARS = 500
INTER_CHUNK_SILENCE_SEC = 0.15
DEFAULT_MODEL = "1.7b-custom-voice"


@dataclass(frozen=True)
class PipelineResult:
    wav_path: Path
    vtt_path: Path
    duration_sec: float
    chunk_count: int


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = CHUNK_TARGET_CHARS) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + (1 if current else 0)
        if current and current_len + sent_len > max_chars:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def write_webvtt_word_aligned(path: Path, items: list[ForcedAlignItem]) -> None:
    chunks = ["WEBVTT", ""]
    cue_idx = 0
    for word in items:
        if not word.text.strip():
            continue
        cue_idx += 1
        chunks.append(str(cue_idx))
        chunks.append(
            f"{_format_vtt_timestamp(float(word.start_time))} --> "
            f"{_format_vtt_timestamp(float(word.end_time))}"
        )
        chunks.append(word.text)
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")


def process_document(
    text: str,
    out_wav: Path,
    out_vtt: Path,
    *,
    model_key: str = DEFAULT_MODEL,
    speaker: str | None = None,
    language: str = "English",
    qwen_model: Qwen3TTSModel | None = None,
    forced_aligner: Qwen3ForcedAligner | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> PipelineResult:
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("no text chunks")

    if qwen_model is None:
        qwen_model = load_qwen_tts_model(model_key)
    if forced_aligner is None:
        forced_aligner = load_forced_aligner()

    speakers = qwen_model.get_supported_speakers()
    if qwen_model.model.tts_model_type != "custom_voice":
        raise ValueError("Only custom_voice models are supported for document pipeline")

    if speaker is None:
        if not speakers:
            raise ValueError("speaker is required")
        speaker = speakers[0]

    use_cuda = torch.cuda.is_available()

    all_audio: list[np.ndarray] = []
    all_items: list[ForcedAlignItem] = []
    time_offset = 0.0
    sr: int | None = None

    for idx, chunk in enumerate(chunks):
        if on_progress:
            on_progress(idx + 1, len(chunks))

        if use_cuda:
            torch.cuda.synchronize()

        wavs, sample_rate = qwen_model.generate_custom_voice(
            text=chunk,
            language=language,
            speaker=speaker,
        )
        wav = np.asarray(wavs[0], dtype=np.float32)
        sr = sample_rate

        if use_cuda:
            torch.cuda.synchronize()

        t_align = time.perf_counter()
        align_results = forced_aligner.align(
            audio=(wav, sample_rate),
            text=chunk,
            language=language,
        )
        if use_cuda:
            torch.cuda.synchronize()
        del t_align

        items = align_results[0].items
        for item in items:
            if not item.text.strip():
                continue
            all_items.append(
                ForcedAlignItem(
                    text=item.text,
                    start_time=float(item.start_time) + time_offset,
                    end_time=float(item.end_time) + time_offset,
                )
            )

        all_audio.append(wav)
        chunk_dur = len(wav) / float(sample_rate)
        time_offset += chunk_dur

        if idx + 1 < len(chunks):
            silence = np.zeros(int(INTER_CHUNK_SILENCE_SEC * sample_rate), dtype=np.float32)
            all_audio.append(silence)
            time_offset += INTER_CHUNK_SILENCE_SEC

    assert sr is not None
    merged = np.concatenate(all_audio)
    duration_sec = len(merged) / float(sr)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, merged, sr)
    write_webvtt_word_aligned(out_vtt, all_items)

    return PipelineResult(
        wav_path=out_wav,
        vtt_path=out_vtt,
        duration_sec=duration_sec,
        chunk_count=len(chunks),
    )
