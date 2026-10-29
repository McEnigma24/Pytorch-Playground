"""Korpus, słownik znaków i Dataset sekwencji."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

from log_print import log_print

# Wolne Lektury — UTF-8, public domain
PAN_TADEUSZ_URL = "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt"
CORPUS_FILENAME = "pan_tadeusz.txt"


def resolve_data_dir() -> Path:
    env = os.environ.get("LANGUAGE_RNN_DATA_DIR")
    if env:
        return Path(env)

    project_root = Path(__file__).resolve().parents[2]
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    candidates = [
        Path("/workspace/data/language"),
        project_root / "data" / "language",
        project_root / "build" / "data" / "language",
        cache_root / "language_rnn" / "corpus",
    ]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError(
        "Brak zapisywalnego katalogu na korpus (ustaw LANGUAGE_RNN_DATA_DIR)"
    )


def ensure_pan_tadeusz(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or resolve_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / CORPUS_FILENAME
    if path.is_file() and path.stat().st_size > 10_000:
        return path

    log_print(f"Pobieram korpus: {PAN_TADEUSZ_URL}")
    req = urllib.request.Request(
        PAN_TADEUSZ_URL,
        headers={"User-Agent": "Language_RNN-playground/0.1"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    text = raw.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    log_print(f"Zapisano {len(text):,} znaków → {path}")
    return path


def load_corpus_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = chars
    return char2idx, idx2char


def encode_text(text: str, char2idx: dict[str, int]) -> torch.Tensor:
    return torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)


class CharSeqDataset(Dataset):
    """Okno [i : i+seq_len] → predykcja następnego znaku na każdej pozycji."""

    def __init__(self, encoded: torch.Tensor, seq_len: int):
        if encoded.ndim != 1:
            raise ValueError("encoded musi być tensorem 1D")
        if len(encoded) <= seq_len:
            raise ValueError("tekst za krótki względem seq_len")
        self.encoded = encoded
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.encoded) - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.encoded[index : index + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def split_encoded(
    encoded: torch.Tensor,
    train_frac: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(encoded)
    split = int(n * train_frac)
    return encoded[:split].clone(), encoded[split:].clone()
