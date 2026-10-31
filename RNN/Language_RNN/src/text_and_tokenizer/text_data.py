"""Korpus Pan Tadeusz, tokenizer BPE (Hugging Face) i Dataset sekwencji tokenów."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from log_print import log_print

PAN_TADEUSZ_URL = "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt"
CORPUS_FILENAME = "pan-tadeusz.txt"
DEFAULT_VOCAB_SIZE = 8_192


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    env = os.environ.get("LANGUAGE_RNN_DATA_DIR")
    if env:
        return Path(env)

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    candidates = [
        Path("/workspace/data/language"),
        project_root() / "build" / "data" / "language",
        cache_root / "rnn_lstm" / "corpus",
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
    raise RuntimeError("Brak zapisywalnego katalogu na tokenizer (ustaw LANGUAGE_RNN_DATA_DIR)")


def _corpus_candidates(data_dir: Path | None) -> list[Path]:
    data_dir = data_dir or resolve_data_dir()
    roots = [project_root() / "data", data_dir]
    names = (CORPUS_FILENAME, "pan_tadeusz.txt", "pan-tadeusz.txt")
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in names:
            path = root / name
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def resolve_corpus_path(data_dir: Path | None = None) -> Path:
    env = os.environ.get("LANGUAGE_RNN_CORPUS")
    if env:
        path = Path(env)
        if path.is_file():
            return path
        raise FileNotFoundError(f"LANGUAGE_RNN_CORPUS nie istnieje: {path}")

    for path in _corpus_candidates(data_dir):
        if path.is_file() and path.stat().st_size > 10_000:
            return path

    return _download_corpus(data_dir)


def _download_corpus(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or resolve_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / CORPUS_FILENAME

    log_print(f"Brak lokalnego korpusu — pobieram: {PAN_TADEUSZ_URL}")
    req = urllib.request.Request(PAN_TADEUSZ_URL, headers={"User-Agent": "Language_RNN-playground/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    text = raw.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    log_print(f"Zapisano {len(text):,} znaków → {path}")
    return path


def ensure_pan_tadeusz(data_dir: Path | None = None) -> Path:
    """Zachowane dla kompatybilności — zwraca ścieżkę do korpusu."""
    return resolve_corpus_path(data_dir)


def load_corpus_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _tokenizer_cache_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or resolve_data_dir()) / "bpe_bytelevel_tokenizer"


def load_bpe_tokenizer(
    corpus_path: Path,
    vocab_size: int | None = None,
    cache_dir: Path | None = None,
) -> PreTrainedTokenizerFast:
    """BPE na bajtach (ByteLevel, jak GPT-2) — spacje w tokenach, nie giną między słowami."""
    vocab_size = vocab_size or int(os.environ.get("LANGUAGE_RNN_VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
    cache_dir = cache_dir or _tokenizer_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = cache_dir / "tokenizer.json"

    if tokenizer_json.is_file():
        log_print(f"Tokenizer BPE z cache: {tokenizer_json}")
        return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))

    log_print(f"Trenuję ByteLevel BPE (vocab={vocab_size:,}) na {corpus_path.name}…")
    raw = Tokenizer(BPE(unk_token="<unk>"))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<|endoftext|>"],
        show_progress=False,
    )
    raw.train([str(corpus_path)], trainer)
    raw.save(str(tokenizer_json))

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<unk>",
        eos_token="<|endoftext|>",
    )
    log_print(f"Tokenizer zapisany → {cache_dir} (vocab={tok.vocab_size:,})")
    return tok


def encode_text(text: str, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)


class CharSeqDataset(Dataset):
    """Okno [i : i+seq_len] → predykcja następnego tokenu na każdej pozycji."""

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


def split_encoded(encoded: torch.Tensor, train_frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(encoded)
    split = int(n * train_frac)
    return encoded[:split].clone(), encoded[split:].clone()
