"""Korpus Speakleash, tokenizer BPE (Unicode + byte_fallback) i Dataset sekwencji tokenów."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from speakleash import Speakleash
from torch.utils.data import Dataset
from tokenizers import Tokenizer, decoders, normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from log_print import log_print

SPEAKLEASH_DATASET = "wolne_lektury_corpus"
CORPUS_FILENAME = "corpus.txt"
DEFAULT_VOCAB_SIZE = 32_000
BYTE_VOCAB_RESERVE = 256


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return package_dir().parents[1]


def data_dir() -> Path:
    env = os.environ.get("LANGUAGE_RNN_DATA_DIR")
    if env:
        return Path(env)
    return package_dir() / "data"


def text_dir() -> Path:
    return data_dir() / "text"


def speakleash_dir() -> Path:
    return data_dir() / "speakleash"


def tokenizer_dir() -> Path:
    env = os.environ.get("LANGUAGE_RNN_TOKENIZER_DIR")
    if env:
        return Path(env)
    return data_dir() / "tokenizer"


def default_corpus_path() -> Path:
    return text_dir() / CORPUS_FILENAME


def _materialize_corpus(dest: Path) -> Path:
    text_dir().mkdir(parents=True, exist_ok=True)
    store = speakleash_dir()
    store.mkdir(parents=True, exist_ok=True)
    log_print(f"Speakleash: {SPEAKLEASH_DATASET} → {dest}")
    sl = Speakleash(replicate_dir=str(store))
    ds = sl.get(SPEAKLEASH_DATASET)
    if ds is None:
        raise RuntimeError(f"Brak datasetu Speakleash: {SPEAKLEASH_DATASET!r}")
    stream = ds.data
    if stream is None:
        raise RuntimeError(f"Nie udało się pobrać Speakleash/{SPEAKLEASH_DATASET}")

    doc_count = 0
    with dest.open("w", encoding="utf-8") as out:
        for doc in stream:
            if not doc:
                continue
            doc = doc.replace("\r\n", "\n").replace("\r", "\n")
            out.write(doc)
            out.write("\n\n")
            doc_count += 1

    size = dest.stat().st_size
    log_print(
        f"Korpus: {doc_count:,} dokumentów, {size:,} B ({SPEAKLEASH_DATASET})"
    )
    return dest


def resolve_corpus_path() -> Path:
    dest = default_corpus_path()
    if dest.is_file() and dest.stat().st_size > 10_000:
        return dest
    return _materialize_corpus(dest)


def _inject_byte_tokens_json(tokenizer_json: Path) -> int:
    """Dopina <0x00>..<0xFF> do model.vocab w tokenizer.json (bez przebudowy merges)."""
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab: dict[str, int] = data["model"]["vocab"]
    next_id = max(vocab.values()) + 1
    added = 0
    for i in range(BYTE_VOCAB_RESERVE):
        tok = f"<0x{i:02x}>"
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
            added += 1
    tokenizer_json.write_text(json.dumps(data), encoding="utf-8")
    return added


def _apply_corpus_char_limit(text: str) -> str:
    raw = os.environ.get("LANGUAGE_RNN_MAX_CORPUS_CHARS", "").strip()
    if not raw:
        return text
    limit = int(raw)
    if len(text) <= limit:
        return text
    log_print(
        f"Obcinam korpus: {len(text):,} → {limit:,} znaków (LANGUAGE_RNN_MAX_CORPUS_CHARS)"
    )
    return text[:limit]


def load_corpus_text(path: Path | None = None) -> str:
    path = path or resolve_corpus_path()
    log_print(f"Wczytuję korpus {path.name}…")
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _apply_corpus_char_limit(text)
    log_print(f"Korpus w pamięci: {len(text):,} znaków")
    return text


def load_tokenizer(
    corpus_path: Path | None = None,
    vocab_size: int | None = None,
) -> PreTrainedTokenizerFast:
    """NFC + Metaspace, BPE z byte_fallback — zapis w data/tokenizer/."""
    vocab_size = vocab_size or int(os.environ.get("LANGUAGE_RNN_VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
    store = tokenizer_dir()
    store.mkdir(parents=True, exist_ok=True)
    tokenizer_json = store / "tokenizer.json"

    if tokenizer_json.is_file():
        tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
        if tok.vocab_size != vocab_size:
            log_print(
                f"UWAGA: tokenizer ma vocab={tok.vocab_size:,}, oczekiwano {vocab_size:,} — "
                f"usuń {store} i przebuduj BPE"
            )
        log_print(f"Tokenizer z {store}")
        return tok

    corpus_path = corpus_path or resolve_corpus_path()
    merge_vocab = max(vocab_size - BYTE_VOCAB_RESERVE, 512)
    log_print(
        f"Trenuję BPE byte_fallback (vocab={vocab_size:,}, merge={merge_vocab:,}) na {corpus_path.name}…"
    )
    raw = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    raw.normalizer = normalizers.NFC()
    raw.pre_tokenizer = Metaspace()
    raw.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Metaspace()])
    trainer = BpeTrainer(
        vocab_size=merge_vocab,
        special_tokens=["<unk>", "<|endoftext|>"],
        show_progress=False,
    )
    raw.train([str(corpus_path)], trainer)
    raw.save(str(tokenizer_json))
    byte_added = _inject_byte_tokens_json(tokenizer_json)
    vocab_n = len(json.loads(tokenizer_json.read_text(encoding="utf-8"))["model"]["vocab"])
    if vocab_n != vocab_size:
        log_print(
            f"UWAGA: tokenizer vocab={vocab_n:,} ≠ docelowe {vocab_size:,} "
            f"(merge={merge_vocab:,}, dopięte bajty={byte_added})"
        )

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<unk>",
        eos_token="<|endoftext|>",
    )
    tok.save_pretrained(store)
    log_print(f"Tokenizer zapisany → {store} (vocab={tok.vocab_size:,})")
    return tok


def encode_text(text: str, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    chunk_chars = int(os.environ.get("LANGUAGE_RNN_ENCODE_CHUNK_CHARS", "2_000_000").replace("_", ""))
    if len(text) <= chunk_chars:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    log_print(f"Tokenizuję korpus w blokach po {chunk_chars:,} znaków…")
    ids: list[int] = []
    for start in range(0, len(text), chunk_chars):
        part = text[start : start + chunk_chars]
        ids.extend(tokenizer.encode(part, add_special_tokens=False))
        if start == 0 or (start // chunk_chars) % 5 == 0:
            log_print(f"  …{min(start + chunk_chars, len(text)):,}/{len(text):,} znaków → {len(ids):,} tokenów")
    return torch.tensor(ids, dtype=torch.long)


class TokenSeqDataset(Dataset):
    """Okno [i : i+seq_len] → predykcja następnego tokenu na każdej pozycji."""

    def __init__(self, encoded: torch.Tensor, seq_len: int, stride: int = 1):
        if encoded.ndim != 1:
            raise ValueError("encoded musi być tensorem 1D")
        if stride < 1:
            raise ValueError("stride musi być >= 1")
        if len(encoded) <= seq_len:
            raise ValueError("tekst za krótki względem seq_len")
        self.encoded = encoded
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self) -> int:
        span = len(self.encoded) - self.seq_len
        return (span + self.stride - 1) // self.stride

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        chunk = self.encoded[start : start + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def split_encoded(encoded: torch.Tensor, train_frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(encoded)
    split = int(n * train_frac)
    return encoded[:split].clone(), encoded[split:].clone()


def prepare_language_data(
    seq_len: int,
    *,
    window_stride: int = 1,
) -> tuple[TokenSeqDataset, TokenSeqDataset, PreTrainedTokenizerFast, int]:
    corpus_path = resolve_corpus_path()
    text = load_corpus_text(corpus_path)
    tokenizer = load_tokenizer(corpus_path)
    vocab_size = tokenizer.vocab_size
    log_print("Koduję tekst na id tokenów (może chwilę potrwać)…")
    n_chars = len(text)
    encoded = encode_text(text, tokenizer)
    del text
    train_enc, val_enc = split_encoded(encoded, train_frac=0.9)
    train_ds = TokenSeqDataset(train_enc, seq_len, stride=window_stride)
    val_ds = TokenSeqDataset(val_enc, seq_len, stride=window_stride)
    log_print(
        f"Korpus: {SPEAKLEASH_DATASET}, {n_chars:,} znaków → {len(encoded):,} tokenów, "
        f"vocab={vocab_size}, stride={window_stride}, "
        f"train_windows={len(train_ds):,}, val_windows={len(val_ds):,}"
    )
    return train_ds, val_ds, tokenizer, vocab_size
