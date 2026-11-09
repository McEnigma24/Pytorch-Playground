"""Korpus Speakleash, tokenizer BPE (Unicode + byte_fallback) i Dataset sekwencji tokenów."""

from __future__ import annotations

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
DEFAULT_VOCAB_SIZE = 8_192


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


def load_corpus_text(path: Path | None = None) -> str:
    path = path or resolve_corpus_path()
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def load_tokenizer(
    corpus_path: Path | None = None,
    vocab_size: int | None = None,
) -> PreTrainedTokenizerFast:
    """NFC + Metaspace, BPE z byte_fallback — zapis w data/tokenizer/."""
    corpus_path = corpus_path or resolve_corpus_path()
    vocab_size = vocab_size or int(os.environ.get("LANGUAGE_RNN_VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
    store = tokenizer_dir()
    store.mkdir(parents=True, exist_ok=True)
    tokenizer_json = store / "tokenizer.json"

    if tokenizer_json.is_file():
        log_print(f"Tokenizer z {store}")
        return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))

    log_print(f"Trenuję BPE byte_fallback (vocab={vocab_size:,}) na {corpus_path.name}…")
    raw = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    raw.normalizer = normalizers.NFC()
    raw.pre_tokenizer = Metaspace()
    raw.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Metaspace()])
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
    tok.save_pretrained(store)
    log_print(f"Tokenizer zapisany → {store} (vocab={tok.vocab_size:,})")
    return tok


def encode_text(text: str, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)


class TokenSeqDataset(Dataset):
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


def prepare_language_data(seq_len: int) -> tuple[TokenSeqDataset, TokenSeqDataset, PreTrainedTokenizerFast, int]:
    corpus_path = resolve_corpus_path()
    text = load_corpus_text(corpus_path)
    tokenizer = load_tokenizer(corpus_path)
    vocab_size = tokenizer.vocab_size
    encoded = encode_text(text, tokenizer)
    train_enc, val_enc = split_encoded(encoded, train_frac=0.9)
    train_ds = TokenSeqDataset(train_enc, seq_len)
    val_ds = TokenSeqDataset(val_enc, seq_len)
    log_print(
        f"Korpus: {SPEAKLEASH_DATASET}, {len(text):,} znaków → {len(encoded):,} tokenów, "
        f"vocab={vocab_size}, train_windows={len(train_ds):,}, val_windows={len(val_ds):,}"
    )
    return train_ds, val_ds, tokenizer, vocab_size
