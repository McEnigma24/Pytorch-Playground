"""Korpus Speakleash, tokenizer BPE (Unicode + byte_fallback) i Dataset sekwencji tokenów."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

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
DEFAULT_TRAIN_FRAC = 0.9
ENCODED_CACHE_VERSION = 1
ENCODED_META_FILENAME = "meta.json"
ENCODED_TRAIN_FILENAME = "train_ids.pt"
ENCODED_VAL_FILENAME = "val_ids.pt"

# Ten sam klucz w procesie → bez ponownego wczytywania z dysku (np. dwa train.main w main.py).
_SESSION_CACHE_KEY: tuple[Any, ...] | None = None
_SESSION_TRAIN_IDS: torch.Tensor | None = None
_SESSION_VAL_IDS: torch.Tensor | None = None
_SESSION_N_CHARS: int = 0


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


def encoded_dir() -> Path:
    env = os.environ.get("LANGUAGE_RNN_ENCODED_DIR")
    if env:
        return Path(env)
    return data_dir() / "encoded"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _encoded_cache_mode() -> str:
    raw = os.environ.get("LANGUAGE_RNN_ENCODED_CACHE", "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("force", "rebuild"):
        return "force"
    return "auto"


def _max_corpus_chars_config() -> int | None:
    raw = os.environ.get("LANGUAGE_RNN_MAX_CORPUS_CHARS", "").strip()
    return int(raw) if raw else None


def _train_frac() -> float:
    raw = os.environ.get("LANGUAGE_RNN_TRAIN_FRAC", "").strip()
    if not raw:
        return DEFAULT_TRAIN_FRAC
    return float(raw)


def _cache_key_tuple(
    *,
    corpus_path: Path,
    corpus_sha256: str,
    tokenizer_sha256: str,
    vocab_size: int,
    max_chars: int | None,
    train_frac: float,
) -> tuple[Any, ...]:
    return (
        ENCODED_CACHE_VERSION,
        str(corpus_path.resolve()),
        corpus_sha256,
        tokenizer_sha256,
        vocab_size,
        max_chars,
        train_frac,
    )


def _expected_meta(
    *,
    corpus_path: Path,
    corpus_sha256: str,
    corpus_size_bytes: int,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    vocab_size: int,
    max_chars: int | None,
    train_frac: float,
    n_chars_encoded: int,
    n_train_tokens: int,
    n_val_tokens: int,
) -> dict[str, Any]:
    return {
        "version": ENCODED_CACHE_VERSION,
        "dataset": SPEAKLEASH_DATASET,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": corpus_sha256,
        "corpus_size_bytes": corpus_size_bytes,
        "max_corpus_chars": max_chars,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": vocab_size,
        "train_frac": train_frac,
        "n_chars_encoded": n_chars_encoded,
        "n_train_tokens": n_train_tokens,
        "n_val_tokens": n_val_tokens,
    }


def _meta_fingerprint(meta: dict[str, Any]) -> tuple[Any, ...]:
    return _cache_key_tuple(
        corpus_path=Path(meta["corpus_path"]),
        corpus_sha256=str(meta["corpus_sha256"]),
        tokenizer_sha256=str(meta["tokenizer_sha256"]),
        vocab_size=int(meta["vocab_size"]),
        max_chars=meta.get("max_corpus_chars"),
        train_frac=float(meta["train_frac"]),
    )


def _cache_stale_reason(expected: dict[str, Any], stored: dict[str, Any]) -> str | None:
    if int(stored.get("version", -1)) != ENCODED_CACHE_VERSION:
        return f"stara wersja cache ({stored.get('version')})"
    for key in (
        "corpus_sha256",
        "corpus_size_bytes",
        "max_corpus_chars",
        "tokenizer_sha256",
        "vocab_size",
        "train_frac",
    ):
        if stored.get(key) != expected.get(key):
            return f"zmiana {key!r}: {stored.get(key)!r} → {expected.get(key)!r}"
    if stored.get("corpus_path") != expected.get("corpus_path"):
        return "inna ścieżka korpusu"
    return None


def _save_encoded_cache(
    store: Path,
    *,
    meta: dict[str, Any],
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
) -> None:
    store.mkdir(parents=True, exist_ok=True)
    meta_path = store / ENCODED_META_FILENAME
    train_path = store / ENCODED_TRAIN_FILENAME
    val_path = store / ENCODED_VAL_FILENAME
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_train = train_path.with_suffix(".pt.tmp")
    tmp_val = val_path.with_suffix(".pt.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(train_ids, tmp_train)
    torch.save(val_ids, tmp_val)
    tmp_meta.replace(meta_path)
    tmp_train.replace(train_path)
    tmp_val.replace(val_path)
    log_print(f"Cache encoded zapisany → {store}")


def _load_encoded_cache(store: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor] | None:
    meta_path = store / ENCODED_META_FILENAME
    train_path = store / ENCODED_TRAIN_FILENAME
    val_path = store / ENCODED_VAL_FILENAME
    if not (meta_path.is_file() and train_path.is_file() and val_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_ids = torch.load(train_path, map_location="cpu", weights_only=True)
    val_ids = torch.load(val_path, map_location="cpu", weights_only=True)
    if not isinstance(train_ids, torch.Tensor) or not isinstance(val_ids, torch.Tensor):
        raise RuntimeError(f"Uszkodzony cache encoded w {store}")
    return meta, train_ids, val_ids


def _get_train_val_encoded(
    corpus_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    tokenizer_json: Path,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Wspólny train/val split tokenów — cache na dysku + w sesji procesu."""
    global _SESSION_CACHE_KEY, _SESSION_TRAIN_IDS, _SESSION_VAL_IDS, _SESSION_N_CHARS

    vocab_size = tokenizer.vocab_size
    max_chars = _max_corpus_chars_config()
    train_frac = _train_frac()
    corpus_sha256 = _sha256_file(corpus_path)
    corpus_size = corpus_path.stat().st_size
    tokenizer_sha256 = _sha256_file(tokenizer_json)
    cache_key = _cache_key_tuple(
        corpus_path=corpus_path,
        corpus_sha256=corpus_sha256,
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=vocab_size,
        max_chars=max_chars,
        train_frac=train_frac,
    )

    if _SESSION_CACHE_KEY == cache_key and _SESSION_TRAIN_IDS is not None and _SESSION_VAL_IDS is not None:
        log_print("Cache encoded: używam tensorów z tej sesji procesu")
        return _SESSION_TRAIN_IDS, _SESSION_VAL_IDS, _SESSION_N_CHARS

    mode = _encoded_cache_mode()
    store = encoded_dir()
    placeholder_meta = _expected_meta(
        corpus_path=corpus_path,
        corpus_sha256=corpus_sha256,
        corpus_size_bytes=corpus_size,
        tokenizer_path=tokenizer_json,
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=vocab_size,
        max_chars=max_chars,
        train_frac=train_frac,
        n_chars_encoded=0,
        n_train_tokens=0,
        n_val_tokens=0,
    )

    if mode != "off" and mode != "force":
        loaded = _load_encoded_cache(store)
        if loaded is not None:
            stored_meta, train_ids, val_ids = loaded
            reason = _cache_stale_reason(placeholder_meta, stored_meta)
            if reason is None and _meta_fingerprint(stored_meta) == cache_key:
                log_print(
                    f"Cache encoded: wczytano {len(train_ids):,} + {len(val_ids):,} tokenów z {store}"
                )
                _SESSION_CACHE_KEY = cache_key
                _SESSION_TRAIN_IDS = train_ids
                _SESSION_VAL_IDS = val_ids
                _SESSION_N_CHARS = int(stored_meta.get("n_chars_encoded", 0))
                return train_ids, val_ids, _SESSION_N_CHARS
            if reason:
                log_print(f"Cache encoded: przebudowa ({reason})")

    text = load_corpus_text(corpus_path)
    n_chars = len(text)
    log_print("Koduję tekst na id tokenów (może chwilę potrwać)…")
    encoded = encode_text(text, tokenizer)
    del text
    train_ids, val_ids = split_encoded(encoded, train_frac=train_frac)
    del encoded

    if mode != "off":
        meta = _expected_meta(
            corpus_path=corpus_path,
            corpus_sha256=corpus_sha256,
            corpus_size_bytes=corpus_size,
            tokenizer_path=tokenizer_json,
            tokenizer_sha256=tokenizer_sha256,
            vocab_size=vocab_size,
            max_chars=max_chars,
            train_frac=train_frac,
            n_chars_encoded=n_chars,
            n_train_tokens=len(train_ids),
            n_val_tokens=len(val_ids),
        )
        _save_encoded_cache(store, meta=meta, train_ids=train_ids, val_ids=val_ids)

    _SESSION_CACHE_KEY = cache_key
    _SESSION_TRAIN_IDS = train_ids
    _SESSION_VAL_IDS = val_ids
    _SESSION_N_CHARS = n_chars
    return train_ids, val_ids, n_chars


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
    tokenizer = load_tokenizer(corpus_path)
    vocab_size = tokenizer.vocab_size
    tokenizer_json = tokenizer_dir() / "tokenizer.json"
    train_enc, val_enc, n_chars = _get_train_val_encoded(corpus_path, tokenizer, tokenizer_json)
    train_ds = TokenSeqDataset(train_enc, seq_len, stride=window_stride)
    val_ds = TokenSeqDataset(val_enc, seq_len, stride=window_stride)
    n_tokens = len(train_enc) + len(val_enc)
    log_print(
        f"Korpus: {SPEAKLEASH_DATASET}, {n_chars:,} znaków → {n_tokens:,} tokenów, "
        f"vocab={vocab_size}, stride={window_stride}, "
        f"train_windows={len(train_ds):,}, val_windows={len(val_ds):,}"
    )
    return train_ds, val_ds, tokenizer, vocab_size
