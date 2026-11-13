"""Smoke test: Metaspace decoder przywraca spacje po decode."""
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
TOK_JSON = ROOT / "src/text_and_tokenizer/data/tokenizer/tokenizer.json"


def main() -> None:
    tok = PreTrainedTokenizerFast(tokenizer_file=str(TOK_JSON))
    samples = [
        "W polskim tekście literatura.",
        "ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ",
    ]
    print("=== Round-trip encode -> decode ===")
    for s in samples:
        ids = tok.encode(s, add_special_tokens=False)
        back = tok.decode(ids, skip_special_tokens=True)
        status = "OK" if back == s else "FAIL"
        print(f"{status}: {s!r}")
        if back != s:
            print(f"     got: {back!r}")
        print(f"     pieces[:6]: {tok.convert_ids_to_tokens(ids)[:6]}")

    decoded = tok.decode(tok.encode("jedna druga", add_special_tokens=False))
    print()
    print("jedna druga ->", repr(decoded))
    if "\u2581" in decoded:
        raise SystemExit("FAIL: U+2581 (▁) nadal w decode")
    print("Spacje OK (brak ▁ w decode).")

    # Symulacja generate(): lista ID z seed + „wygenerowanych” tokenów
    seed = "W polskim"
    seed_ids = tok.encode(seed, add_special_tokens=False)
    extra = tok.encode(" tekście.", add_special_tokens=False)
    gen_ids = seed_ids + extra
    text = tok.decode(gen_ids, skip_special_tokens=True)
    print()
    print("=== Jak w generate() ===")
    print("seed + tokeny ->", repr(text))


if __name__ == "__main__":
    main()
