"""Inferencja — ostatni best_eval z saved_models/ + tokenizer z text_and_tokenizer."""

from __future__ import annotations

from pathlib import Path

from text_and_tokenizer import load_tokenizer, tokenizer_dir
from train import load_checkpoint, project_root
from log_print import log_print


def find_latest_run_dir(architecture: str, root: Path | None = None) -> Path:
    root = root or project_root() / "saved_models"
    if not root.is_dir():
        raise FileNotFoundError(f"Brak katalogu saved_models: {root}")

    prefix = f"{architecture}-"
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"Brak runów {prefix}* w {root} — najpierw wytrenuj (train.main architecture={architecture!r})"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_best_eval_pt(run_dir: Path, architecture: str) -> Path:
    path = run_dir / f"{architecture}_best_eval.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Brak {path.name} w {run_dir}")
    return path


def load_for_inference(
    architecture: str,
    checkpoint_path: Path | None = None,
    device: str | None = None,
):
    if checkpoint_path is None:
        run_dir = find_latest_run_dir(architecture)
        checkpoint_path = resolve_best_eval_pt(run_dir, architecture)
        log_print(f"Run: {run_dir.name}")

    tok = load_tokenizer()
    payload, net = load_checkpoint(checkpoint_path, device=device)
    if net.vocab_size != tok.vocab_size:
        raise RuntimeError(
            f"Tokenizer ({tok.vocab_size}) ≠ checkpoint ({net.vocab_size}). "
            f"Sprawdź {tokenizer_dir()}"
        )
    log_print(f"Wczytano: {checkpoint_path.name} (epoka {payload['epoch']}, ppl={payload.get('eval_perplexity', float('nan')):.2f})")
    return net, payload


def main(
    *,
    architecture: str,
    seed: str = "Litwo, Ojczyzno moja!",
    length: int = 500,
    temperature: float = 0.85,
    checkpoint_path: Path | str | None = None,
) -> str:
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    net, _payload = load_for_inference(architecture, checkpoint_path=ckpt)
    text = net.generate(net, seed=seed, length=length, temperature=temperature)
    log_print("--- Generacja (best_eval) ---")
    log_print(text)
    return text
