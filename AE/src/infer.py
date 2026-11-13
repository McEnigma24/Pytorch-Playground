"""Inferencja — checkpoint z saved_models/ + podgląd rekonstrukcji / sample."""

from __future__ import annotations

from pathlib import Path

from train import checkpoint_run_dir, load_checkpoint, resolve_device
from paths import find_latest_run_dir
from log_print import log_print


def resolve_best_eval_pt(run_dir: Path, architecture: str) -> Path:
    path = run_dir / f"{architecture}_best_eval.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Brak {path.name} w {run_dir}")
    return path


def main(
    *,
    architecture: str = "vanilla_ae",
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> None:
    device = resolve_device()

    if checkpoint_path is None:
        if run_dir is None:
            run_dir = find_latest_run_dir(architecture)
        else:
            run_dir = Path(run_dir)
        checkpoint_path = resolve_best_eval_pt(run_dir, architecture)
    else:
        checkpoint_path = Path(checkpoint_path)
        run_dir = checkpoint_run_dir(checkpoint_path)

    log_print(f"Inferencja {architecture}: {checkpoint_path}")
    _payload, net = load_checkpoint(checkpoint_path, device=device, save_dir=run_dir)
    net.preview_batch(net, batch_size=16)
    log_print("Dodaj zapis PNG w preview_batch / save_inference_artifacts w modelu.")
