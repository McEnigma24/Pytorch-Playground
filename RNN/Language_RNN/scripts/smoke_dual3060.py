"""Krótki smoke DUAL3060 (Lightning DDP). Uruchamiaj jako plik, nie przez stdin — spawn wymaga __file__."""

from __future__ import annotations

import os

import train
from training_presets import DUAL3060_ALL


def main() -> None:
    budget = float(os.environ.get("LANGUAGE_RNN_TIME_BUDGET_MIN", "1.5"))
    for architecture, preset in DUAL3060_ALL:
        p = dict(preset)
        p["time_budget_min"] = budget
        p["batch_size"] = max(2, p.get("batch_size", 8) // 4)
        print(f"\n=== smoke {architecture} ===", flush=True)
        train.main(tensorboard_dir=None, architecture=architecture, **p)


if __name__ == "__main__":
    main()