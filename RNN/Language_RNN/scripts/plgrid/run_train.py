#!/usr/bin/env python3
"""Pojedynczy run treningu — pod SLURM array (preset smoke | grid125M | grid110M)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from log_print import tensorboard_run_dir, reset_log_timer  # noqa: E402
import train  # noqa: E402

ALL_ARCH = (
    "rnnRNN",
    "rnnGRU",
    "rnnLSTM",
    "rnnBaseLSTM",
    "rnnAWDLSTM",
    "rnnGoldenLSTM",
    "transformer_nanoGPT",
    "transformer_miniLLaMA",
    "transformer_GPT2small",
)

PRESETS: dict[str, dict] = {
    "smoke": dict(
        epochs=1,
        run_tag="smoke",
        eval_every=1,
        checkpoint_every=1,
        embed_dim=64,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ff_size=128,
        dropout=0.1,
        seq_len=64,
        window_stride=64,
        batch_size=8,
    ),
    "grid125M": dict(
        epochs=10,
        run_tag="125M",
        eval_every=1,
        checkpoint_every=1,
        embed_dim=1536,
        hidden_size=1536,
        num_layers=4,
        dropout=0.3,
        seq_len=1024,
        window_stride=1024,
        batch_size=32,
    ),
    # Jedna epoka — test na PL Grid (jak ręczne wywołania w src/main.py).
    "test125M_1ep": dict(
        epochs=1,
        run_tag="125M",
        eval_every=5,
        checkpoint_every=5,
        embed_dim=1536,
        hidden_size=1536,
        num_layers=4,
        dropout=0.3,
        seq_len=1024,
        window_stride=1024,
        batch_size=32,
    ),
    "grid110M_gpt2": dict(
        epochs=10,
        run_tag="110M",
        eval_every=1,
        checkpoint_every=1,
        embed_dim=768,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        ff_size=3072,
        dropout=0.1,
        seq_len=1024,
        window_stride=1024,
        batch_size=16,
    ),
    "grid110M_llama": dict(
        epochs=10,
        run_tag="110M",
        eval_every=1,
        checkpoint_every=1,
        embed_dim=768,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        ff_size=2048,
        dropout=0.1,
        seq_len=1024,
        window_stride=1024,
        batch_size=16,
    ),
    "grid110M_nano": dict(
        epochs=10,
        run_tag="110M",
        eval_every=1,
        checkpoint_every=1,
        embed_dim=1024,
        hidden_size=1024,
        num_layers=6,
        num_heads=16,
        ff_size=4096,
        dropout=0.1,
        seq_len=1024,
        window_stride=1024,
        batch_size=16,
    ),
}


def preset_for(architecture: str, name: str) -> dict:
    if name != "auto":
        return dict(PRESETS[name])
    if architecture.startswith("transformer_GPT2small"):
        return dict(PRESETS["grid110M_gpt2"])
    if architecture == "transformer_miniLLaMA":
        return dict(PRESETS["grid110M_llama"])
    if architecture.startswith("transformer_"):
        return dict(PRESETS["grid110M_nano"])
    return dict(PRESETS["grid125M"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=ALL_ARCH)
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", *PRESETS],
        help="auto: 125M RNN / 110M transformer wg architektury",
    )
    parser.add_argument("--tensorboard-root", default=os.environ.get("TENSORBOARD_LOGDIR", "/tmp/lrnn_runs"))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    reset_log_timer()
    kw = preset_for(args.architecture, args.preset)
    if args.epochs is not None:
        kw["epochs"] = args.epochs

    tb = tensorboard_run_dir(str(Path(args.tensorboard_root) / args.architecture))
    train.main(tensorboard_dir=tb, architecture=args.architecture, **kw)


if __name__ == "__main__":
    main()
