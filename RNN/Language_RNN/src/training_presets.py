"""Presety hparamów: PLGrid (big) + lokalne GPU ~6 GB."""

from __future__ import annotations

# Najlepszy LSTM vs najlepszy decoder-only transformer z obecnego zestawu.
LOCAL_BEST_LSTM = "rnnAWDLSTM"
LOCAL_BEST_TRANSFORMER = "transformer_miniLLaMA"

_LOCAL6G_BASE = dict(
    epochs=5,
    run_tag="local6G",
    eval_every=1,
    checkpoint_every=1,
    seq_len=256,
    window_stride=256,
)

LOCAL6G_AWD: dict = {
    **_LOCAL6G_BASE,
    "embed_dim": 384,
    "hidden_size": 384,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 8,
}

LOCAL6G_LLAMA: dict = {
    **_LOCAL6G_BASE,
    "embed_dim": 256,
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 4,
    "ff_size": 512,
    "dropout": 0.1,
    "batch_size": 4,
}

# Big picture (PLGrid) — tylko dokumentacja; wołaj train.main z tych dictów ręcznie.
GRID125M_RNN = dict(
    embed_dim=1536,
    hidden_size=1536,
    num_layers=4,
    dropout=0.3,
    seq_len=1024,
    window_stride=1024,
    batch_size=32,
    run_tag="125M",
)

GRID110M_LLAMA = dict(
    embed_dim=768,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    ff_size=2048,
    dropout=0.1,
    seq_len=1024,
    window_stride=1024,
    batch_size=16,
    run_tag="110M",
)
