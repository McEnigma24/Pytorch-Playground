"""Presety hparamów: PLGrid (big) + lokalne GPU 2× RTX 3060 12 GB."""

from __future__ import annotations

CHECKPOINT_PRUNE_POST_TRAINING = dict(
    checkpoint_prune_post_training_mod=10,
    checkpoint_prune_post_training_tail=5,
)

# Referencje do inferencji / compare (najlepsze z obecnego zestawu).
LOCAL_BEST_LSTM = "rnnAWDLSTM"
LOCAL_BEST_TRANSFORMER = "transformer_miniLLaMA"

_LOCAL6G_BASE = dict(
    epochs=8,
    run_tag="local6G",
    eval_every=1,
    checkpoint_every=1,
    **CHECKPOINT_PRUNE_POST_TRAINING,
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

# --- 2× RTX 3060, ~110M param, pełny korpus, 90 min / model (Lightning DDP) --- #

_DUAL3060_BASE = dict(
    run_tag="dual3060",
    epochs=999,   # time per model will limit this
    checkpoint_every=1,
    **CHECKPOINT_PRUNE_POST_TRAINING,
    eval_every=1,
    seq_len=512,
    window_stride=512,
    time_budget_min=90,
    lightning_devices=2,
)

_RNN_125M = dict(
    embed_dim=1536,
    hidden_size=1536,
    num_layers=4,
    dropout=0.2,
    batch_size=16,
    precision="16-mixed",
    learning_rate=1e-3,
)

DUAL3060_RNN: dict = {**_DUAL3060_BASE, **_RNN_125M}
DUAL3060_GRU: dict = {**_DUAL3060_BASE, **_RNN_125M}
DUAL3060_LSTM: dict = {**_DUAL3060_BASE, **_RNN_125M}
DUAL3060_AWD: dict = {**_DUAL3060_BASE, **_RNN_125M}

_TRANSFORMER_COMMON = dict(
    dropout=0.1,
    batch_size=12,
    precision="bf16-mixed",
    learning_rate=6e-4,
)

DUAL3060_LLAMA: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 768,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "ff_size": 2048,
}

DUAL3060_GPT2: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 768,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "ff_size": 3072,
}

DUAL3060_NANO: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 1024,
    "hidden_size": 1024,
    "num_layers": 6,
    "num_heads": 16,
    "ff_size": 4096,
}

DUAL3060_ALL: list[tuple[str, dict]] = [
    ("rnnRNN", DUAL3060_RNN),
    ("rnnGRU", DUAL3060_GRU),
    ("rnnLSTM", DUAL3060_LSTM),
    ("rnnAWDLSTM", DUAL3060_AWD),
    ("transformer_miniLLaMA", DUAL3060_LLAMA),
    ("transformer_GPT2small", DUAL3060_GPT2),
    ("transformer_nanoGPT", DUAL3060_NANO),
]

# Big picture (PLGrid) — tylko dokumentacja; wołaj train.main z tych dictów ręcznie.
# GRID125M_RNN = dict(
#     embed_dim=1536,
#     hidden_size=1536,
#     num_layers=4,
#     dropout=0.3,
#     seq_len=1024,
#     window_stride=1024,
#     batch_size=32,
#     run_tag="125M",
# )

# GRID110M_LLAMA = dict(
#     embed_dim=768,
#     hidden_size=768,
#     num_layers=12,
#     num_heads=12,
#     ff_size=2048,
#     dropout=0.1,
#     seq_len=1024,
#     window_stride=1024,
#     batch_size=16,
#     run_tag="110M",
# )