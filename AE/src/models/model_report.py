"""Metadane modelu: parametry, bajty, warstwy, dropout."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _count_layer_types(module: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in module.modules():
        if child is module:
            continue
        name = child.__class__.__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def _dropout_info(module: nn.Module, configured: float) -> dict[str, Any]:
    drop_p: list[float] = []
    weight_drop: list[float] = []
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            drop_p.append(float(child.p))
        if child.__class__.__name__ == "WeightDropLSTM":
            weight_drop.append(float(getattr(child, "dropout", 0.0)))
    return {
        "configured": configured,
        "Dropout": drop_p,
        "WeightDrop": weight_drop,
    }


def build_model_report(
    module: nn.Module,
    *,
    architecture: str,
    description: str,
    configured_dropout: float,
) -> dict[str, Any]:
    core = module
    if hasattr(module, "ae"):
        core = module.ae
    elif hasattr(module, "rnn"):
        core = module.rnn

    state = core.state_dict()
    bytes_total = 0
    dtypes: set[str] = set()
    for t in state.values():
        bytes_total += t.numel() * t.element_size()
        dtypes.add(str(t.dtype).removeprefix("torch."))

    num_rnn_layers = int(getattr(core, "num_layers", 0))
    weight_tying = bool(getattr(core, "weight_tying", False))

    block_params: dict[str, int] = {}
    if hasattr(core, "embedding"):
        block_params["embedding"] = sum(p.numel() for p in core.embedding.parameters())
    if hasattr(core, "tok_emb"):
        block_params["tok_emb"] = sum(p.numel() for p in core.tok_emb.parameters())
    if hasattr(core, "lstm"):
        block_params["lstm"] = sum(p.numel() for p in core.lstm.parameters())
    if hasattr(core, "layers"):
        block_params["layers"] = sum(p.numel() for p in core.layers.parameters())
    if hasattr(core, "blocks"):
        block_params["blocks"] = sum(p.numel() for p in core.blocks.parameters())
    if hasattr(core, "fc"):
        block_params["fc"] = sum(p.numel() for p in core.fc.parameters())
    if hasattr(core, "lm_head"):
        block_params["lm_head"] = sum(p.numel() for p in core.lm_head.parameters())

    params_total = sum(p.numel() for p in core.parameters())

    return {
        "architecture": architecture,
        "description": description,
        "params": params_total,
        "model_bytes": bytes_total,
        "dtype": sorted(dtypes),
        "num_layers": num_rnn_layers,
        "layer_types": _count_layer_types(core),
        "dropout": _dropout_info(core, configured_dropout),
        "weight_tying": weight_tying,
        "block_params": block_params,
    }
