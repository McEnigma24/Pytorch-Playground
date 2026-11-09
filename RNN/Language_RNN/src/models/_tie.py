"""Powiązanie wag embedding ↔ głowa LM (gdy wymiary się zgadzają)."""

from __future__ import annotations

import torch.nn as nn


def maybe_tie_lm_head(embedding: nn.Embedding, head: nn.Linear) -> bool:
    if embedding.embedding_dim != head.in_features:
        return False
    if head.out_features != embedding.num_embeddings:
        return False
    head.weight = embedding.weight
    return True
