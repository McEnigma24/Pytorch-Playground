"""Model: embedding → LSTM → linear (następny znak)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. Embedding: indeks znaku → wektor cech
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. LSTM — batch_first: (B, T, features)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
        )

        # 3. Logity nad słownikiem znaków
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out = self.embedding(x)
        out, hidden = self.lstm(out, hidden)
        out = out.reshape(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        device = weight.device
        h = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        c = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        return h, c
