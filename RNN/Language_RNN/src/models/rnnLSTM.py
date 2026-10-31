"""Model: embedding → stacked LSTM (residual + LayerNorm + dropout) → linear."""

from __future__ import annotations

import torch
import torch.nn as nn


class rnnLSTM(nn.Module):

    @staticmethod
    def name() -> str:
        return "rnnLSTM"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.input_proj = (
            nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else nn.Identity()
        )

        self.layers = nn.ModuleList(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True) for _ in range(num_layers)
        )
        self.layer_norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers))

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out = self.embed_dropout(self.embedding(x))
        out = self.input_proj(out)

        if hidden is None:
            layer_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        else:
            h, c = hidden
            layer_states = [(h[i : i + 1], c[i : i + 1]) for i in range(self.num_layers)]

        new_h: list[torch.Tensor] = []
        new_c: list[torch.Tensor] = []
        for i, lstm in enumerate(self.layers):
            residual = out
            out, state = lstm(out, layer_states[i])
            out = self.dropouts[i](out)
            out = self.layer_norms[i](out + residual)
            new_h.append(state[0])
            new_c.append(state[1])

        h_out = torch.cat(new_h, dim=0)
        c_out = torch.cat(new_c, dim=0)
        logits = self.fc(out.reshape(-1, self.hidden_size))
        return logits, (h_out, c_out)

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        h = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        c = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        return h, c
