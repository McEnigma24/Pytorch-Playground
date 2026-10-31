"""Model: embedding → stacked RNN (residual + LayerNorm + dropout) → linear."""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_STACKED_LAYERS = 3


class rnnRNN(nn.Module):

    @staticmethod
    def name() -> str:
        return "rnnRNN"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = NUM_STACKED_LAYERS
        self.dropout_p = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.input_proj = (
            nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else nn.Identity()
        )

        self.layers = nn.ModuleList(
            nn.RNN(hidden_size, hidden_size, num_layers=1, batch_first=True, nonlinearity="tanh")
            for _ in range(NUM_STACKED_LAYERS)
        )
        self.layer_norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(NUM_STACKED_LAYERS))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(NUM_STACKED_LAYERS))

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.embed_dropout(self.embedding(x))
        out = self.input_proj(out)

        if hidden is None:
            layer_states: list[torch.Tensor | None] = [None] * self.num_layers
        else:
            layer_states = [hidden[i : i + 1] for i in range(self.num_layers)]

        new_h: list[torch.Tensor] = []
        for i, rnn in enumerate(self.layers):
            residual = out
            out, state = rnn(out, layer_states[i])
            out = self.dropouts[i](out)
            out = self.layer_norms[i](out + residual)
            new_h.append(state)

        h_out = torch.cat(new_h, dim=0)
        logits = self.fc(out.reshape(-1, self.hidden_size))
        return logits, h_out

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
