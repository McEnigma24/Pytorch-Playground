"""Model: embedding → głęboki LSTM (przyczynowy) → linear (opcjonalnie weight tying)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models._tie import maybe_tie_lm_head


class rnnBaseLSTM(nn.Module):

    @staticmethod
    def name() -> str:
        return "rnnBaseLSTM"

    @staticmethod
    def description() -> str:
        return "Klasyczny jednokierunkowy LSTM, opcjonalnie weight tying embedding↔głowa."

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.3,
        num_layers: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.weight_tying = False

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.weight_tying = maybe_tie_lm_head(self.embedding, self.fc)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out.reshape(-1, self.hidden_size))
        return logits, hidden

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        dev, dtype = weight.device, weight.dtype
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev, dtype=dtype)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev, dtype=dtype)
        return h, c
