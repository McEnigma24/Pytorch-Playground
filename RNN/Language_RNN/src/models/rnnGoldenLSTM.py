"""Model: embedding → spatial dropout → LSTM → pooling przyczynowy → linear."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._tie import maybe_tie_lm_head


class rnnGoldenLSTM(nn.Module):

    @staticmethod
    def name() -> str:
        return "rnnGoldenLSTM"

    @staticmethod
    def description() -> str:
        return "Spatial Dropout + LSTM + [h_t, cummax, cumavg] (przyczynowo), weight tying opcjonalnie."

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.3,
        num_layers: int = 1,
        spatial_dropout: float = 0.25,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.weight_tying = False

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.spatial_dropout = nn.Dropout1d(spatial_dropout)
        self.input_proj = (
            nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else nn.Identity()
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        concat_dim = hidden_size * 3
        self.fc1 = nn.Linear(concat_dim, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.weight_tying = maybe_tie_lm_head(self.embedding, self.fc)

    def _features(
        self,
        lstm_out: torch.Tensor,
        pool_max: torch.Tensor | None,
        pool_sum: torch.Tensor | None,
        pool_count: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """lstm_out: (B, T, H)."""
        b, t, _ = lstm_out.shape
        if t == 1 and pool_max is not None:
            step = lstm_out[:, 0, :]
            new_max = torch.maximum(pool_max, step)
            new_sum = pool_sum + step
            new_count = pool_count + 1.0
            cummean = new_sum / new_count
            feats = torch.cat([step, new_max, cummean], dim=-1).unsqueeze(1)
            return feats, new_max, new_sum, new_count

        cummax_vals, _ = lstm_out.cummax(dim=1)
        denom = torch.arange(1, t + 1, device=lstm_out.device, dtype=lstm_out.dtype).view(1, t, 1)
        cummean = lstm_out.cumsum(dim=1) / denom
        feats = torch.cat((lstm_out, cummax_vals, cummean), dim=-1)
        last_max = cummax_vals[:, -1, :]
        last_sum = lstm_out.sum(dim=1)
        last_count = lstm_out.new_full((b, 1), float(t))
        return feats, last_max, last_sum, last_count

    def forward(
        self,
        x: torch.Tensor,
        hidden: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        emb = self.spatial_dropout(emb)
        emb = emb.transpose(1, 2)
        emb = self.input_proj(emb)

        lstm_h = None
        pool_max = pool_sum = pool_count = None
        if hidden is not None:
            lstm_h = hidden.get("lstm")
            pool_max = hidden.get("pool_max")
            pool_sum = hidden.get("pool_sum")
            pool_count = hidden.get("pool_count")

        lstm_out, lstm_h = self.lstm(emb, lstm_h)
        feats, pool_max, pool_sum, pool_count = self._features(
            lstm_out, pool_max, pool_sum, pool_count
        )
        h = self.drop(self.relu(self.fc1(feats)))
        logits = self.fc(h.reshape(-1, h.size(-1)))

        new_hidden = {
            "lstm": lstm_h,
            "pool_max": pool_max,
            "pool_sum": pool_sum,
            "pool_count": pool_count,
        }
        return logits, new_hidden

    def init_hidden(self, batch_size: int) -> dict:
        weight = next(self.parameters()).data
        dev, dtype = weight.device, weight.dtype
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev, dtype=dtype)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev, dtype=dtype)
        return {
            "lstm": (h, c),
            "pool_max": torch.full((batch_size, self.hidden_size), float("-inf"), device=dev, dtype=dtype),
            "pool_sum": torch.zeros(batch_size, self.hidden_size, device=dev, dtype=dtype),
            "pool_count": torch.zeros(batch_size, 1, device=dev, dtype=dtype),
        }
