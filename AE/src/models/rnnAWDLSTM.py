"""Model: embedding → stacked LSTM + WeightDrop (DropConnect na weight_hh) → linear.

AWD-LSTM (Merity et al.): WeightDrop — w treningu losowo zeruje *całe wiersze*
macierzy wag rekurencyjnych (tu weight_hh_l0), nie aktywacje (to zwykły Dropout).
Cel: silniejsza regularyzacja niż sam dropout na hidden, mniejsze przeuczenie LSTM.

To NIE jest „weight decay” (L2 w optymalizatorze). Weight decay dodaje karę ‖w‖²
do loss (u nas AdamW może to robić osobno); WeightDrop maskuje wagi przed forwardem.

Ścieżka: embed → dropout → [ WeightDropLSTM + dropout ]×L → linear (opcjonalnie tied z embed).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._tie import maybe_tie_lm_head


class WeightDropLSTM(nn.Module):
    """Opakowanie LSTM: wagi z listy `weights` trzymane jako *_raw; w train() przed
    forwardem nakładany F.dropout na macierz wag (DropConnect na połączeniach hh)."""

    def __init__(self, module: nn.LSTM, weights: list[str], dropout: float) -> None:
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        for name_w in weights:
            raw = getattr(module, name_w)
            delattr(module, name_w)
            self.register_parameter(f"{name_w}_raw", nn.Parameter(raw.data))
        self.module.flatten_parameters = lambda: None

    def _setweights(self) -> None:
        # Zwykły tensor (nie Parameter) przez LSTM.__setattr__, żeby odświeżyć _flat_weights (cuDNN).
        for name_w in self.weights:
            raw: nn.Parameter = getattr(self, f"{name_w}_raw")
            if self.training and self.dropout > 0:
                w = F.dropout(raw, p=self.dropout, training=True)
            else:
                w = raw.view_as(raw)
            self.module._parameters.pop(name_w, None)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self._setweights()
            return self.module(*args)


class rnnAWDLSTM(nn.Module):
    """Kilka jednowarstwowych LSTM w kolejności (głębokość = liczba modułów),
    każdy z WeightDrop na wagach hidden→hidden; embed_dropout + dropout między warstwami."""

    @staticmethod
    def name() -> str:
        return "rnnAWDLSTM"

    @staticmethod
    def description() -> str:
        return "AWD-LSTM: WeightDrop na weight_hh, stos jednowarstwowych LSTM, weight tying opcjonalnie."

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.3,
        num_layers: int = 4,
        weight_drop: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.weight_drop = weight_drop
        self.weight_tying = False

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.input_proj = (
            nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else nn.Identity()
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            raw = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
            self.layers.append(WeightDropLSTM(raw, ["weight_hh_l0"], weight_drop))
        self.inter_drop = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers))

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.weight_tying = maybe_tie_lm_head(self.embedding, self.fc)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out = self.embed_dropout(self.embedding(x))
        out = self.input_proj(out)

        if hidden is None:
            states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        else:
            h, c = hidden
            states = [(h[i : i + 1], c[i : i + 1]) for i in range(self.num_layers)]

        new_h: list[torch.Tensor] = []
        new_c: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            out, state = layer(out, states[i])
            out = self.inter_drop[i](out)
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
