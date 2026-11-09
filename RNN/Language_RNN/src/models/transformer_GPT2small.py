"""Model: embedding + pozycje → GPT-2 Small (pre-LN, new GELU) → linear (tied)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, residual_scale: float, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.residual_scale = residual_scale
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.size()
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_len = q.size(2)
        seq_k = k.size(2)
        mask = torch.tril(torch.ones(seq_k, seq_k, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask[-q_len:, :], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y)) * self.residual_scale
        return y, new_cache


class MLP(nn.Module):
    def __init__(self, n_embd: int, n_ff: int, residual_scale: float, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, n_ff)
        self.proj = nn.Linear(n_ff, n_embd)
        self.residual_scale = residual_scale
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(new_gelu(self.fc(x)))) * self.residual_scale


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_ff: int, residual_scale: float, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, residual_scale, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, n_ff, residual_scale, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        xa, cache = self.attn(self.ln1(x), cache)
        x = x + xa
        x = x + self.mlp(self.ln2(x))
        return x, cache


class transformer_GPT2small(nn.Module):

    @staticmethod
    def name() -> str:
        return "transformer_GPT2small"

    @staticmethod
    def description() -> str:
        return "Klon GPT-2 Small: pre-LN, learnable pozycje, new GELU, skalowanie residuali."

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
        num_layers: int = 12,
        num_heads: int | None = None,
        ff_size: int | None = None,
        seq_len: int = 1024,
    ):
        super().__init__()
        n_embd = hidden_size
        n_head = num_heads or max(1, n_embd // 64)
        n_ff = ff_size or 4 * n_embd
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.num_layers = num_layers
        self.num_heads = n_head
        self.ff_size = n_ff
        self.block_size = seq_len
        self.dropout_p = dropout
        self.weight_tying = True
        res_scale = 1.0 / math.sqrt(2 * num_layers)

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(n_embd, n_head, n_ff, res_scale, dropout) for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: list | None = None,
        *,
        pos_start: int = 0,
    ) -> tuple[torch.Tensor, list]:
        b, t = x.size()
        if t > self.block_size:
            x = x[:, -self.block_size :]
            t = x.size(1)
            pos_start = max(0, pos_start + (t - self.block_size))
        pos = torch.arange(pos_start, pos_start + t, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        caches: list = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if hidden is None else hidden[i]
            h, layer_cache = block(h, layer_cache)
            caches.append(layer_cache)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits.reshape(-1, self.vocab_size), caches

    def init_hidden(self, batch_size: int) -> None:
        return None
