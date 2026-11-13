"""Model: embedding + pozycje → pre-LN Transformer (GPT / lab01) → linear (tied)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.size()
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)
        seq_k = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_k, seq_k, device=x.device, dtype=torch.bool))
        q_len = q.size(2)
        att = att.masked_fill(~mask[-q_len:, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y)), new_cache


class MLP(nn.Module):
    def __init__(self, n_embd: int, n_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, n_ff)
        self.proj = nn.Linear(n_ff, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_ff: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, n_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        xa, cache = self.attn(self.ln1(x), cache, pos_start)
        x = x + xa
        x = x + self.mlp(self.ln2(x))
        return x, cache


class transformer_nanoGPT(nn.Module):

    @staticmethod
    def name() -> str:
        return "transformer_nanoGPT"

    @staticmethod
    def description() -> str:
        return "Decoder GPT (pre-LayerNorm, GELU FFN, learnable pozycje), KV cache przy generacji."

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_size: int | None = None,
        seq_len: int = 256,
    ):
        super().__init__()
        n_embd = hidden_size
        n_ff = ff_size or 4 * n_embd
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = n_ff
        self.block_size = seq_len
        self.dropout_p = dropout
        self.weight_tying = True

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(n_embd, num_heads, n_ff, seq_len, dropout) for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        positions = torch.arange(pos_start, pos_start + t, device=x.device).unsqueeze(0)
        x = self.drop(self.tok_emb(x) + self.pos_emb(positions))
        caches: list = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if hidden is None else hidden[i]
            x, layer_cache = block(x, layer_cache, pos_start)
            caches.append(layer_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits.reshape(-1, self.vocab_size), caches

    def init_hidden(self, batch_size: int) -> None:
        return None
