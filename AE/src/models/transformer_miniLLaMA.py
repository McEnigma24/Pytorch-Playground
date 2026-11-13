"""Model: embedding → RMSNorm + RoPE + SwiGLU (styl mini-LLaMA) → linear (tied).

Blok dekodera (pre-norm, jak LLaMA):
  tok_emb → [ Block × N ] → RMSNorm → lm_head

  Block:
    x → RMSNorm → CausalRoPEAttention → (+) → RMSNorm → SwiGLU → (+)

  CausalRoPEAttention — uwaga tylko wstecz (LM), Q/K z RoPE, KV cache przy inferencji.
  RMSNorm — normalizacja warstwowa (tańsza od LayerNorm).
  SwiGLU — FFN z bramką SiLU (dwa projekcje × bramka, potem w dół).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._rope import apply_rope


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm: skaluje aktywacje przez RMS (bez średniej).
    Stabilizuje trening; w LLaMA zamiennik LayerNorm (mniej operacji)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Feed-forward: SiLU(w1(x)) * w2(x), potem w3 — bramkowany FFN (np. PaLM/LLaMA)."""

    def __init__(self, n_embd: int, n_ff: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_ff, bias=False)
        self.w2 = nn.Linear(n_embd, n_ff, bias=False)
        self.w3 = nn.Linear(n_ff, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class CausalRoPEAttention(nn.Module):
    """Multi-head self-attention z maską przyczynową (tylko tokeny ≤ bieżący) i RoPE na Q/K.
    RoPE koduje pozycję w płaszczyźnie (sin/cos), bez stałego wektora pozycji.
    cache przechowuje dotychczasowe K/V przy autoregresyjnej generacji."""

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.size()
        qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, pos, self.head_dim)
        k = apply_rope(k, pos, self.head_dim)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_len = q.size(2)
        seq_k = k.size(2)
        mask = torch.tril(torch.ones(seq_k, seq_k, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask[-q_len:, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y)), new_cache


class Block(nn.Module):
    """Jeden transformer block (pre-norm): norm → attn → residual → norm → SwiGLU → residual."""

    def __init__(self, n_embd: int, n_head: int, n_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalRoPEAttention(n_embd, n_head, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, n_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        xa, cache = self.attn(self.norm1(x), cache, pos)
        x = x + xa
        x = x + self.mlp(self.norm2(x))
        return x, cache


def _round_ff_swiglu(n_embd: int, mult: float = 8 / 3) -> int:
    raw = int(n_embd * mult)
    return ((raw + 255) // 256) * 256


class transformer_miniLLaMA(nn.Module):

    @staticmethod
    def name() -> str:
        return "transformer_miniLLaMA"

    @staticmethod
    def description() -> str:
        return "Mini-LLaMA: RMSNorm pre-norm, RoPE, SwiGLU, tied embeddings, KV cache."

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
        n_ff = ff_size or _round_ff_swiglu(n_embd)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.num_layers = num_layers
        self.num_heads = n_head
        self.ff_size = n_ff
        self.block_size = seq_len
        self.dropout_p = dropout
        self.weight_tying = True

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, n_ff, dropout) for _ in range(num_layers))
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

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
        pos = torch.arange(pos_start, pos_start + t, device=x.device)
        h = self.drop(self.tok_emb(x))
        caches: list = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if hidden is None else hidden[i]
            h, layer_cache = block(h, layer_cache, pos)
            caches.append(layer_cache)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        return logits.reshape(-1, self.vocab_size), caches

    def init_hidden(self, batch_size: int) -> None:
        return None
