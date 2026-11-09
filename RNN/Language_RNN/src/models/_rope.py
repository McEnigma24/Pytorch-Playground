"""Rotary Position Embedding (RoPE) — q,k przed atencją."""

from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor, pos: torch.Tensor, head_dim: int) -> torch.Tensor:
    half = head_dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / half))
    angles = pos.to(dtype=x.dtype, device=x.device).reshape(-1, 1) * freqs.reshape(1, -1)
    # (T, half) → (1, 1, T, half) obok (B, H, T, half)
    while angles.dim() < 4:
        angles = angles.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
