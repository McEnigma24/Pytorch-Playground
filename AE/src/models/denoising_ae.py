"""Denoising AE — zakłócone wejście, wyjście = czysty obraz."""

from __future__ import annotations

import torch
import torch.nn as nn


class denoising_ae(nn.Module):

    @staticmethod
    def name() -> str:
        return "denoising_ae"

    @staticmethod
    def description() -> str:
        return "Denoising AE: corrupt(x) → encoder → decoder → x_hat ≈ x."

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 28,
        latent_dim: int = 32,
        noise_std: float = 0.3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.in_channels = in_channels
        self.image_size = image_size
        self.dropout_p = dropout

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("denoising_ae.corrupt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("denoising_ae.forward")
