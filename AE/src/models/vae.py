"""Variational autoencoder — μ, log σ², reparametryzacja, ELBO."""

from __future__ import annotations

import torch
import torch.nn as nn


class vae(nn.Module):

    @staticmethod
    def name() -> str:
        return "vae"

    @staticmethod
    def description() -> str:
        return "VAE: encoder → (μ, logvar) → z → decoder, loss = recon + β·KL."

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 28,
        latent_dim: int = 16,
        kl_weight: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.in_channels = in_channels
        self.image_size = image_size
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vae.forward — rekonstrukcja (decode(reparam(encode(x))))")

    def loss_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError("vae.loss_parts — recon + kl")
