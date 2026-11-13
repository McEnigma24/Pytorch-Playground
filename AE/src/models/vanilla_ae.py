"""Vanilla autoencoder — TU implementujesz encoder + decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class vanilla_ae(nn.Module):

    @staticmethod
    def name() -> str:
        return "vanilla_ae"

    @staticmethod
    def description() -> str:
        return "Prosty AE: encoder → latent → decoder (MNIST 28×28)."

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 28,
        latent_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.image_size = image_size
        self.dropout_p = dropout

        # --- Twoja sieć ---
        # self.encoder = ...
        # self.decoder = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vanilla_ae.forward — zaimplementuj rekonstrukcję")
