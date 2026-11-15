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

        self.encoder = nn.Sequential(
            # 28 * 28   784
            nn.Linear(self.image_size * self.image_size, 24 * 24),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(24 * 24, 20 * 20),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(20 * 20, 16 * 16),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(16 * 16, 12 * 12),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(12 * 12, 8 * 8),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(8 * 8, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 8 * 8),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(8 * 8, 12 * 12),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(12 * 12, 16 * 16),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(16 * 16, 20 * 20),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(20 * 20, 24 * 24),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            # 28 * 28   784
            nn.Linear(24 * 24, self.image_size * self.image_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b = x.size(0)
        x = x.flatten(1)  # (B, 1, 28, 28) → (B, 784)

        x = self.encoder(x)
        x = self.decoder(x)

        x = x.view(b, self.in_channels, self.image_size, self.image_size) # inflate

        return x


