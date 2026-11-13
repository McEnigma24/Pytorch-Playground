"""DDPM — harmonogram szumu + ε_θ(x_t, t)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ddpm(nn.Module):

    @staticmethod
    def name() -> str:
        return "ddpm"

    @staticmethod
    def description() -> str:
        return "DDPM: forward diffusion + sieć predykująca szum; sampling osobno."

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 28,
        timesteps: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.dropout_p = dropout

    def training_loss(self, x0: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError("ddpm.training_loss — losuj t, ε, MSE(ε_θ, ε)")

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError("ddpm.sample — reverse diffusion")
