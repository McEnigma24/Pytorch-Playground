"""Moduł w pakiecie packet — import: from packet.demo import ..."""

from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter


def log_demo_scalars(log_dir: str, steps: int = 10) -> str:
    writer = SummaryWriter(log_dir=log_dir)
    for step in range(steps):
        writer.add_scalar("demo/loss", 1.0 / (step + 1), step)
    writer.close()
    return log_dir
