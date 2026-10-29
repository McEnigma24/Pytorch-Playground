"""Trening char-RNN — ten sam układ co SimpleNet w simple_stuff.py."""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from language_rnn.char_rnn import CharRNN
from language_rnn.text_data import (
    CharSeqDataset,
    build_vocab,
    encode_text,
    ensure_pan_tadeusz,
    load_corpus_text,
    split_encoded,
)
from log_print import log_print, tensorboard_run_dir


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharLanguageNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        seq_len: int = 100,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Datasets #
        self.dataset_prep()

        # Compute ENV #
        self.device = resolve_device(device)

        # Network #
        self.init_network()

        self.to(self.device)

    def init_network(self) -> None:
        self.cost_function = nn.CrossEntropyLoss()
        self.rnn = CharRNN(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.rnn(x, hidden)

    def dataset_prep(self) -> None:
        corpus_path = ensure_pan_tadeusz()
        text = load_corpus_text(corpus_path)
        self.char2idx, self.idx2char = build_vocab(text)
        self.vocab_size = len(self.idx2char)

        encoded = encode_text(text, self.char2idx)
        train_enc, val_enc = split_encoded(encoded, train_frac=0.9)

        self.train_ds = CharSeqDataset(train_enc, self.seq_len)
        self.val_ds = CharSeqDataset(val_enc, self.seq_len)

        log_print(
            f"Korpus: {len(text):,} znaków, vocab={self.vocab_size}, "
            f"train_windows={len(self.train_ds):,}, val_windows={len(self.val_ds):,}"
        )

    def _data_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        pin_memory = self.device.type == "cuda"
        num_workers = 2 if pin_memory else 0
        kwargs: dict = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            kwargs["num_workers"] = num_workers
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def _log_custom_scalars_layout(self, writer: SummaryWriter) -> None:
        writer.add_custom_scalars({
            "train_and_eval": {
                "loss": ["Multiline", ["train/loss", "eval/loss"]],
                "perplexity": ["Multiline", ["train/perplexity", "eval/perplexity"]],
            },
        })

    def trainn(
        self,
        model: CharLanguageNet,
        epochs: int = 30,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        train_loader = self._data_loader(self.train_ds, batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            running_loss = torch.zeros((), device=self.device)
            total_targets = 0

            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_n = x.size(0)

                hidden = model.rnn.init_hidden(batch_n)
                optimizer.zero_grad()
                logits, _hidden = model(x, hidden)
                loss = self.cost_function(logits, y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                n_targets = y.numel()
                running_loss += loss.detach() * n_targets
                total_targets += n_targets

            avg_loss = (running_loss / total_targets).item()
            perplexity = math.exp(min(avg_loss, 20.0))
            log_print(
                f"Epoka {epoch + 1}: loss={avg_loss:.4f} "
                f"train_perplexity={perplexity:.2f}"
            )
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/perplexity", perplexity, step)

            if epoch % 5 == 0 or epoch == epochs - 1:
                self.evall(model, batch_size=batch_size, writer=writer, epoch=epoch + 1)
                sample = self.generate(model, seed="Litwo", length=200, temperature=0.8)
                log_print(f"Generacja (200 zn.): {sample[:200]!r}")

    def evall(
        self,
        model: CharLanguageNet,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        epoch: int | None = None,
    ) -> tuple[float, float]:
        model.to(self.device)
        val_loader = self._data_loader(self.val_ds, batch_size, shuffle=False)

        model.eval()
        running_loss = torch.zeros((), device=self.device)
        total_targets = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_n = x.size(0)
                hidden = model.rnn.init_hidden(batch_n)
                logits, _ = model(x, hidden)
                loss = self.cost_function(logits, y.reshape(-1))
                n_targets = y.numel()
                running_loss += loss * n_targets
                total_targets += n_targets

        if total_targets:
            avg_loss = (running_loss / total_targets).item()
            perplexity = math.exp(min(avg_loss, 20.0))
        else:
            avg_loss = 0.0
            perplexity = 1.0

        log_print(
            f"Val: loss={avg_loss:.4f}, eval_perplexity={perplexity:.2f}"
        )
        if writer is not None and epoch is not None:
            writer.add_scalar("eval/loss", avg_loss, epoch)
            writer.add_scalar("eval/perplexity", perplexity, epoch)
        return avg_loss, perplexity

    def generate(
        self,
        model: CharLanguageNet,
        seed: str = "Litwo",
        length: int = 400,
        temperature: float = 1.0,
    ) -> str:
        model.eval()
        if not seed:
            seed = self.idx2char[0]

        chars: list[str] = []
        hidden = model.rnn.init_hidden(1)

        with torch.no_grad():
            for ch in seed:
                if ch not in self.char2idx:
                    continue
                idx = self.char2idx[ch]
                x = torch.tensor([[idx]], device=self.device, dtype=torch.long)
                logits, hidden = model(x, hidden)
                chars.append(ch)

            last_idx = self.char2idx.get(seed[-1], 0)
            for _ in range(length):
                x = torch.tensor([[last_idx]], device=self.device, dtype=torch.long)
                logits, hidden = model(x, hidden)
                logits = logits.squeeze(0) / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                last_idx = int(torch.multinomial(probs, num_samples=1).item())
                chars.append(self.idx2char[last_idx])

        return "".join(chars)


def main(tensorboard_dir: str | None = None) -> None:
    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
        tensorboard_dir = tensorboard_run_dir(log_root)

    device = resolve_device()
    log_print(f"Urządzenie treningu: {device}")
    log_print(f"TensorBoard run: {tensorboard_dir}")

    net = CharLanguageNet(device=device)
    writer = SummaryWriter(tensorboard_dir)
    net._log_custom_scalars_layout(writer)
    epochs = 30
    try:
        net.trainn(net, epochs=epochs, writer=writer)
        net.evall(net, writer=writer, epoch=epochs)
        final = net.generate(net, seed="Litwo, Ojczyzno moja!", length=500, temperature=0.85)
        log_print("--- Ostateczna generacja ---")
        log_print(final)
    finally:
        writer.close()
