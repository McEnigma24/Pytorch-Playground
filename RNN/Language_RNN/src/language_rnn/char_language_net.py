"""Trening char-RNN — ten sam układ co SimpleNet w simple_stuff.py."""

from __future__ import annotations

import copy
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from language_rnn.char_rnn import CharRNN
from language_rnn.text_data import (
    CharSeqDataset,
    encode_text,
    load_bpe_tokenizer,
    load_corpus_text,
    resolve_corpus_path,
    split_encoded,
)
from log_print import log_print, tensorboard_run_dir


DEFAULT_MODEL_NAME = "lstm"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_save_dir(model_name: str = DEFAULT_MODEL_NAME, when: datetime | None = None) -> Path:
    """Nowy katalog runu: saved_models/lstm-YYYY.MM.DD-HHMMSS (o ile nie podano SAVE_DIR)."""
    env = os.environ.get("LANGUAGE_RNN_SAVE_DIR")
    if env:
        root = Path(env)
    else:
        when = when or datetime.now()
        root = project_root() / "saved_models" / f"{model_name}-{when:%Y.%m.%d-%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def resolve_resume_checkpoint_path() -> Path | None:
    raw = os.environ.get("LANGUAGE_RNN_CHECKPOINT", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"LANGUAGE_RNN_CHECKPOINT nie istnieje: {path}")
    return path


def bootstrap_tokenizer_for_resume(save_dir: Path, checkpoint_path: Path) -> None:
    tok_dir = save_dir / "tokenizer"
    if tok_dir.is_dir() and any(tok_dir.iterdir()):
        return
    src = checkpoint_run_dir(checkpoint_path) / "tokenizer"
    if not src.is_dir():
        log_print(f"Brak tokenizera przy checkpointcie ({src}) — używę korpusu/BPE")
        return
    tok_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = tok_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    log_print(f"Tokenizer skopiowany: {src} → {tok_dir}")


def load_checkpoint_payload(path: Path | str) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharLanguageNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        seq_len: int = 100,
        device: torch.device | str | None = None,
        save_dir: Path | str | None = None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_name = DEFAULT_MODEL_NAME
        self.save_dir = Path(save_dir) if save_dir is not None else None

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
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor,torch.Tensor] | None = None)      -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.rnn(x, hidden)

    def dataset_prep(self) -> None:
        corpus_path = resolve_corpus_path()
        text = load_corpus_text(corpus_path)

        tok_dir = None
        if self.save_dir is not None:
            candidate = self.save_dir / "tokenizer"
            if candidate.is_dir() and any(candidate.iterdir()):
                tok_dir = candidate

        if tok_dir is not None:
            from transformers import PreTrainedTokenizerFast

            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
            log_print(f"Tokenizer z zapisu: {tok_dir}")
        else:
            self.tokenizer = load_bpe_tokenizer(corpus_path)

        self.vocab_size = self.tokenizer.vocab_size

        encoded = encode_text(text, self.tokenizer)
        train_enc, val_enc = split_encoded(encoded, train_frac=0.9)

        self.train_ds = CharSeqDataset(train_enc, self.seq_len)
        self.val_ds = CharSeqDataset(val_enc, self.seq_len)

        log_print(
            f"Korpus: {corpus_path.name}, {len(text):,} znaków → {len(encoded):,} tokenów, "
            f"vocab={self.vocab_size}, train_windows={len(self.train_ds):,}, val_windows={len(self.val_ds):,}"
        )

    def _data_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        pin_memory = self.device.type == "cuda"
        num_workers = 2 if pin_memory else 0
        kwargs: dict = {"batch_size": batch_size, "shuffle": shuffle, "pin_memory": pin_memory}
        if num_workers > 0:
            kwargs["num_workers"] = num_workers
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def _log_custom_scalars_layout(self, writer: SummaryWriter) -> None:
        writer.add_custom_scalars(
            {
                "train_and_eval": {
                    "loss": ["Multiline", ["train/loss", "eval/loss"]],
                    "perplexity": ["Multiline", ["train/perplexity", "eval/perplexity"]],
                }
            }
        )

    def _training_hparams(self) -> dict:
        return {
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
        }

    def _ensure_save_dir(self) -> Path:
        if self.save_dir is None:
            self.save_dir = resolve_save_dir(self.model_name)
            log_print(f"Zapis modeli: {self.save_dir}")
        return self.save_dir

    def _ensure_tokenizer_on_disk(self, root: Path) -> None:
        tok_dir = root / "tokenizer"
        if tok_dir.is_dir() and any(tok_dir.iterdir()):
            return
        tok_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(tok_dir)
        log_print(f"Tokenizer zapisany → {tok_dir}")

    def _checkpoint_payload(
        self,
        model: CharLanguageNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "best_eval_loss": best_eval_loss,
            "hparams": self._training_hparams(),
        }

    def save_model_bundle(
        self,
        model: CharLanguageNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
        stem: str,
        *,
        export_onnx: bool = True,
    ) -> Path:
        model.eval()
        root = self._ensure_save_dir()
        self._ensure_tokenizer_on_disk(root)
        pt_path = root / f"{self.model_name}_{stem}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            pt_path,
        )
        if export_onnx:
            self.export_to_onnx(model, root / f"{self.model_name}_{stem}.onnx")
        log_print(f"Zapis {stem}: {pt_path.name}" + (f", {self.model_name}_{stem}.onnx" if export_onnx else ""))
        return pt_path

    def save_checkpoint(
        self,
        model: CharLanguageNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> Path:
        root = self._ensure_save_dir()
        self._ensure_tokenizer_on_disk(root)
        path = root / "checkpoints" / f"epoch_{epoch:04d}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            path,
        )
        log_print(f"Checkpoint → {path}")
        return path

    def save_final(
        self,
        model: CharLanguageNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> None:
        self.save_model_bundle(
            model,
            optimizer,
            epoch,
            eval_loss,
            eval_perplexity,
            best_eval_loss,
            stem="final",
            export_onnx=True,
        )

    def export_to_onnx(self, model: CharLanguageNet, onnx_path: Path) -> None:
        """Jeden krok LSTM: token (B,1), h0, c0 → logity (B,V), h1, c1 (do inferencji)."""

        class RnnStep(nn.Module):
            def __init__(self, core: CharRNN) -> None:
                super().__init__()
                self.core = core

            def forward(
                self,
                token_ids: torch.Tensor,
                h0: torch.Tensor,
                c0: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                logits, (h1, c1) = self.core(token_ids, (h0, c0))
                return logits, h1, c1

        core = copy.deepcopy(model.rnn).cpu().eval()
        wrapper = RnnStep(core)
        batch = 1
        dummy_x = torch.zeros(batch, 1, dtype=torch.long)
        h0, c0 = core.init_hidden(batch)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            (dummy_x, h0, c0),
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input", "h0", "c0"],
            output_names=["logits", "h1", "c1"],
            dynamic_axes={
                "input": {0: "batch"},
                "h0": {1: "batch"},
                "c0": {1: "batch"},
                "logits": {0: "batch"},
                "h1": {1: "batch"},
                "c1": {1: "batch"},
            },
        )

    def trainn(
        self,
        model: CharLanguageNet,
        epochs: int = 30,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        max_grad_norm: float = 1.0,
        resume_payload: dict | None = None,
    ) -> tuple[torch.optim.Optimizer, float]:
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        start_epoch = 0
        best_eval_loss = float("inf")
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            start_epoch = int(resume_payload["epoch"])
            best_eval_loss = float(resume_payload.get("best_eval_loss", resume_payload.get("eval_loss", float("inf"))))
            log_print(
                f"Wczytano wagi + optimizer (po {start_epoch} epokach), "
                f"val_ppl={resume_payload.get('eval_perplexity', float('nan')):.2f}, "
                f"best_eval_loss={best_eval_loss:.4f}"
            )

        if start_epoch >= epochs:
            log_print(
                f"Checkpoint ma epokę {start_epoch} ≥ cel {epochs} — pomijam trening. "
                f"Ustaw więcej epok (np. LANGUAGE_RNN_EPOCHS=60)."
            )
            return optimizer, best_eval_loss

        train_loader = self._data_loader(self.train_ds, batch_size, shuffle=True)

        for epoch in range(start_epoch, epochs):
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
            log_print(f"Epoka {epoch + 1}: loss={avg_loss:.4f} train_perplexity={perplexity:.2f}")
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/perplexity", perplexity, step)

            eval_loss, eval_ppl = self.evall(model, batch_size=batch_size, writer=writer, epoch=epoch + 1)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model_bundle(
                    model,
                    optimizer,
                    epoch + 1,
                    eval_loss,
                    eval_ppl,
                    best_eval_loss,
                    stem="best_eval",
                    export_onnx=True,
                )
                log_print(f"★ Nowy best eval: loss={eval_loss:.4f}, ppl={eval_ppl:.2f}")

            if epoch % 5 == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch + 1, eval_loss, eval_ppl, best_eval_loss)
                sample = self.generate(model, seed="Litwo", length=200, temperature=0.8)
                log_print(f"Generacja (200 tok.): {sample[:200]!r}")

        return optimizer, best_eval_loss

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

        log_print(f"Val: loss={avg_loss:.4f}, eval_perplexity={perplexity:.2f}")
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
        model.to(self.device)
        tok = self.tokenizer
        seed_ids = tok.encode(seed or "", add_special_tokens=False)
        if not seed_ids:
            seed_ids = [tok.unk_token_id or 0]

        token_ids: list[int] = []
        hidden = model.rnn.init_hidden(1)

        with torch.no_grad():
            for idx in seed_ids:
                x = torch.tensor([[idx]], device=self.device, dtype=torch.long)
                logits, hidden = model(x, hidden)
                token_ids.append(idx)

            last_idx = seed_ids[-1]
            for _ in range(length):
                x = torch.tensor([[last_idx]], device=self.device, dtype=torch.long)
                logits, hidden = model(x, hidden)
                logits = logits.squeeze(0) / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                last_idx = int(torch.multinomial(probs, num_samples=1).item())
                token_ids.append(last_idx)

        return tok.decode(token_ids, skip_special_tokens=True)


def _net_from_hparams(hp: dict, device: torch.device, save_dir: Path) -> CharLanguageNet:
    net = CharLanguageNet(
        embed_dim=hp.get("embed_dim", 128),
        hidden_size=hp.get("hidden_size", 256),
        num_layers=hp.get("num_layers", 3),
        dropout=hp.get("dropout", 0.3),
        seq_len=hp.get("seq_len", 100),
        device=device,
        save_dir=save_dir,
    )
    net.model_name = hp.get("model_name", DEFAULT_MODEL_NAME)
    net.save_dir = save_dir
    return net


def load_checkpoint(
    path: Path | str,
    device: torch.device | str | None = None,
    *,
    save_dir: Path | str | None = None,
) -> tuple[dict, CharLanguageNet]:
    """Wczytuje checkpoint .pt — tokenizer z save_dir lub katalogu checkpointu."""
    path = Path(path)
    payload = load_checkpoint_payload(path)
    hp = payload["hparams"]
    dev = resolve_device(device)
    resolved_save = Path(save_dir) if save_dir is not None else checkpoint_run_dir(path)
    net = _net_from_hparams(hp, dev, resolved_save)
    net.load_state_dict(payload["model_state_dict"])
    net.to(dev)
    return payload, net


def build_net_for_run(
    device: torch.device,
    save_dir: Path,
    checkpoint_path: Path | None,
) -> tuple[CharLanguageNet, dict | None]:
    if checkpoint_path is None:
        net = CharLanguageNet(device=device, save_dir=save_dir)
        net.save_dir = save_dir
        return net, None

    bootstrap_tokenizer_for_resume(save_dir, checkpoint_path)
    payload = load_checkpoint_payload(checkpoint_path)
    log_print(f"Wznawiam z checkpointu: {checkpoint_path} (epoka {payload['epoch']})")
    net = _net_from_hparams(payload["hparams"], device, save_dir)
    return net, payload


def main(tensorboard_dir: str | None = None) -> None:
    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
        tensorboard_dir = tensorboard_run_dir(log_root)

    device = resolve_device()
    log_print(f"Urządzenie treningu: {device}")
    log_print(f"TensorBoard run: {tensorboard_dir}")

    ckpt_path = resolve_resume_checkpoint_path()
    save_env = os.environ.get("LANGUAGE_RNN_SAVE_DIR", "").strip()
    if ckpt_path is not None and not save_env:
        raise ValueError(
            "Wznowienie wymaga katalogu zapisu: ./start <checkpoint.pt> <save_dir>"
        )
    save_dir = Path(save_env) if save_env else resolve_save_dir(DEFAULT_MODEL_NAME)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_print(f"Katalog zapisu: {save_dir}")
    if ckpt_path:
        log_print(f"Checkpoint wejściowy: {ckpt_path}")

    net, resume_payload = build_net_for_run(device, save_dir, ckpt_path)

    writer = SummaryWriter(tensorboard_dir)
    net._log_custom_scalars_layout(writer)
    epochs = int(os.environ.get("LANGUAGE_RNN_EPOCHS", "30"))
    try:
        optimizer, best_eval_loss = net.trainn(net, epochs=epochs, writer=writer, resume_payload=resume_payload)
        eval_loss, eval_ppl = net.evall(net, writer=writer, epoch=epochs)
        net.save_final(net, optimizer, epochs, eval_loss, eval_ppl, best_eval_loss)
        log_print(f"Best eval loss w runie: {best_eval_loss:.4f} (pliki lstm_best_eval.*)")
        final = net.generate(net, seed="Litwo, Ojczyzno moja!", length=500, temperature=0.85)
        log_print("--- Ostateczna generacja (wagi final, nie best) ---")
        log_print(final)
    finally:
        writer.close()
