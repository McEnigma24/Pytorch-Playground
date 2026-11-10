"""Trening language RNN — ten sam układ co SimpleNet w old_stuff.mnist."""

from __future__ import annotations

import copy
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model_report import build_model_report
from models.rnnAWDLSTM import rnnAWDLSTM
from models.rnnBaseLSTM import rnnBaseLSTM
from models.rnnGoldenLSTM import rnnGoldenLSTM
from models.rnnGRU import rnnGRU
from models.rnnLSTM import rnnLSTM
from models.rnnRNN import rnnRNN
from models.transformer_GPT2small import transformer_GPT2small
from models.transformer_miniLLaMA import transformer_miniLLaMA
from models.transformer_nanoGPT import transformer_nanoGPT
from text_and_tokenizer import load_tokenizer, prepare_language_data, tokenizer_dir
from metrics.plot_training import plot_training_run
from metrics.run_metrics import RunMetricsLogger
from log_print import log_print, tensorboard_run_dir
from run_paths import (
    prepare_resumed_run,
    project_root,
    resolve_resume_from_env_or_arg,
    resolve_save_dir,
    resolve_source_run_dir,
)


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_checkpoint_payload(path: Path | str) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


TRANSFORMER_ARCHITECTURES = frozenset(
    {"transformer_nanoGPT", "transformer_miniLLaMA", "transformer_GPT2small"}
)

def _default_num_layers(architecture: str) -> int:
    match architecture:
        case "rnnBaseLSTM" | "rnnAWDLSTM":
            return 4
        case "rnnGoldenLSTM":
            return 1
        case "transformer_nanoGPT":
            return 6
        case "transformer_miniLLaMA" | "transformer_GPT2small":
            return 12
        case _:
            return 3


def _default_num_heads(hidden_size: int, num_heads: int | None) -> int:
    if num_heads is not None:
        return num_heads
    return max(1, hidden_size // 64)


class LanguageModelNet(nn.Module):
    def __init__(
        self,
        architecture: str,
        embed_dim: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.3,
        seq_len: int = 100,
        device: torch.device | str | None = None,
        save_dir: Path | str | None = None,
        *,
        num_layers: int | None = None,
        num_heads: int | None = None,
        ff_size: int | None = None,
        window_stride: int = 1,
        prepare_data: bool = True,
    ):
        super().__init__()

        self.architecture = architecture
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers if num_layers is not None else _default_num_layers(architecture)
        self.num_heads = _default_num_heads(hidden_size, num_heads)
        self.ff_size = ff_size
        self.window_stride = window_stride
        self.model_name = architecture
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.metrics_logger: RunMetricsLogger | None = None
        self.train_ds = None
        self.val_ds = None

        # Datasets (trening) albo sam tokenizer (inferencja) #
        if prepare_data:
            self.dataset_prep()
        else:
            self.tokenizer_prep()

        # Compute ENV #
        self.device = resolve_device(device)

        # Network #
        self.init_network()

        self.to(self.device)

    def init_network(self) -> None:
        self.cost_function = nn.CrossEntropyLoss()
        kwargs = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "num_layers": self.num_layers,
        }
        match self.architecture:
            case "rnnLSTM":
                self.rnn = rnnLSTM(**kwargs)
            case "rnnGRU":
                self.rnn = rnnGRU(**kwargs)
            case "rnnRNN":
                self.rnn = rnnRNN(**kwargs)
            case "rnnBaseLSTM":
                self.rnn = rnnBaseLSTM(**kwargs)
            case "rnnAWDLSTM":
                self.rnn = rnnAWDLSTM(**kwargs)
            case "rnnGoldenLSTM":
                self.rnn = rnnGoldenLSTM(**kwargs)
            case "transformer_nanoGPT":
                self.rnn = transformer_nanoGPT(
                    **kwargs,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    seq_len=self.seq_len,
                )
            case "transformer_miniLLaMA":
                self.rnn = transformer_miniLLaMA(
                    **kwargs,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    seq_len=self.seq_len,
                )
            case "transformer_GPT2small":
                self.rnn = transformer_GPT2small(
                    **kwargs,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    seq_len=self.seq_len,
                )
            case _:
                raise ValueError(f"Nieznana architektura: {self.architecture!r}")

    def forward(self, x: torch.Tensor, hidden=None, *, pos_start: int = 0):
        if self.architecture in TRANSFORMER_ARCHITECTURES:
            return self.rnn(x, hidden, pos_start=pos_start)
        return self.rnn(x, hidden)

    def tokenizer_prep(self) -> None:
        """Wczytuje gotowy BPE — bez korpusu i datasetów (inferencja / load_checkpoint)."""
        self.tokenizer = load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size

    def dataset_prep(self) -> None:
        self.train_ds, self.val_ds, self.tokenizer, self.vocab_size = prepare_language_data(
            self.seq_len,
            window_stride=self.window_stride,
        )

    def _data_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        pin_memory = self.device.type == "cuda"
        num_workers = 2 if pin_memory else 0
        kwargs: dict = {"batch_size": batch_size, "shuffle": shuffle, "pin_memory": pin_memory}
        if num_workers > 0:
            kwargs["num_workers"] = num_workers
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def _clamp_eval_batch_size(self, batch_size: int) -> int:
        """Ogranicza batch walidacji, gdy szczyt pamięci (gł. logits [B,S,V]) przekracza VRAM."""
        bs = max(1, int(batch_size))
        env_cap = os.environ.get("LANGUAGE_RNN_EVAL_BATCH_CAP", "").strip()
        if env_cap:
            try:
                bs = min(bs, max(1, int(env_cap)))
            except ValueError:
                pass

        if self.architecture not in TRANSFORMER_ARCHITECTURES or self.device.type != "cuda":
            return bs

        # fp32 logits + margines na CE / fragmentację
        bytes_per_batch = max(1, 8 * self.seq_len * self.vocab_size)
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return bs

        if free <= 0:
            return bs

        budget = int(free * 0.45)
        max_by_mem = max(1, budget // bytes_per_batch)
        if max_by_mem < bs:
            log_print(
                f"Eval batch: {bs} → {max_by_mem} "
                f"(seq={self.seq_len}, vocab={self.vocab_size:,}, wolne VRAM ~{free / 2**30:.2f} GiB)"
            )
            bs = max_by_mem
        return bs

    def _evall_once(
        self,
        model: LanguageModelNet,
        batch_size: int,
        writer: SummaryWriter | None,
        epoch: int | None,
    ) -> tuple[float, float]:
        model.to(self.device)
        val_loader = self._data_loader(self.val_ds, batch_size, shuffle=False)
        log_print(f"Walidacja: {len(val_loader):,} batchy (batch_size={batch_size})")

        model.eval()
        running_loss = torch.zeros((), device=self.device)
        total_targets = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_n = x.size(0)
                hidden = self._batch_hidden(batch_n)
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
        hp = {
            "architecture": self.architecture,
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_size": self.ff_size,
            "window_stride": self.window_stride,
            "vocab_size": self.vocab_size,
            "tokenizer_dir": str(tokenizer_dir()),
        }
        if self.architecture in TRANSFORMER_ARCHITECTURES:
            hp["ff_size"] = getattr(self.rnn, "ff_size", self.ff_size)
        return hp

    def _model_report(self) -> dict:
        desc_fn = getattr(type(self.rnn), "description", None)
        description = desc_fn() if callable(desc_fn) else ""
        return build_model_report(
            self,
            architecture=self.architecture,
            description=description,
            configured_dropout=self.dropout,
        )

    def _ensure_save_dir(self) -> Path:
        if self.save_dir is None:
            self.save_dir = resolve_save_dir(self.architecture)
            log_print(f"Zapis modeli: {self.save_dir}")
        return self.save_dir

    def _checkpoint_payload(
        self,
        model: LanguageModelNet,
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
        model: LanguageModelNet,
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
        pt_path = root / f"{self.model_name}_{stem}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            pt_path,
        )
        do_onnx = export_onnx and self.architecture == "rnnLSTM"
        if do_onnx:
            self.export_to_onnx(model, root / f"{self.model_name}_{stem}.onnx")
        elif export_onnx:
            log_print(f"ONNX pominięty dla {self.architecture}")
        log_print(f"Zapis {stem}: {pt_path.name}" + (f", {self.model_name}_{stem}.onnx" if do_onnx else ""))
        return pt_path

    def save_checkpoint(
        self,
        model: LanguageModelNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> Path:
        root = self._ensure_save_dir()
        path = root / "checkpoints" / f"epoch_{epoch:04d}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            path,
        )
        log_print(f"Checkpoint → {path}")
        return path

    def save_final(
        self,
        model: LanguageModelNet,
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

    def export_to_onnx(self, model: LanguageModelNet, onnx_path: Path) -> None:
        """Jeden krok LSTM: token (B,1), h0, c0 → logity (B,V), h1, c1 (do inferencji)."""

        class RnnStep(nn.Module):
            def __init__(self, core: rnnLSTM) -> None:
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

    def _batch_hidden(self, batch_n: int):
        if self.architecture in TRANSFORMER_ARCHITECTURES:
            return None
        match self.architecture:
            case "rnnLSTM" | "rnnBaseLSTM" | "rnnAWDLSTM":
                return self.rnn.init_hidden(batch_n)
            case "rnnGoldenLSTM":
                return self.rnn.init_hidden(batch_n)
            case "rnnGRU" | "rnnRNN":
                return self.rnn.init_hidden(batch_n)

    def trainn(
        self,
        model: LanguageModelNet,
        epochs: int = 30,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        max_grad_norm: float = 1.0,
        resume_payload: dict | None = None,
        checkpoint_every: int = 5,
        eval_every: int = 5,
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
                f"Ustaw więcej epok (np. epochs=60 w train.main)."
            )
            return optimizer, best_eval_loss

        train_loader = self._data_loader(self.train_ds, batch_size, shuffle=True)

        for epoch in range(start_epoch, epochs):
            log_print(
                f"Epoka {epoch + 1}/{epochs}: {len(train_loader):,} batchy (batch_size={batch_size})"
            )
            t_train0 = time.perf_counter()
            model.train()
            running_loss = torch.zeros((), device=self.device)
            total_targets = 0

            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_n = x.size(0)

                hidden = self._batch_hidden(batch_n)
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
            train_seconds = time.perf_counter() - t_train0
            perplexity = math.exp(min(avg_loss, 20.0))
            log_print(
                f"Epoka {epoch + 1}: loss={avg_loss:.4f} train_perplexity={perplexity:.2f} "
                f"({train_seconds:.1f}s train)"
            )
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/perplexity", perplexity, step)

            t_eval0 = time.perf_counter()
            eval_loss, eval_ppl = self.evall(model, batch_size=batch_size, writer=writer, epoch=epoch + 1)
            eval_seconds = time.perf_counter() - t_eval0

            if self.metrics_logger is not None:
                self.metrics_logger.log_epoch(
                    epoch + 1,
                    avg_loss,
                    perplexity,
                    eval_loss,
                    eval_ppl,
                    train_seconds,
                    eval_seconds,
                )

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

            if epoch % checkpoint_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch + 1, eval_loss, eval_ppl, best_eval_loss)

            if epoch % eval_every == 0 or epoch == epochs - 1:
                sample, _stats = self.generate(model, seed="W", length=200, temperature=0.8)
                log_print(f"Generacja (200 tok.): {sample[:200]!r}")

        return optimizer, best_eval_loss

    def evall(
        self,
        model: LanguageModelNet,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        epoch: int | None = None,
    ) -> tuple[float, float]:
        bs = self._clamp_eval_batch_size(batch_size)
        while True:
            try:
                return self._evall_once(model, bs, writer, epoch)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or bs <= 1:
                    raise
                torch.cuda.empty_cache()
                new_bs = max(1, bs // 2)
                log_print(
                    f"CUDA OOM przy walidacji (batch_size={bs}) — ponawiam z batch_size={new_bs}"
                )
                bs = new_bs

    def _perplexity_from_logits(
        self,
        logits_list: list[torch.Tensor],
        targets: list[int],
        temperature: float = 1.0,
    ) -> float:
        if not targets or len(logits_list) < len(targets):
            return float("nan")
        running = 0.0
        for logit, target in zip(logits_list[: len(targets)], targets):
            scaled = logit.squeeze(0) / max(temperature, 1e-6)
            loss = self.cost_function(
                scaled.unsqueeze(0),
                torch.tensor([target], device=scaled.device, dtype=torch.long),
            )
            running += loss.item()
        avg = running / len(targets)
        return math.exp(min(avg, 20.0))

    def generate(
        self,
        model: LanguageModelNet,
        seed: str = "W",
        length: int = 400,
        temperature: float = 1.0,
        *,
        measure: bool = True,
    ) -> tuple[str, dict]:
        model.eval()
        model.to(self.device)
        tok = self.tokenizer
        seed_ids = tok.encode(seed or "", add_special_tokens=False)
        if not seed_ids:
            seed_ids = [tok.unk_token_id or 0]

        token_ids: list[int] = []
        prefill_logits: list[torch.Tensor] = []
        gen_logits: list[torch.Tensor] = []
        prefill_seconds = 0.0
        generate_seconds = 0.0

        with torch.no_grad():
            if self.architecture in TRANSFORMER_ARCHITECTURES:
                pos_start = 0
                ids_tensor = torch.tensor([seed_ids], device=self.device, dtype=torch.long)
                if measure:
                    _sync_device(self.device)
                    t_prefill = time.perf_counter()
                window = ids_tensor[:, -self.seq_len :]
                pos_start = max(0, len(seed_ids) - window.size(1))
                logits_flat, cache = model(window, None, pos_start=pos_start)
                logits = logits_flat.view(window.size(0), window.size(1), -1)
                prefill_logits.extend(logits[:, i, :] for i in range(logits.size(1)))
                token_ids.extend(seed_ids)
                if measure:
                    _sync_device(self.device)
                    prefill_seconds = time.perf_counter() - t_prefill
                    _sync_device(self.device)
                    t_gen = time.perf_counter()
                logits = logits[:, -1, :]
                for _ in range(length):
                    gen_logits.append(logits)
                    scaled = logits.squeeze(0) / max(temperature, 1e-6)
                    last_idx = int(torch.multinomial(torch.softmax(scaled, dim=-1), 1).item())
                    token_ids.append(last_idx)
                    total_len = len(token_ids)
                    if total_len >= self.seq_len:
                        restart = max(self.seq_len // 2, 1)
                        chunk = torch.tensor([token_ids[-restart:]], device=self.device, dtype=torch.long)
                        pos_start = total_len - restart
                        logits_flat, cache = model(chunk, None, pos_start=pos_start)
                        logits = logits_flat.view(1, chunk.size(1), -1)[:, -1, :]
                    else:
                        step = torch.tensor([[last_idx]], device=self.device, dtype=torch.long)
                        pos_start = total_len - 1
                        logits_flat, cache = model(step, cache, pos_start=pos_start)
                        logits = logits_flat.view(1, 1, -1)[:, -1, :]
                if measure:
                    _sync_device(self.device)
                    generate_seconds = time.perf_counter() - t_gen
            else:
                match self.architecture:
                    case "rnnGRU" | "rnnRNN" | "rnnLSTM" | "rnnBaseLSTM" | "rnnAWDLSTM" | "rnnGoldenLSTM":
                        hidden = model.rnn.init_hidden(1)
                        if measure:
                            _sync_device(self.device)
                            t_prefill = time.perf_counter()
                        # Prefill #
                        for idx in seed_ids:
                            x = torch.tensor([[idx]], device=self.device, dtype=torch.long)
                            logits, hidden = model(x, hidden)
                            prefill_logits.append(logits)
                            token_ids.append(idx)
                        if measure:
                            _sync_device(self.device)
                            prefill_seconds = time.perf_counter() - t_prefill
                            _sync_device(self.device)
                            t_gen = time.perf_counter()
                        # Generate #
                        for _ in range(length):
                            gen_logits.append(logits)
                            scaled = logits.squeeze(0) / max(temperature, 1e-6)
                            probs = torch.softmax(scaled, dim=-1)
                            last_idx = int(torch.multinomial(probs, num_samples=1).item())
                            token_ids.append(last_idx)
                            x = torch.tensor([[last_idx]], device=self.device, dtype=torch.long)
                            logits, hidden = model(x, hidden)
                        if measure:
                            _sync_device(self.device)
                            generate_seconds = time.perf_counter() - t_gen
                    case _:
                        raise ValueError(f"Generacja nieobsługiwana: {self.architecture!r}")

        prompt_targets = seed_ids[1:]
        prompt_logits = prefill_logits[:-1] if len(prefill_logits) > 1 else []
        generated_targets = token_ids[len(seed_ids) :]

        stats = {
            "prompt_tokens": len(seed_ids),
            "generated_tokens": length,
            "prefill_seconds": prefill_seconds,
            "generate_seconds": generate_seconds,
            "inference_seconds": prefill_seconds + generate_seconds,
            "tokens_per_second": (
                length / generate_seconds if measure and generate_seconds > 0 else float("nan")
            ),
            "prompt_perplexity": self._perplexity_from_logits(prompt_logits, prompt_targets, 1.0),
            "generate_perplexity": self._perplexity_from_logits(gen_logits, generated_targets, 1.0),
            "sampling_perplexity": self._perplexity_from_logits(
                gen_logits, generated_targets, temperature
            ),
            "temperature": temperature,
        }
        text = tok.decode(token_ids, skip_special_tokens=True)
        return text, stats


def _net_from_hparams(
    hp: dict,
    device: torch.device,
    save_dir: Path,
    *,
    prepare_data: bool = True,
) -> LanguageModelNet:
    architecture = hp.get("architecture") or hp.get("model_name")
    if not architecture:
        raise ValueError("checkpoint hparams bez architecture / model_name")
    net = LanguageModelNet(
        architecture=architecture,
        embed_dim=hp.get("embed_dim", 128),
        hidden_size=hp.get("hidden_size", 256),
        dropout=hp.get("dropout", 0.3),
        seq_len=hp.get("seq_len", 100),
        device=device,
        save_dir=save_dir,
        num_layers=hp.get("num_layers"),
        num_heads=hp.get("num_heads"),
        ff_size=hp.get("ff_size"),
        window_stride=hp.get("window_stride", 1),
        prepare_data=prepare_data,
    )
    net.model_name = hp.get("model_name", architecture)
    net.save_dir = save_dir
    return net


def load_checkpoint(
    path: Path | str,
    device: torch.device | str | None = None,
    *,
    save_dir: Path | str | None = None,
) -> tuple[dict, LanguageModelNet]:
    """Wczytuje checkpoint .pt — tokenizer z text_and_tokenizer/data/tokenizer."""
    path = Path(path)
    payload = load_checkpoint_payload(path)
    hp = payload["hparams"]
    dev = resolve_device(device)
    resolved_save = Path(save_dir) if save_dir is not None else checkpoint_run_dir(path)
    net = _net_from_hparams(hp, dev, resolved_save, prepare_data=False)
    if net.vocab_size != hp.get("vocab_size", net.vocab_size):
        raise RuntimeError(
            f"vocab checkpoint ({hp.get('vocab_size')}) ≠ tokenizer ({net.vocab_size}) — "
            f"retrening tokenizera w {tokenizer_dir()}"
        )
    net.load_state_dict(payload["model_state_dict"])
    net.to(dev)
    return payload, net


def build_net_for_run(
    device: torch.device,
    save_dir: Path,
    checkpoint_path: Path | None,
    architecture: str,
    **model_kwargs,
) -> tuple[LanguageModelNet, dict | None]:
    if checkpoint_path is None:
        net = LanguageModelNet(architecture=architecture, device=device, save_dir=save_dir, **model_kwargs)
        net.save_dir = save_dir
        return net, None

    payload = load_checkpoint_payload(checkpoint_path)
    log_print(f"Wznawiam z checkpointu: {checkpoint_path} (epoka {payload['epoch']})")
    net = _net_from_hparams(payload["hparams"], device, save_dir)
    return net, payload


def main(
    tensorboard_dir: str | None = None,
    *,
    architecture: str,
    epochs: int = 30,
    checkpoint_every: int = 5,
    eval_every: int = 5,
    run_tag: str | None = None,
    resume: str | None = None,
    embed_dim: int = 128,
    hidden_size: int = 256,
    num_layers: int | None = None,
    num_heads: int | None = None,
    ff_size: int | None = None,
    dropout: float = 0.3,
    seq_len: int = 100,
    window_stride: int = 1,
    batch_size: int = 128,
) -> None:
    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
        tensorboard_dir = tensorboard_run_dir(log_root)

    device = resolve_device()
    log_print(f"Architektura: {architecture}")
    log_print(f"Urządzenie treningu: {device}")
    log_print(f"TensorBoard run: {tensorboard_dir}")

    resume_spec = resolve_resume_from_env_or_arg(resume)
    save_env = os.environ.get("LANGUAGE_RNN_SAVE_DIR", "").strip()
    resumed_from: dict[str, str] | None = None
    ckpt_path: Path | None = None

    if save_env:
        save_dir = Path(save_env)
    else:
        save_dir = resolve_save_dir(architecture, run_tag=run_tag)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if resume_spec:
        source_run = resolve_source_run_dir(architecture, resume_spec)
        log_print(f"Wznawiam z runu: {source_run}")
        ckpt_path = prepare_resumed_run(source_run, save_dir, architecture)
        resumed_from = {"run_dir": str(source_run), "checkpoint": str(ckpt_path)}

    log_print(f"Katalog zapisu: {save_dir}")
    if ckpt_path:
        log_print(f"Checkpoint wejściowy: {ckpt_path}")

    model_kwargs = {
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_size": ff_size,
        "window_stride": window_stride,
    }
    net, resume_payload = build_net_for_run(device, save_dir, ckpt_path, architecture, **model_kwargs)

    metrics_logger = RunMetricsLogger(save_dir)
    metrics_logger.write_meta(
        architecture=architecture,
        run_id=save_dir.name,
        tensorboard_dir=tensorboard_dir,
        hparams=net._training_hparams(),
        resumed_from=resumed_from,
    )
    report = net._model_report()
    metrics_logger.write_model_report(report, repr(net.rnn))
    log_print(
        f"Model: {report['params']:,} param, {report['model_bytes']:,} B, "
        f"tying={report['weight_tying']}, dtype={report['dtype']}"
    )
    net.metrics_logger = metrics_logger

    writer = SummaryWriter(tensorboard_dir)
    net._log_custom_scalars_layout(writer)
    ckpt_every = int(os.environ.get("LANGUAGE_RNN_CHECKPOINT_EVERY", str(checkpoint_every)))
    sample_every = int(os.environ.get("LANGUAGE_RNN_EVAL_EVERY", str(eval_every)))
    start_epoch = int(resume_payload["epoch"]) if resume_payload else 0
    try:
        t_run0 = time.perf_counter()
        optimizer, best_eval_loss = net.trainn(
            net,
            epochs=epochs,
            batch_size=batch_size,
            writer=writer,
            resume_payload=resume_payload,
            checkpoint_every=ckpt_every,
            eval_every=sample_every,
        )
        eval_loss, eval_ppl = net.evall(
            net, batch_size=batch_size, writer=writer, epoch=epochs
        )
        net.save_final(net, optimizer, epochs, eval_loss, eval_ppl, best_eval_loss)
        train_total_seconds = time.perf_counter() - t_run0
        metrics_logger.write_timing(
            train_total_seconds=train_total_seconds,
            epochs=epochs,
            start_epoch=start_epoch,
        )
        plot_training_run(save_dir)
        log_print(f"Czas treningu: {train_total_seconds:.1f}s")
        log_print(f"Best eval loss w runie: {best_eval_loss:.4f} (pliki {architecture}_best_eval.*)")
        final, _stats = net.generate(net, seed="W polskim", length=500, temperature=0.85)
        log_print("--- Ostateczna generacja (wagi final, nie best) ---")
        log_print(final)
    finally:
        writer.close()
