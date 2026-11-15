"""PyTorch Lightning — trening LanguageModelNet (2× GPU DDP)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR

from log_print import log_print
from metrics.run_metrics import RunMetricsLogger
from train import LanguageModelNet, _fmt_duration, _log_eta_from_first_batch, _sync_device


def _param_groups_no_decay(model: nn.Module, weight_decay: float) -> list[dict]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class LitLanguageModel(pl.LightningModule):
    def __init__(
        self,
        net: LanguageModelNet,
        *,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 500,
        total_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps or 10_000_000
        self.cost_function = net.cost_function
        self._tokens_per_step = 0
        self.save_hyperparameters(ignore=["net"])

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate") and getattr(self.net, "tokenizer", None) is None:
            self.net.tokenizer_prep()

    def forward(self, x: torch.Tensor, hidden=None, *, pos_start: int = 0):
        return self.net(x, hidden, pos_start=pos_start)

    def _batch_hidden(self, batch_n: int):
        return self.net._batch_hidden(batch_n)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        hidden = self._batch_hidden(x.size(0))
        logits, _ = self.net(x, hidden)
        loss = self.cost_function(logits, y.reshape(-1))
        self._tokens_per_step = y.numel()
        self.log("train/loss_step", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        hidden = self._batch_hidden(x.size(0))
        logits, _ = self.net(x, hidden)
        loss = self.cost_function(logits, y.reshape(-1))
        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            _param_groups_no_decay(self.net, weight_decay=0.1),
            lr=self.learning_rate,
        )
        total = max(self.total_steps, 1)
        warmup = min(self.warmup_steps, max(1, total // 20))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(warmup)
            if total <= warmup:
                return 1.0
            progress = (step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = total_steps

def _loss_from_batch_outputs(outputs) -> torch.Tensor | None:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        if isinstance(loss, torch.Tensor):
            return loss
    if hasattr(outputs, "loss") and isinstance(outputs.loss, torch.Tensor):
        return outputs.loss
    return None


def _epoch_train_loss(trainer: pl.Trainer, fallback_sum: float, fallback_n: int) -> float:
    cm = trainer.callback_metrics
    for key in ("train/loss_step_epoch", "train/loss_step"):
        v = cm.get(key)
        if v is not None:
            return float(v)
    return fallback_sum / max(1, fallback_n)


def _perplexity_for_log(loss: float) -> tuple[float, str]:
    capped = min(loss, 20.0)
    ppl = math.exp(capped)
    if loss > 20.0:
        return ppl, f"{ppl:.2f} (cap; loss={loss:.2f})"
    return ppl, f"{ppl:.2f}"


class TimeBudgetCallback(Callback):
    def __init__(self, budget_seconds: float, warmup_probe_steps: int = 50) -> None:
        super().__init__()
        self.budget_seconds = budget_seconds
        self.warmup_probe_steps = warmup_probe_steps
        self._t0: float | None = None
        self._planned_logged = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        self._t0 = time.perf_counter()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: LitLanguageModel, outputs, batch, batch_idx: int
    ) -> None:
        step = trainer.global_step
        if (
            trainer.is_global_zero
            and not self._planned_logged
            and step >= self.warmup_probe_steps
            and self._t0 is not None
        ):
            elapsed = time.perf_counter() - self._t0
            sps = step / max(elapsed, 1e-6)
            planned = max(step, int(self.budget_seconds * sps))
            pl_module.set_total_steps(planned)
            n_batch = max(1, trainer.num_training_batches)
            log_print(
                f"plan treningu: ~{planned:,} kroków ≈ {planned / n_batch:.1f} epok "
                f"w budżecie {_fmt_duration(self.budget_seconds)} ({sps:.2f} krok/s)"
            )
            self._planned_logged = True
        if self._t0 is not None and (time.perf_counter() - self._t0) >= self.budget_seconds:
            if trainer.is_global_zero:
                log_print(
                    f"Budżet czasu wyczerpany ({_fmt_duration(self.budget_seconds)}) — stop treningu"
                )
            trainer.should_stop = True


def _log_tb(trainer: pl.Trainer, metrics: dict[str, float], step: int) -> None:
    logger = trainer.logger
    if logger is None:
        return
    if hasattr(logger, "log_metrics"):
        logger.log_metrics(metrics, step=step)
    elif hasattr(logger, "experiment"):
        for k, v in metrics.items():
            logger.experiment.add_scalar(k, v, step)


class RunLogCallback(Callback):
    """Checkpoint co epokę, CSV co epokę; best_eval po walidacji (co eval_every)."""

    def __init__(
        self,
        *,
        checkpoint_every: int,
        eval_every: int,
        max_epochs_cap: int,
        metrics_save_dir: str | None = None,
        resume_best_eval: float = float("inf"),
    ) -> None:
        super().__init__()
        self.metrics_save_dir = metrics_save_dir
        self._metrics_logger: RunMetricsLogger | None = None
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.max_epochs_cap = max_epochs_cap
        self.best_eval_loss = resume_best_eval
        self._epoch_train_sum = 0.0
        self._epoch_train_n = 0
        self._t_train0 = 0.0
        self._t_eval0 = 0.0
        self._global_tokens = 0
        self._eta_logged = False
        self._last_eval_loss: float | None = None
        self._last_eval_ppl: float | None = None
        self._last_eval_seconds = 0.0
        self._val_this_epoch = False
        self.last_optimizer: torch.optim.Optimizer | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        if trainer.is_global_zero and self.metrics_save_dir and self._metrics_logger is None:
            self._metrics_logger = RunMetricsLogger(Path(self.metrics_save_dir))
            pl_module.net.metrics_logger = self._metrics_logger

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        if not trainer.is_global_zero:
            return
        self._val_this_epoch = False
        self._t_train0 = time.perf_counter()
        self._epoch_train_sum = 0.0
        self._epoch_train_n = 0
        self._eta_logged = False
        loader = trainer.train_dataloader
        n = len(loader) if loader is not None else 0
        bs = getattr(loader, "batch_size", "?")
        log_print(
            f"Epoka {trainer.current_epoch + 1}/{self.max_epochs_cap}: {n:,} batchy (batch_size={bs})"
        )

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: LitLanguageModel, outputs, batch, batch_idx: int
    ) -> None:
        loss = _loss_from_batch_outputs(outputs)
        if loss is not None:
            self._epoch_train_sum += loss.detach().float().item()
            self._epoch_train_n += 1
        self._global_tokens += pl_module._tokens_per_step * trainer.world_size
        if trainer.is_global_zero and batch_idx == 0 and not self._eta_logged:
            t0 = time.perf_counter()
            _sync_device(pl_module.device)
            _log_eta_from_first_batch(
                phase="trening epoki",
                batch_seconds=time.perf_counter() - t0,
                n_batches=len(trainer.train_dataloader),
            )
            self._eta_logged = True

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        if trainer.is_global_zero:
            self._t_eval0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        if not trainer.is_global_zero:
            return
        self._val_this_epoch = True
        eval_loss_t = trainer.callback_metrics.get("val_loss")
        if eval_loss_t is None:
            return
        eval_loss = float(eval_loss_t)
        eval_ppl, eval_ppl_s = _perplexity_for_log(eval_loss)
        self._last_eval_loss = eval_loss
        self._last_eval_ppl = eval_ppl
        self._last_eval_seconds = time.perf_counter() - self._t_eval0
        ep = trainer.current_epoch + 1
        log_print(f"Val: loss={eval_loss:.4f}, eval_perplexity={eval_ppl_s}")
        _log_tb(
            trainer,
            {"eval/loss": eval_loss, "eval/perplexity": eval_ppl},
            step=trainer.global_step,
        )

        net = pl_module.net
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            net.save_model_bundle(
                net,
                opt,
                ep,
                eval_loss,
                eval_ppl,
                self.best_eval_loss,
                stem="best_eval",
                export_onnx=True,
            )
            log_print(f"★ Nowy best eval: loss={eval_loss:.4f}, ppl={eval_ppl:.2f}")

        do_sample = ep % self.eval_every == 0 or ep == 1
        if do_sample:
            sample, _ = net.generate(net, seed="W", length=200, temperature=0.8)
            log_print(f"Generacja (200 tok.): {sample[:200]!r}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        if not trainer.is_global_zero:
            return
        train_seconds = time.perf_counter() - self._t_train0
        train_loss = _epoch_train_loss(trainer, self._epoch_train_sum, self._epoch_train_n)
        train_ppl, train_ppl_s = _perplexity_for_log(train_loss)
        ep = trainer.current_epoch + 1
        log_print(
            f"Epoka {ep}: loss={train_loss:.4f} train_perplexity={train_ppl_s} "
            f"({train_seconds:.1f}s train)"
        )
        _log_tb(
            trainer,
            {"train/loss": train_loss, "train/perplexity": train_ppl},
            step=trainer.global_step,
        )

        if self._val_this_epoch and self._last_eval_loss is not None:
            eval_loss = self._last_eval_loss
            eval_ppl = float(self._last_eval_ppl or math.exp(min(eval_loss, 20.0)))
            eval_seconds = self._last_eval_seconds
        else:
            eval_loss = float("nan")
            eval_ppl = float("nan")
            eval_seconds = 0.0

        if self._metrics_logger is not None:
            self._metrics_logger.log_epoch(
                float(ep),
                train_loss,
                train_ppl,
                eval_loss,
                eval_ppl,
                train_seconds,
                eval_seconds,
                step=trainer.global_step,
                tokens_seen=self._global_tokens,
            )

        opt = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        self.last_optimizer = opt
        ckpt_eval = float(self._last_eval_loss or self.best_eval_loss)
        ckpt_ppl = float(self._last_eval_ppl or math.exp(min(ckpt_eval, 20.0)))
        if ep % self.checkpoint_every == 0:
            pl_module.net.save_checkpoint(
                pl_module.net,
                opt,
                ep,
                ckpt_eval if ckpt_eval == ckpt_eval else 0.0,
                ckpt_ppl,
                self.best_eval_loss,
            )


class ResumeLitCallback(Callback):
    def __init__(self, payload: dict | None) -> None:
        super().__init__()
        self.payload = payload

    def on_fit_start(self, trainer: pl.Trainer, pl_module: LitLanguageModel) -> None:
        if self.payload is None:
            return
        pl_module.net.load_state_dict(self.payload["model_state_dict"])
        opt = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        try:
            opt.load_state_dict(self.payload["optimizer_state_dict"])
        except Exception:
            pass
        if trainer.is_global_zero:
            log_print(f"Lightning: wczytano resume (epoka {self.payload.get('epoch')})")


def default_lightning_precision(architecture: str) -> str:
    from train import TRANSFORMER_ARCHITECTURES

    if architecture in TRANSFORMER_ARCHITECTURES:
        return "bf16-mixed"
    return "16-mixed"