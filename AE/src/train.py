"""Trening autoenkoderów — ten sam układ co LanguageModelNet / SimpleNet w old_stuff.mnist."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models.denoising_ae import denoising_ae
from models.ddpm import ddpm
from models.model_report import build_model_report
from models.vae import vae
from models.vanilla_ae import vanilla_ae
from metrics.plot_training import plot_training_run
from metrics.run_metrics import RunMetricsLogger
from log_print import log_print, tensorboard_run_dir
from paths import (
    prepare_resumed_run,
    project_root,
    resolve_resume_from_env_or_arg,
    resolve_save_dir,
    resolve_source_run_dir,
)

AE_ARCHITECTURES = frozenset({"vanilla_ae", "denoising_ae", "vae", "ddpm"})
DIFFUSION_ARCHITECTURES = frozenset({"ddpm"})


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


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} h {minutes} min"


def _log_eta_from_first_batch(*, phase: str, batch_seconds: float, n_batches: int) -> None:
    if n_batches <= 1 or batch_seconds <= 0:
        return
    est_total = batch_seconds * n_batches
    log_print(
        f"  szac. {phase}: ~{_fmt_duration(est_total)} "
        f"(1. batch {batch_seconds:.2f}s × {n_batches:,} batchy)"
    )


def _nan_ppl() -> float:
    return float("nan")


class AutoencoderNet(nn.Module):
    def __init__(
        self,
        architecture: str,
        latent_dim: int = 32,
        image_size: int = 28,
        in_channels: int = 1,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        save_dir: Path | str | None = None,
        *,
        noise_std: float = 0.3,
        kl_weight: float = 1.0,
        timesteps: int = 1000,
        prepare_data: bool = True,
    ):
        super().__init__()

        self.architecture = architecture
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.dropout = dropout
        self.noise_std = noise_std
        self.kl_weight = kl_weight
        self.timesteps = timesteps
        self.model_name = architecture
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.metrics_logger: RunMetricsLogger | None = None
        self.train_ds = None
        self.val_ds = None

        if prepare_data:
            self.dataset_prep()

        self.device = resolve_device(device)
        self.init_network()
        self.to(self.device)

    def init_network(self) -> None:
        self.cost_function = nn.MSELoss()
        kw = {
            "in_channels": self.in_channels,
            "image_size": self.image_size,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
        }
        match self.architecture:
            case "vanilla_ae":
                self.ae = vanilla_ae(**kw)
            case "denoising_ae":
                self.ae = denoising_ae(**kw, noise_std=self.noise_std)
            case "vae":
                self.ae = vae(**kw, kl_weight=self.kl_weight)
            case "ddpm":
                self.ae = ddpm(**kw, timesteps=self.timesteps)
            case _:
                raise ValueError(f"Nieznana architektura: {self.architecture!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae(x)

    def _loss_on_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Zwraca (loss skalarny, liczba elementów do uśrednienia)."""
        if self.architecture == "ddpm":
            loss, _extra = self.ae.training_loss(x)
            n = x.numel()
            return loss, n
        if self.architecture == "vae":
            loss, _extra = self.ae.loss_parts(x)
            return loss, x.numel()
        if self.architecture == "denoising_ae":
            noisy = self.ae.corrupt(x)
            recon = self.ae(noisy)
            loss = self.cost_function(recon, x)
            return loss, x.numel()
        recon = self.ae(x)
        loss = self.cost_function(recon, x)
        return loss, x.numel()

    def dataset_prep(self) -> None:
        root = project_root() / "data" / "mnist"
        root.mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_ds = datasets.MNIST(str(root), train=True, download=True, transform=transform)
        self.val_ds = datasets.MNIST(str(root), train=False, download=True, transform=transform)

    def _data_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        pin_memory = self.device.type == "cuda"
        num_workers = 2 if pin_memory else 0
        kwargs: dict = {"batch_size": batch_size, "shuffle": shuffle, "pin_memory": pin_memory}
        if num_workers > 0:
            kwargs["num_workers"] = num_workers
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def _evall_once(
        self,
        model: AutoencoderNet,
        batch_size: int,
        writer: SummaryWriter | None,
        epoch: int | None,
    ) -> tuple[float, float]:
        model.to(self.device)
        val_loader = self._data_loader(self.val_ds, batch_size, shuffle=False)
        log_print(f"Walidacja: {len(val_loader):,} batchy (batch_size={batch_size})")

        model.eval()
        running_loss = torch.zeros((), device=self.device)
        total_elems = 0
        n_batches = len(val_loader)
        batch_idx = 0
        with torch.no_grad():
            for x, _y in val_loader:
                t_batch0 = time.perf_counter()
                x = x.to(self.device, non_blocking=True)
                loss, n = model._loss_on_batch(x)
                running_loss += loss.detach() * n
                total_elems += n
                _sync_device(self.device)
                batch_idx += 1
                if batch_idx == 1:
                    _log_eta_from_first_batch(
                        phase="walidacja",
                        batch_seconds=time.perf_counter() - t_batch0,
                        n_batches=n_batches,
                    )

        avg_loss = (running_loss / max(total_elems, 1)).item()
        log_print(f"Val: loss={avg_loss:.6f}")
        if writer is not None and epoch is not None:
            writer.add_scalar("eval/loss", avg_loss, epoch)
        return avg_loss, _nan_ppl()

    def _log_custom_scalars_layout(self, writer: SummaryWriter) -> None:
        writer.add_custom_scalars(
            {
                "train_and_eval": {
                    "loss": ["Multiline", ["train/loss", "eval/loss"]],
                }
            }
        )

    def _training_hparams(self) -> dict:
        return {
            "architecture": self.architecture,
            "model_name": self.model_name,
            "latent_dim": self.latent_dim,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "dropout": self.dropout,
            "noise_std": self.noise_std,
            "kl_weight": self.kl_weight,
            "timesteps": self.timesteps,
            "data_root": str(project_root() / "data" / "mnist"),
        }

    def _model_report(self) -> dict:
        desc_fn = getattr(type(self.ae), "description", None)
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
        model: AutoencoderNet,
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
        model: AutoencoderNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
        stem: str,
    ) -> Path:
        model.eval()
        root = self._ensure_save_dir()
        pt_path = root / f"{self.model_name}_{stem}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            pt_path,
        )
        log_print(f"Zapis {stem}: {pt_path.name}")
        return pt_path

    def save_checkpoint(
        self,
        model: AutoencoderNet,
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
        model: AutoencoderNet,
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
        )

    def preview_batch(self, model: AutoencoderNet, batch_size: int = 8) -> None:
        """Opcjonalny podgląd po ewaluacji — Ty możesz tu dodać zapis PNG."""
        loader = self._data_loader(self.val_ds, batch_size, shuffle=True)
        x, _ = next(iter(loader))
        x = x.to(self.device)
        try:
            with torch.no_grad():
                if self.architecture == "ddpm":
                    _ = model.ae.sample(min(8, x.size(0)), self.device)
                else:
                    _ = model(x)
        except NotImplementedError as exc:
            log_print(f"Podgląd pominięty (brak implementacji): {exc}")

    def trainn(
        self,
        model: AutoencoderNet,
        epochs: int = 30,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        max_grad_norm: float = 1.0,
        resume_payload: dict | None = None,
        checkpoint_every: int = 5,
        eval_every: int = 5,
        lr: float = 1e-3,
    ) -> tuple[torch.optim.Optimizer, float]:
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        start_epoch = 0
        best_eval_loss = float("inf")
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            start_epoch = int(resume_payload["epoch"])
            best_eval_loss = float(
                resume_payload.get("best_eval_loss", resume_payload.get("eval_loss", float("inf")))
            )
            log_print(
                f"Wczytano wagi + optimizer (po {start_epoch} epokach), "
                f"best_eval_loss={best_eval_loss:.6f}"
            )

        if start_epoch >= epochs:
            log_print(
                f"Checkpoint ma epokę {start_epoch} ≥ cel {epochs} — pomijam trening. "
                f"Ustaw więcej epok w train.main."
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
            total_elems = 0
            n_batches = len(train_loader)
            batch_idx = 0

            for x, _y in train_loader:
                t_batch0 = time.perf_counter()
                x = x.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss, n = model._loss_on_batch(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                running_loss += loss.detach() * n
                total_elems += n
                _sync_device(self.device)
                batch_idx += 1
                if batch_idx == 1:
                    _log_eta_from_first_batch(
                        phase="trening epoki",
                        batch_seconds=time.perf_counter() - t_batch0,
                        n_batches=n_batches,
                    )

            avg_loss = (running_loss / max(total_elems, 1)).item()
            train_seconds = time.perf_counter() - t_train0
            log_print(f"Epoka {epoch + 1}: loss={avg_loss:.6f} ({train_seconds:.1f}s train)")
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", avg_loss, step)

            t_eval0 = time.perf_counter()
            eval_loss, eval_ppl = self.evall(model, batch_size=batch_size, writer=writer, epoch=epoch + 1)
            eval_seconds = time.perf_counter() - t_eval0

            if self.metrics_logger is not None:
                self.metrics_logger.log_epoch(
                    epoch + 1,
                    avg_loss,
                    _nan_ppl(),
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
                )
                log_print(f"★ Nowy best eval: loss={eval_loss:.6f}")

            if epoch % checkpoint_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch + 1, eval_loss, eval_ppl, best_eval_loss)

            if epoch % eval_every == 0 or epoch == epochs - 1:
                self.preview_batch(model, batch_size=min(8, batch_size))

        return optimizer, best_eval_loss

    def evall(
        self,
        model: AutoencoderNet,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        epoch: int | None = None,
    ) -> tuple[float, float]:
        return self._evall_once(model, batch_size, writer, epoch)


def _net_from_hparams(
    hp: dict,
    device: torch.device,
    save_dir: Path,
    *,
    prepare_data: bool = True,
) -> AutoencoderNet:
    architecture = hp.get("architecture") or hp.get("model_name")
    if not architecture:
        raise ValueError("checkpoint hparams bez architecture / model_name")
    net = AutoencoderNet(
        architecture=architecture,
        latent_dim=hp.get("latent_dim", 32),
        image_size=hp.get("image_size", 28),
        in_channels=hp.get("in_channels", 1),
        dropout=hp.get("dropout", 0.0),
        device=device,
        save_dir=save_dir,
        noise_std=hp.get("noise_std", 0.3),
        kl_weight=hp.get("kl_weight", 1.0),
        timesteps=hp.get("timesteps", 1000),
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
) -> tuple[dict, AutoencoderNet]:
    path = Path(path)
    payload = load_checkpoint_payload(path)
    hp = payload["hparams"]
    dev = resolve_device(device)
    resolved_save = Path(save_dir) if save_dir is not None else checkpoint_run_dir(path)
    net = _net_from_hparams(hp, dev, resolved_save, prepare_data=False)
    net.dataset_prep()
    net.load_state_dict(payload["model_state_dict"])
    net.to(dev)
    return payload, net


def build_net_for_run(
    device: torch.device,
    save_dir: Path,
    checkpoint_path: Path | None,
    architecture: str,
    **model_kwargs,
) -> tuple[AutoencoderNet, dict | None]:
    if checkpoint_path is None:
        net = AutoencoderNet(architecture=architecture, device=device, save_dir=save_dir, **model_kwargs)
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
    latent_dim: int = 32,
    image_size: int = 28,
    in_channels: int = 1,
    dropout: float = 0.0,
    noise_std: float = 0.3,
    kl_weight: float = 1.0,
    timesteps: int = 1000,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    if architecture not in AE_ARCHITECTURES:
        raise ValueError(f"Nieznana architektura {architecture!r}. Dostępne: {sorted(AE_ARCHITECTURES)}")

    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "runs")
        tensorboard_dir = tensorboard_run_dir(log_root)

    device = resolve_device()
    log_print(f"Architektura: {architecture}")
    log_print(f"Urządzenie treningu: {device}")
    log_print(f"TensorBoard run: {tensorboard_dir}")

    resume_spec = resolve_resume_from_env_or_arg(resume)
    save_env = os.environ.get("AE_SAVE_DIR", "").strip() or os.environ.get("LANGUAGE_RNN_SAVE_DIR", "").strip()
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
        "latent_dim": latent_dim,
        "image_size": image_size,
        "in_channels": in_channels,
        "dropout": dropout,
        "noise_std": noise_std,
        "kl_weight": kl_weight,
        "timesteps": timesteps,
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
    metrics_logger.write_model_report(report, repr(net.ae))
    log_print(
        f"Model: {report['params']:,} param, {report['model_bytes']:,} B, "
        f"dtype={report['dtype']}"
    )
    net.metrics_logger = metrics_logger

    writer = SummaryWriter(tensorboard_dir)
    net._log_custom_scalars_layout(writer)
    ckpt_every = int(os.environ.get("AE_CHECKPOINT_EVERY", os.environ.get("LANGUAGE_RNN_CHECKPOINT_EVERY", str(checkpoint_every))))
    sample_every = int(os.environ.get("AE_EVAL_EVERY", os.environ.get("LANGUAGE_RNN_EVAL_EVERY", str(eval_every))))
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
            lr=lr,
        )
        eval_loss, eval_ppl = net.evall(net, batch_size=batch_size, writer=writer, epoch=epochs)
        net.save_final(net, optimizer, epochs, eval_loss, eval_ppl, best_eval_loss)
        train_total_seconds = time.perf_counter() - t_run0
        metrics_logger.write_timing(
            train_total_seconds=train_total_seconds,
            epochs=epochs,
            start_epoch=start_epoch,
        )
        plot_training_run(save_dir)
        log_print(f"Czas treningu: {train_total_seconds:.1f}s")
        log_print(f"Best eval loss w runie: {best_eval_loss:.6f} (pliki {architecture}_best_eval.*)")
        net.preview_batch(net)
    finally:
        writer.close()
