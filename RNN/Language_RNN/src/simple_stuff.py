"""Pojedynczy moduł (plik) na poziomie src/ — import: from module import ..."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mnists import MNIST
from torchvision import datasets, transforms

from log_print import log_print, tensorboard_run_dir

# # Przełączamy model w tryb ewaluacji (wyłącza np. Dropout)
# model.eval()
# model.train()

# # Wyłączamy silnik Autograd
# with torch.no_grad():
#     # Ten kod wykona się zużywając ułamek pamięci
#     predictions = model(dane_testowe)
# Batch Norm
# Layer Norm
# Adam
# Softmax
# Weight Decay
# L1, L2 norm on Weights
# Norma w Warstwie, żeby nie przekraczała jakichś wartości
# Skip connections
# Training data augmentation
# Dropout
# Label Smoothing
# Gradient Clipping


# global_network="1 layer"
# global_network="Simple MLP, skip connections"
global_network = "LSTM"

# case "LSTM":
#     print("a")


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        # Datasets #
        self.dataset_prep()

        # Compute ENV #
        self.device = resolve_device(device)

        # Network #
        self.init_network(input_size, output_size)

        # Compute ENV - had to be at the end #
        self.to(self.device)

    def init_network(
        self,
        input_size: int,
        output_size: int,
    ):
        # Network Params #
        # aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        # można argmax, bez softMaxu -> to zmienia tylko skalę a nie to która
        # odpowiedź jest najlepsza
        self.cost_function = nn.CrossEntropyLoss()

        match global_network:
            case "1 layer":
                self.bn1 = nn.BatchNorm1d(input_size)
                self.layer1 = nn.Linear(input_size, output_size)
                # self.dropout1 = nn.Dropout(p=0.3)    # nie na końcu bo wtedy dropujemy
                # wiadomość wychodzącą xd

            case "Simple MLP, skip connections":
                # layer sizes #
                self.network_input_size = input_size
                self.network_output_size = output_size
                self.hidden_size = 100
                # self.hidden_size = 250
                self.stem_output_size = 100
                self.head_input_size = 100

                # STEM #
                self.stem_layer = nn.Linear(self.network_input_size, self.stem_output_size)
                self.stem_batchNorm = nn.BatchNorm1d(self.stem_output_size)
                self.stem_dropout = nn.Dropout(p=0.3)

                self.net = nn.Sequential(
                    nn.Linear(self.stem_output_size, self.hidden_size),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(self.hidden_size, self.head_input_size),
                    nn.BatchNorm1d(self.head_input_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )

                self.head_layer = nn.Linear(self.head_input_size, self.network_output_size)

                # BatchNorm się raczej tutaj nie robi, Dropout ucinało by nam ostateczne
                # klasy, czyli moglimyśmy nie móc wybrać poprawnej opcji #
                # self.head_batchNorm = nn.BatchNorm1d(self.network_output_size)
                # self.head_dropout = nn.Dropout(p=0.3)

            case "LSTM":
                print("a")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        match global_network:
            case "1 layer":
                x = self.bn1(x)
                x = self.layer1(x)
                x = self.dropout1(x)

            case "Simple MLP, skip connections":
                x = self.stem_layer(x)

                for i in range(5):
                    x = x + self.net(x)

                x = self.head_layer(x)

            case "LSTM":
                print("a")

        return x

    def dataset_prep(self) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),  # średnia/std na train MNIST
            ]
        )

        self.train_ds = datasets.MNIST(
    "/workspace/data/mnist",
    train=True,
    download=True,
     transform=transform)
        self.test_ds = datasets.MNIST(
    "/workspace/data/mnist",
    train=False,
    download=True,
     transform=transform)

    def _data_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        # num_workers>0: osobne procesy składają batch w tle (główna pętla mniej czeka na I/O).
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

    def _log_model_graph(self, writer: SummaryWriter, model: nn.Module) -> None:
        was_training = model.training
        model.eval()
        dummy_input = torch.randn(1, self.network_input_size, device=self.device)
        writer.add_graph(model, dummy_input)
        model.train(was_training)

    def _log_custom_scalars_layout(self, writer: SummaryWriter) -> None:
        writer.add_custom_scalars(
            {
                "train_and_eval": {
                    "accuracy": ["Multiline", ["train/accuracy", "eval/accuracy"]],
                    "loss": ["Multiline", ["train/loss", "eval/loss"]],
                },
            }
        )

    def trainn(
        self,
        model,
        epochs=50,
        batch_size=2048,
        writer: SummaryWriter | None = None,
    ):

        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_loader = self._data_loader(self.train_ds, batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            running_loss = torch.zeros((), device=self.device)
            correct = torch.zeros((), device=self.device, dtype=torch.long)
            total = 0

            for x, y in train_loader:
                x = torch.flatten(x, start_dim=1).to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(x)  # (B, 784) → (B, 10)
                loss = self.cost_function(logits, y)  # (B, 10)
                loss.backward()  # matrixes of corrections (gradients for each weight)
                optimizer.step()

                batch_n = y.size(0)
                running_loss += loss.detach() * batch_n
                correct += (logits.argmax(dim=1) == y).sum()
                total += batch_n

            avg_loss = (running_loss / total).item()
            correct_n = correct.item()
            accuracy = correct_n / total
            log_print(f"Epoka {epoch + 1}: loss={avg_loss:.4f} train_accuracy={accuracy:.2%}")
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/accuracy", accuracy, step)
            if epoch % 10 == 0 and epoch != 0:
                self.evall(model, writer=writer, epoch=epoch + 1)

    def evall(
        self,
        model,
        batch_size: int = 2048,
        writer: SummaryWriter | None = None,
        epoch: int | None = None,
    ) -> tuple[float, float]:
        # batch_size jak w treningu — bez tego DataLoader domyślnie daje batch=1 (wolno)
        model.to(self.device)
        test_loader = self._data_loader(self.test_ds, batch_size, shuffle=False)

        model.eval()
        running_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = torch.flatten(x, start_dim=1).to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = model(x)
                loss = self.cost_function(logits, y)

                batch_n = y.size(0)
                running_loss += loss * batch_n
                correct += (logits.argmax(dim=1) == y).sum()
                total += batch_n

        if total:
            avg_loss = (running_loss / total).item()
            correct_n = correct.item()
            accuracy = correct_n / total
        else:
            avg_loss = 0.0
            correct_n = 0
            accuracy = 0.0
        log_print(f"Test: loss={avg_loss:.4f}, eval_accuracy={accuracy:.2%} ({correct_n}/{total})")
        if writer is not None and epoch is not None:
            writer.add_scalar("eval/loss", avg_loss, epoch)
            writer.add_scalar("eval/accuracy", accuracy, epoch)
        return avg_loss, accuracy


def main(tensorboard_dir: str | None = None) -> None:
    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
        tensorboard_dir = tensorboard_run_dir(log_root)

    device = resolve_device()
    log_print(f"Urządzenie treningu: {device}")
    log_print(f"TensorBoard run: {tensorboard_dir}")

    net = SimpleNet(input_size=28 * 28, output_size=10, device=device)
    writer = SummaryWriter(tensorboard_dir)
    net._log_model_graph(writer, net)
    net._log_custom_scalars_layout(writer)
    epochs = 200
    try:
        net.trainn(net, epochs=epochs, writer=writer)
        net.evall(net, writer=writer, epoch=epochs)
    finally:
        writer.close()
