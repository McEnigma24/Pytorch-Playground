"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
from datetime import datetime

from module import run_label
from packet.demo import log_demo_scalars


def main() -> None:
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
    run_dir = os.path.join(log_root, datetime.now().strftime("%Y%m%d-%H%M%S"))

    print(run_label())
    try:
        import torch

        print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_demo_scalars(run_dir)
        print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        print("Brak torch — uruchom na hoście: ./scripts/drun sync")


if __name__ == "__main__":
    main()
