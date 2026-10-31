"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess

import infer
import old_stuff
import train
from log_print import log_print, reset_log_timer, tensorboard_run_dir

# experiment = "mnist"

# experiment = "rnnRNN"
experiment = "rnnLSTM"
# experiment = "rnnGRU"

# experiment = "transformer_nanoGPT"

# experiment = "infer_rnnLSTM"


default_run = False


def main() -> str:
    reset_log_timer()
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
    run_dir = tensorboard_run_dir(log_root)

    try:
        import torch

        log_print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        log_print("Brak torch — uruchom na hoście: ./scripts/drun sync")

    subprocess.run(["clear"], check=False)
    return run_dir

if __name__ == "__main__":
    run_dir = main()

    # Default run #
    if default_run == True:
        match experiment:

            case "mnist":
                old_stuff.main(tensorboard_dir=run_dir)

            case "rnnLSTM" | "rnnGRU" | "rnnRNN":
                train.main(tensorboard_dir=run_dir, architecture=experiment)

            case "infer_rnnLSTM" | "infer_rnnGRU" | "infer_rnnRNN":
                infer.main(architecture=experiment.removeprefix("infer_"))

    # Manual Stuff #


    

    train.main(tensorboard_dir=run_dir, architecture="rnnRNN")
    train.main(tensorboard_dir=run_dir, architecture="rnnLSTM")
    train.main(tensorboard_dir=run_dir, architecture="rnnGRU")

