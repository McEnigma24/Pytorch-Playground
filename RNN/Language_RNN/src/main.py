"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess

import rnn_lstm
import simple_stuff
from log_print import log_print, reset_log_timer, tensorboard_run_dir

# experiment = "mnist"

# experiment = "rnn"
experiment = "rnn_lstm"
# experiment = "rnn_gru"

# experiment = "transformer_nanoGPT"


default_run = True



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
                simple_stuff.main(tensorboard_dir=run_dir)

            case "rnn_lstm":
                rnn_lstm.main(tensorboard_dir=run_dir)



    # Manual Stuff #
