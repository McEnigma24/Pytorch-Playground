"""Domyślny punkt wejścia: run_main (styl Language_RNN, eksperymenty AE)."""

from __future__ import annotations

import os
import subprocess

import infer
import old_stuff
import train
from log_print import log_print, reset_log_timer, tensorboard_run_dir

# experiment = "simple_mlp"

experiment = "vanilla_ae"
# experiment = "denoising_ae"
# experiment = "vae"
# experiment = "ddpm"

# experiment = "infer_vanilla_ae"


default_run = True



def main() -> str:
    reset_log_timer()
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "runs")
    run_dir = tensorboard_run_dir(log_root)

    try:
        import torch

        log_print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        log_print("Brak torch — source ./activate.sh")

    subprocess.run(["clear"], check=False)
    return run_dir

if __name__ == "__main__":
    run_dir = main()

    # Default run #
    if default_run is True:
        match experiment:

            case "simple_mlp":
                old_stuff.main(tensorboard_dir=run_dir)

            case "vanilla_ae" | "denoising_ae" | "vae" | "ddpm":
                train.main(tensorboard_dir=run_dir, architecture=experiment)

            case "infer_vanilla_ae" | "infer_denoising_ae" | "infer_vae" | "infer_ddpm":
                infer.main(architecture=experiment.removeprefix("infer_"))

    # Manual Stuff #

    # train.main(tensorboard_dir=run_dir, architecture="vanilla_ae", epochs=5, run_tag="quick", batch_size=128)
    # train.main(tensorboard_dir=run_dir, architecture="vanilla_ae", resume="latest", epochs=10)
    # infer.main(architecture="vanilla_ae")
