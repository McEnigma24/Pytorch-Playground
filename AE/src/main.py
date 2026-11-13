"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess

import infer
import old_stuff
import train
from log_print import log_print, reset_log_timer, tensorboard_run_dir




experiment = "rnnLSTM"




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

            case "simple_mlp":
                old_stuff.main(tensorboard_dir=run_dir)

            case (
                "rnnLSTM" | "rnnGRU" | "rnnRNN" | "rnnBaseLSTM" | "rnnAWDLSTM" | "rnnGoldenLSTM"
                | "transformer_nanoGPT" | "transformer_miniLLaMA" | "transformer_GPT2small"
            ):
                train.main(tensorboard_dir=run_dir, architecture=experiment)

            case "infer_rnnLSTM" | "infer_rnnGRU" | "infer_rnnRNN":
                infer.main(architecture=experiment.removeprefix("infer_"))



    # Manual Stuff #

    # seed = "Na zamku"
    # length = 25

    # infer.main(architecture="rnnRNN", seed=seed,
    #             checkpoint_path="saved_models/rnnRNN-2026.11.01-011843-quick-test/rnnRNN_final.pt",
    #             length=length,
    # )

    # infer.main(architecture="rnnLSTM", seed=seed,
    #             checkpoint_path="saved_models/rnnLSTM-2026.11.01-012810-quick-test/rnnLSTM_final.pt",
    #             length=length,
    # )

    # infer.main(architecture="rnnGRU", seed=seed,
    #             checkpoint_path="saved_models/rnnGRU-2026.11.01-014552-quick-test/rnnGRU_final.pt",
    #             length=length,
    # )







    # infer.compare_models(
    #     models=[
    #         ("rnnRNN", "saved_models/rnnRNN-2026.11.01-011843-quick-test/rnnRNN_final.pt"),
    #         ("rnnLSTM", "saved_models/rnnLSTM-2026.11.01-012810-quick-test/rnnLSTM_final.pt"),
    #         ("rnnGRU", "saved_models/rnnGRU-2026.11.01-014552-quick-test/rnnGRU_final.pt"),
    #     ],
    #     prompts=[(seed, length), ("W polskim", 50)],
    #     temperature=0.85,
    #     run_tag="manual-compare",
    # )