"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess

import infer
import old_stuff
import train
from log_print import log_print, reset_log_timer, tensorboard_run_dir

# experiment = "simple_mlp"

# experiment = "rnnRNN"
experiment = "rnnLSTM"
# experiment = "rnnGRU"

# experiment = "rnnBaseLSTM"
# experiment = "rnnAWDLSTM"
# experiment = "rnnGoldenLSTM"

# experiment = "transformer_nanoGPT"
# experiment = "transformer_miniLLaMA"
# experiment = "transformer_GPT2small"

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

    # --- presety ~110-125M (vocab 32k, seq_len 1024) --- #

    epochs=1

    rnn_params = dict(embed_dim=1536, hidden_size=1536, num_layers=4, dropout=0.3, seq_len=1024, window_stride=1024, batch_size=32)
    train.main(tensorboard_dir=run_dir, epochs=epochs, checkpoint_every=5, eval_every=5, run_tag="125M", architecture="rnnRNN", **rnn_params)
    train.main(tensorboard_dir=run_dir, epochs=epochs, checkpoint_every=5, eval_every=5, run_tag="125M", architecture="rnnGRU", **rnn_params)
    train.main(tensorboard_dir=run_dir, epochs=epochs, checkpoint_every=5, eval_every=5, run_tag="125M", architecture="rnnLSTM", **rnn_params)
    train.main(tensorboard_dir=run_dir, epochs=epochs, checkpoint_every=5, eval_every=5, run_tag="125M", architecture="rnnAWDLSTM", **rnn_params)
    train.main(tensorboard_dir=run_dir, epochs=epochs, checkpoint_every=5, eval_every=5, run_tag="125M", architecture="rnnBaseLSTM", **rnn_params)
    train.main(tensorboard_dir=run_dir, epochs=epochs, checkpoint_every=5, eval_every=5, run_tag="125M", architecture="rnnGoldenLSTM", **rnn_params)

    # gpt2_params = dict(embed_dim=768, hidden_size=768, num_layers=12, num_heads=12, ff_size=3072, dropout=0.1, seq_len=1024, window_stride=1024, batch_size=16)
    # train.main(tensorboard_dir=run_dir, epochs=epochs, run_tag="110M", architecture="transformer_GPT2small", **gpt2_params)
    # llama_params = dict(embed_dim=768, hidden_size=768, num_layers=12, num_heads=12, ff_size=2048, dropout=0.1, seq_len=1024, window_stride=1024, batch_size=16)
    # train.main(tensorboard_dir=run_dir, epochs=epochs, run_tag="110M", architecture="transformer_miniLLaMA", **llama_params)
    # nano_params = dict(embed_dim=1024, hidden_size=1024, num_layers=6, num_heads=16, ff_size=4096, dropout=0.1, seq_len=1024, window_stride=1024, batch_size=16)
    # train.main(tensorboard_dir=run_dir, epochs=epochs, run_tag="110M", architecture="transformer_nanoGPT", **nano_params)

    # train.main( tensorboard_dir=run_dir, epochs=5, checkpoint_every=5, eval_every=1, run_tag="quick-test", architecture="rnnRNN",  )
    # train.main( tensorboard_dir=run_dir, epochs=5, checkpoint_every=5, eval_every=1, run_tag="quick-test", architecture="rnnLSTM", )
    # train.main( tensorboard_dir=run_dir, epochs=5, checkpoint_every=5, eval_every=1, run_tag="quick-test", architecture="rnnGRU",  )




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

