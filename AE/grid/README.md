# PL Grid — Language_RNN

Uruchamiaj z katalogu repo (`Language_RNN/`).

## Jednorazowy test: 6× RNN, 1 epoka, ~125M

Każdy model = **jedno GPU** (`--gres=gpu:1`). W obu wariantach mogą iść **równolegle**, jeśli klaster da 6 wolnych kart.

```bash
./grid/run_rnn_1ep.sh            # jeden array 21699592_[0-5] — 6 tasków, po 1 GPU
./grid/run_rnn_1ep_separate.sh   # 6 osobnych jobów (czytelniejszy squeue)

./grid/status.sh
./grid/cancel.sh                 # ostatni pojedynczy id w .latest_job_started
# array: scancel 21699592
# osobne: xargs scancel < grid/.latest_jobs_rnn1ep
```

Logi: `grid/output/rnn1ep_<arrayJobId>_<task>.{out,err}`  
TensorBoard: `$SCRATCH/lrnn_runs/<arrayJobId>/`

Preset: `test125M_1ep` w `scripts/plgrid/run_train.py` (jak `src/main.py`).

## Pojedynczy GPU job

```bash
LRNN_SHELL_CMD='uv run python scripts/plgrid/run_train.py --architecture rnnLSTM --preset test125M_1ep --tensorboard-root $SCRATCH/lrnn_test' \
  ./grid/run_gpu.sh 1 8
```

## Środowisko

`local/env.sh` — moduł Python + cache uv/HF na `$SCRATCH`. Przed pierwszym jobem na loginie:

```bash
source local/env.sh && uv sync
```

Na V100 używamy PyTorch z indeksu `pytorch-cu126` (`sm_70` w wheelu).
