#!/bin/bash
# 6 niezależnych jobów (6× gpu:1) — w squeue widać każdy osobno; startują gdy zwolni GPU.
source "$(dirname "${BASH_SOURCE[0]}")/_common_"

mkdir -p grid/output
export LANGUAGE_RNN_ROOT="$GRID_ROOT"

ARCHS=(rnnRNN rnnGRU rnnLSTM rnnAWDLSTM rnnBaseLSTM rnnGoldenLSTM)
IDS=()
for arch in "${ARCHS[@]}"; do
  jid=$(sbatch --parsable \
    --job-name="lrnn_${arch}" \
    --export=ALL,LRNN_ARCH="$arch",LANGUAGE_RNN_ROOT="$GRID_ROOT" \
    grid/job_rnn_one)
  IDS+=("$jid")
  echo "$arch → job $jid"
done
printf '%s\n' "${IDS[@]}" > grid/.latest_jobs_rnn1ep
echo "${IDS[-1]}" > grid/.latest_job_started
squeue -u "$USER" | head -20
