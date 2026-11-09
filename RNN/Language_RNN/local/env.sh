#!/bin/bash
# Źródłuj z grid/job_* (PL Grid). Cache i HF poza $HOME.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# Moduły (dostosuj jeśli na innym klastrze)
module load python/3.12.3-gcccore-13.3.0 2>/dev/null || module load python/3.12 2>/dev/null || true
