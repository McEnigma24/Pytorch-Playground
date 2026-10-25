# syntax=docker/dockerfile:1
#
# Obraz = CUDA/cuDNN + biblioteki systemowe. Python i pakiety = uv w /workspace/build/.
# Cały repo: bind-mount .:/workspace
#
# Build:  docker compose --env-file _config_/config.env build
# Sync:    ./scripts/drun sync
# Shell:   ./shell.sh

FROM ghcr.io/astral-sh/uv:0.6.14 AS uv-bin

FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TZ=Europe/Warsaw

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_CACHE_DIR=/workspace/build/uv-cache \
    UV_PYTHON_INSTALL_DIR=/workspace/build/python \
    UV_PROJECT_ENVIRONMENT=/workspace/build/venv \
    PATH="/workspace/bin:/workspace/build/venv/bin:/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# System (apt) — CUDA runtime deps + buildy natywnych wheeli
# ---------------------------------------------------------------------------
RUN apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    tzdata \
    build-essential \
    pkg-config \
    cmake \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libegl1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    xauth \
    xvfb \
    openssh-client \
    vim \
    less \
    htop \
 && ln -snf /usr/share/zoneinfo/Europe/Warsaw /etc/localtime \
 && echo "Europe/Warsaw" > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /usr/local/bin/uv

WORKDIR /workspace

CMD ["bash"]
