# Docker + uv — architektura

## Podział

| Warstwa | Odpowiedzialność |
|---------|------------------|
| Obraz | CUDA/cuDNN, biblioteki systemowe, **uv** |
| Projekt | `.python-version`, `pyproject.toml`, `uv sync` → `build/venv` |
| `build/` | venv, cache uv, interpreter(y), PID/logi `tb`/`jnb` |
| `runs/` | logi TensorBoard (`SummaryWriter`) |

## Workflow

Tylko **`./shell.sh`** uruchamia kontener z portami **8888** i **6006**. W środku: `tb`, `jnb`, `run_main` (katalog [`bin/`](bin/)).

`./scripts/drun` to narzędzia z hosta (`sync`, `lock`, `verify`, `clean`, `run`) — **bez** `--service-ports`, więc działają przy otwartym shellu.

## Konfiguracja

[`_config_/config.env`](_config_/config.env): `PROJECT_NAME`, porty, `JUPYTER_NOTEBOOK_DIR`, `TENSORBOARD_LOGDIR`, `PYTHONPATH_EXTRA`.

Dwa projekty naraz: różne `JUPYTER_PORT` / `TENSORBOARD_PORT` w każdym `config.env`.

## GPU

Overlay `docker-compose.gpu.yml` gdy Docker ma runtime `nvidia`. `ML_TEMPLATE_GPU=0|1|auto`.

## Host (WSL)

```bash
sudo ./_config_/first_time_init.sh
```
