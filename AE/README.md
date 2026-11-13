# ML template — uv + Docker (CUDA)

Jeden workflow: **zawsze** wchodzisz do środowiska przez `./shell.sh`, reszta działa w kontenerze.

## Pierwszy raz (WSL)

1. Ustaw `PROJECT_NAME` w [`_config_/config.env`](_config_/config.env) (nazwa w `pyproject.toml` zsynchronizuje się automatycznie przy `./shell.sh` / `./scripts/drun`).
2. `sudo ./_config_/first_time_init.sh` — Docker + NVIDIA Container Toolkit.
3. `docker compose --env-file _config_/config.env build`
4. `./scripts/drun sync`

## Codziennie

```bash
./shell.sh          # porty 8888 (Jupyter) i 6006 (TensorBoard) na hoście
tb                  # TensorBoard w tle
jnb                 # Jupyter Lab w tle
run_main            # python src/main.py (Ctrl+C kończy tylko ten skrypt)
exit                # pierwszy terminal — zamyka kontener i serwisy w tle
```

Drugie `./shell.sh` w innym terminalu **dołącza** do tego samego kontenera (`docker exec`). `exit` w drugim terminalu nie zatrzymuje kontenera — zatrzymuje go wyjście z terminala, który uruchomił `compose run`.

| Na hoście | Działanie |
|-----------|-----------|
| `./scripts/drun sync` | `uv sync` (bez zajmowania portów) |
| `./scripts/drun clean` | usuwa `build/` i stary `.venv` w kontenerze |
| `./scripts/drun verify` | test CUDA |

## Katalogi (nie commituj)

| Ścieżka | Znaczenie |
|---------|-----------|
| `build/venv` | środowisko Python (uv) |
| `build/uv-cache`, `build/python` | cache uv i interpreter |
| `build/run`, `build/logs` | PID i logi `tb` / `jnb` |
| `runs/` | logi TensorBoard |

`rm` na hoście dla `build/` może wymagać sudo (pliki z kontenera). Użyj `./scripts/drun clean`.

Szczegóły: [DOCKER.md](DOCKER.md).
