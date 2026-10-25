"""Pakiet packet — folder z modułami (np. packet.demo).

Ten plik uruchamia się przy pierwszym imporcie czegokolwiek z packet.*
(PYTHONPATH=/workspace/src, więc folder src/ jest na ścieżce importów).

Na później — typowe rzeczy, które możesz tu dodać:

  __version__ = "0.1.0"

  # Skróty API (re-export), żeby nie pisać długiej ścieżki:
  # from packet.demo import log_demo_scalars
  # __all__ = ["log_demo_scalars"]

  # Unikaj ciężkich importów (torch, pandas) na top level — każdy
  # import packet.* załaduje je od razu. Lepiej importować z podmodułów.
"""
