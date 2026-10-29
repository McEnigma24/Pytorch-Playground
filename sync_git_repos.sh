#!/bin/bash

clear

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Nie jesteś w repozytorium git." >&2
  exit 1
}

if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  git -C "$ROOT" add -A
  git -C "$ROOT" commit -m "fast"
  git -C "$ROOT" push
else
  echo "Brak zmian — nic do commita."
fi

git -C "$ROOT" pull --rebase
clear

