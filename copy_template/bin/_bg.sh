#!/usr/bin/env bash
# Background service helpers for tb / jnb.
set -euo pipefail

bg_is_running() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile")"
  kill -0 "$pid" 2>/dev/null
}

bg_start() {
  local name="$1"
  local pidfile="$2"
  local logfile="$3"
  shift 3

  mkdir -p "$(dirname "$pidfile")" "$(dirname "$logfile")"

  if bg_is_running "$pidfile"; then
    echo "[${name}] już działa (pid $(cat "$pidfile"))"
    return 0
  fi

  nohup "$@" >>"$logfile" 2>&1 &
  echo $! >"$pidfile"
  echo "[${name}] start pid $(cat "$pidfile"), log: ${logfile}"
}

bg_stop() {
  local name="$1"
  local pidfile="$2"
  if ! bg_is_running "$pidfile"; then
    echo "[${name}] nie działa"
    rm -f "$pidfile"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  kill "$pid" 2>/dev/null || true
  sleep 0.5
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  rm -f "$pidfile"
  echo "[${name}] zatrzymany"
}

bg_status() {
  local name="$1"
  local pidfile="$2"
  if bg_is_running "$pidfile"; then
    echo "[${name}] running, pid $(cat "$pidfile")"
  else
    echo "[${name}] stopped"
    rm -f "$pidfile"
  fi
}

bg_logs() {
  local logfile="$1"
  touch "$logfile"
  tail -f "$logfile"
}
