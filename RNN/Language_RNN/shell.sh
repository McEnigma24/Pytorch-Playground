#!/usr/bin/env bash
set -euo pipefail

clear

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/scripts/dsh"
