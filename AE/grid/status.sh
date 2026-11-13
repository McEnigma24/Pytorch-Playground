#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common_"

watch squeue -u "$USER"
