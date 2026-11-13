#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common_"

scancel "$(cat grid/.latest_job_started)"
