#!/bin/bash

# ============================================================================
# Run Matpower experiments
# ============================================================================

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

box "Running Matpower Experiments"
box "[INFO] Running script: experiments/matpower/auto_run.jl"
julia --project=. experiments/matpower/auto_run.jl

box "[SUCCESS] Matpower experiments completed."
