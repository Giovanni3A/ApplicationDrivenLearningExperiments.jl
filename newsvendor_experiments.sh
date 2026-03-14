#!/bin/bash

# ============================================================================
# Run Julia Newsvendor experiments
# ============================================================================

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

box "Starting Julia Newsvendor Experiments"

# Run newsvendor_2.jl script
box "[INFO] Running script: experiments/newsvendor_2/newsvendor_2.jl"
julia --project=. experiments/newsvendor_2/newsvendor_2.jl

# Run newsvendor_3.jl script
box "[INFO] Running script: experiments/newsvendor_3/newsvendor_3.jl"
julia --project=. experiments/newsvendor_3/newsvendor_3.jl

# Run post-analysis for newsvendor_3
box "Running Newsvendor 3 Post-Analysis"
box "[INFO] Running script: experiments/newsvendor_3/post_analysis.jl"
julia --project=. experiments/newsvendor_3/post_analysis.jl

box "[SUCCESS] Newsvendor experiments completed."
