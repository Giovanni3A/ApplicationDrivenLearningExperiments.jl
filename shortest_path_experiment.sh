#!/bin/bash

# ============================================================================
# Run Shortest Path experiments (Python and Julia)
# ============================================================================

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

box "Running Shortest Path Experiments"

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        box "[INFO] Activating virtual environment..."
        source venv/bin/activate
    else
        box "[ERROR] Virtual environment not found. Please run python_setup.sh first."
        exit 1
    fi
fi

# Run Python shortest_path experiments
cd experiments/shortest_path/python
box "[INFO] Generating data for shortest_path..."
python generate_data.py
box "[INFO] Running shortest_path pyepo script..."
python run_pyepo.py
cd ../../..

# Run Julia shortest_path experiments
box "Running Julia Shortest Path Experiments"
box "[INFO] Running script: experiments/shortest_path/julia/shortest_path.jl"
julia --project=. experiments/shortest_path/julia/shortest_path.jl

# Run post-analysis
box "Running Shortest Path Post-Analysis"
box "[INFO] Running script: experiments/shortest_path/julia/post_analysis.jl"
julia --project=. experiments/shortest_path/julia/post_analysis.jl

box "[SUCCESS] Shortest Path experiments completed."
