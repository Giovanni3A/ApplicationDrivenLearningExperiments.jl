#!/bin/bash

# ============================================================================
# Run Knapsack experiments (Python and Julia)
# ============================================================================

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

box "Running Knapsack Experiments"

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        box "[INFO] Activating virtual environment..."
        source venv/bin/activate
    else
        box "[ERROR] Virtual environment not found. Please run python_setup.sh first."
        return 1
    fi
fi

# Run Python knapsack experiments
cd experiments/knapsack/python
box "[INFO] Generating data for knapsack..."
python generate_data.py
box "[INFO] Running knapsack pyepo script..."
python run_pyepo.py
cd ../../..

# Run Julia knapsack experiments
box "Running Julia Knapsack Experiments"
box "[INFO] Running script: experiments/knapsack/julia/knapsack.jl"
julia --project=. experiments/knapsack/julia/knapsack.jl

# Run post-analysis
box "Running Knapsack Post-Analysis"
box "[INFO] Running script: experiments/knapsack/julia/post_analysis.jl"
julia --project=. experiments/knapsack/julia/post_analysis.jl

box "[SUCCESS] Knapsack experiments completed."
