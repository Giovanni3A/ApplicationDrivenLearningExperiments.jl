#!/bin/bash

# ============================================================================
# Main script to run all experiments
# ============================================================================
# This script orchestrates the execution of all setup and experiment scripts
# in the correct order.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

# ============================================================================
# Setup phase
# ============================================================================
box "Starting Application-Driven Learning Experiments"

# Run Julia setup
box "[INFO] Running Julia setup..."
source julia_setup.sh
if [ $? -ne 0 ]; then
    box "[ERROR] Julia setup failed."
    exit 1
fi

# Run Python setup
box "[INFO] Running Python setup..."
source python_setup.sh
if [ $? -ne 0 ]; then
    box "[ERROR] Python setup failed."
    exit 1
fi

# ============================================================================
# Experiments phase
# ============================================================================
box "Starting Experiments"

# Run Newsvendor experiments
box "[INFO] Running Newsvendor experiments..."
bash newsvendor_experiments.sh
if [ $? -ne 0 ]; then
    box "[ERROR] Newsvendor experiments failed."
    exit 1
fi

# Run Knapsack experiments
box "[INFO] Running Knapsack experiments..."
bash knapsack_experiment.sh
if [ $? -ne 0 ]; then
    box "[ERROR] Knapsack experiments failed."
    exit 1
fi

# Run Shortest Path experiments
box "[INFO] Running Shortest Path experiments..."
bash shortest_path_experiment.sh
if [ $? -ne 0 ]; then
    box "[ERROR] Shortest Path experiments failed."
    exit 1
fi

# Run Matpower experiments
box "[INFO] Running Matpower experiments..."
bash matpower_experiment.sh
if [ $? -ne 0 ]; then
    box "[ERROR] Matpower experiments failed."
    exit 1
fi

# Deactivate virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    box "[INFO] Deactivating virtual environment..."
    deactivate
fi

# ============================================================================
# Success message
# ============================================================================
box "[SUCCESS] All experiments completed."
