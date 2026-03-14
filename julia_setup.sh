#!/bin/bash

# ============================================================================
# Check and install Julia if not present
# ============================================================================

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

box "Checking Julia Installation"
if ! command -v julia &> /dev/null; then
    box "[INFO] Julia not found. Installing Julia..."
    # Install Julia using official method for Amazon Linux
    curl -fsSL https://install.julialang.org | sh
    # Add Julia to PATH for current session
    export PATH="$HOME/.juliaup/bin:$PATH"
    # Verify installation
    if ! command -v julia &> /dev/null; then
        box "[ERROR] Julia installation failed. Please install manually."
        exit 1
    fi
    box "[SUCCESS] Julia installed successfully."
else
    box "[INFO] Julia is already installed: $(julia --version)"
fi
