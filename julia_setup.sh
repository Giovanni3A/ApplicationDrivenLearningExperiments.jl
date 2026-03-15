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
    curl -fsSL https://install.julialang.org | sh -s -- --yes
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

# ============================================================================
# Check and install MPI if not present
# ============================================================================
box "Checking MPI Installation"
if ! command -v mpiexec &> /dev/null; then
    box "[INFO] MPI not found. Installing OpenMPI..."
    if command -v dnf &> /dev/null; then
        sudo dnf install -y openmpi openmpi-devel
    elif command -v yum &> /dev/null; then
        sudo yum install -y openmpi openmpi-devel
    else
        box "[ERROR] Cannot find package manager (dnf/yum). Please install OpenMPI manually."
        exit 1
    fi
    # Add OpenMPI to PATH for current session
    export PATH="/usr/lib64/openmpi/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH"
    if ! command -v mpiexec &> /dev/null; then
        box "[ERROR] MPI installation failed. Please install manually."
        exit 1
    fi
    box "[SUCCESS] OpenMPI installed successfully."
else
    box "[INFO] MPI is already installed: $(mpiexec --version 2>&1 | head -1)"
fi

# ============================================================================
# Instantiate Julia project (install all packages from Manifest.toml)
# ============================================================================
box "[INFO] Instantiating Julia project..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
julia --project="$SCRIPT_DIR" -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"
if [ $? -ne 0 ]; then
    box "[ERROR] Julia project instantiation failed."
    exit 1
fi
box "[SUCCESS] Julia project instantiated and precompiled successfully."
