#!/bin/bash

# ============================================================================
# Check and install Python3 if not present
# ============================================================================

box() {
    local msg="$1"
    local len=${#msg}
    local line=$(printf '%*s' "$((len + 2))" | tr ' ' '-')
    echo "+${line}+"
    echo "| ${msg} |"
    echo "+${line}+"
}

box "Checking Python Installation"
if ! command -v python3 &> /dev/null; then
    box "[INFO] Python3 not found. Installing Python3..."
    # Install Python3 (Amazon Linux 2023 uses dnf)
    if command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip
    else
        box "[ERROR] Cannot find package manager (dnf/yum). Please install Python3 manually."
        exit 1
    fi
    # Verify installation
    if ! command -v python3 &> /dev/null; then
        box "[ERROR] Python3 installation failed. Please install manually."
        exit 1
    fi
    box "[SUCCESS] Python3 installed successfully."
else
    box "[INFO] Python3 is already installed: $(python3 --version)"
fi

# ============================================================================
# Python virtual environment setup
# ============================================================================
box "Python Environment Setup"
box "[INFO] Creating Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    box "[INFO] Virtual environment created."
else
    box "[INFO] Virtual environment already exists."
fi

# Activate virtual environment
box "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
box "[INFO] Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r experiments/knapsack/python/requirements.txt
box "[SUCCESS] Python dependencies installed."
