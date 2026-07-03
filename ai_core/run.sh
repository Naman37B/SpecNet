#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting SpecNet Local Draft Environment..."

# Check if the virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Prefer .venv if present
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    VENV_DIR="venv"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Execute the core logic
python3 speculative_decoding.py

echo "Execution complete."
