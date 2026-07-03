#!/bin/bash
set -euo pipefail

# Automated environment setup for SpecNet ai_core
# - creates a local .venv
# - upgrades pip
# - installs requirements (without torch)
# - installs CUDA-enabled torch if an NVIDIA GPU is present

VENV_DIR=".venv"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)/.."

echo "SpecNet ai_core setup starting..."

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo "Installing Python requirements (excluding torch wheel)..."
pip install --no-deps -r "$ROOT_DIR/requirements.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected. Installing CUDA-enabled PyTorch (cu130)..."
  pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu130 torch
else
  echo "No NVIDIA GPU detected. Installing CPU PyTorch from requirements..."
  # requirements.txt already requests a CPU-compatible torch via extra-index-url
  pip install -r "$ROOT_DIR/requirements.txt"
fi

echo "Setup complete. Activate the venv with: source $VENV_DIR/bin/activate"
