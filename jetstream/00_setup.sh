#!/bin/bash
# =============================================================================
# 00_setup.sh — One-time environment setup for Qwen3 on Jetstream2 g3.xl
#               Ubuntu 22.04 + A100
#
# Run once after provisioning your instance:
#   bash 00_setup.sh
# =============================================================================
set -euo pipefail

VENV_DIR="${HOME}/qwen3-env"
MODEL_DIR="${HOME}/models"

echo "=== System packages ==="
sudo apt-get update -q
sudo apt-get install -y -q \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    git curl wget \
    build-essential

echo "=== Creating Python venv at ${VENV_DIR} ==="
python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing inference dependencies ==="
pip install \
    transformers>=4.51.0 \
    accelerate \
    huggingface_hub \
    fastapi \
    uvicorn[standard] \
    pydantic

echo "=== Verifying GPU ==="
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check NVIDIA driver"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

mkdir -p "${MODEL_DIR}"

echo ""
echo "Setup complete."
echo "  Venv   : ${VENV_DIR}"
echo "  Models : ${MODEL_DIR}"
echo ""
echo "Next: bash 01_download_model.sh"
