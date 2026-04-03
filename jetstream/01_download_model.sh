#!/bin/bash
# =============================================================================
# 01_download_model.sh — Download Qwen3 weights to ~/models
#
# Usage:
#   bash 01_download_model.sh                        # downloads Qwen3-14B
#   MODEL_ID=Qwen/Qwen3-32B bash 01_download_model.sh
#   HF_TOKEN=hf_xxx bash 01_download_model.sh        # for gated models
# =============================================================================
set -euo pipefail

VENV_DIR="${HOME}/qwen3-env"
MODEL_DIR="${HOME}/models"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"

source "${VENV_DIR}/bin/activate"

echo "=== Downloading model: ${MODEL_ID} ==="
echo "    Destination: ${MODEL_DIR}"
echo "    Started: $(date)"
echo ""

# Pass HF_TOKEN if set (needed for gated models)
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    echo "    HuggingFace token: set"
fi

python3 - <<EOF
from huggingface_hub import snapshot_download
import os

snapshot_download(
    "${MODEL_ID}",
    cache_dir="${MODEL_DIR}",
    resume_download=True,
)
print("Download complete.")
EOF

echo ""
echo "Finished: $(date)"
echo "Model cached at: ${MODEL_DIR}"
echo ""
echo "Next: bash 02_serve.sh"
