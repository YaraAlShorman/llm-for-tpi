#!/bin/bash
# =============================================================================
# 02_serve.sh — Start the Qwen3 inference server on a Jetstream2 g3.xl node
#
# Usage:
#   bash 02_serve.sh                                 # Qwen3-14B, port 8000
#   MODEL_ID=Qwen/Qwen3-32B bash 02_serve.sh
#   PORT=8080 bash 02_serve.sh
#   MAX_MODEL_LEN=16384 bash 02_serve.sh
#
# A100 (40 GB) fits:
#   Qwen3-14B bf16  — ~28 GB  (comfortable)
#   Qwen3-32B bf16  — ~64 GB  (does NOT fit; use Int4 quant or 80 GB A100)
#   Qwen3-32B Int4  — ~18 GB  (fits easily with bitsandbytes)
#
# ACCESS FROM YOUR LAPTOP:
#   ssh -L 8000:<jetstream-ip>:8000 ubuntu@<jetstream-ip>
#   curl http://localhost:8000/health
# =============================================================================
set -euo pipefail

VENV_DIR="${HOME}/qwen3-env"
MODEL_DIR="${HOME}/models"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_PY="${SCRIPT_DIR}/../server.py"
if [[ ! -f "${SERVER_PY}" ]]; then
    echo "Error: server.py not found at ${SERVER_PY}"
    echo "Run from inside the cloned llm-for-tpi repo."
    exit 1
fi

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "  Qwen3 Inference Server — Jetstream2"
echo "  Model         : ${MODEL_ID}"
echo "  Max ctx len   : ${MAX_MODEL_LEN}"
echo "  Port          : ${PORT}"
echo "  Host          : $(hostname -I | awk '{print $1}')"
echo "  Start         : $(date)"
echo "============================================"
echo ""
echo "To connect from your local machine:"
echo "  ssh -L ${PORT}:$(hostname -I | awk '{print $1}'):${PORT} ubuntu@<your-instance-ip>"
echo "Then open: http://localhost:${PORT}/health"
echo ""

HF_HOME="${MODEL_DIR}/hf_cache" \
MODEL_ID="${MODEL_ID}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
PORT="${PORT}" \
python3 "${SERVER_PY}"
