#!/bin/bash
#SBATCH --job-name=qwen3-serve
#SBATCH --partition=gpu-h200            # H200 has 141GB VRAM
#SBATCH --account=rhlab        # <-- Replace with your account
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1                        # 1 H200 is sufficient for any Qwen3 variant
#SBATCH --time=08:00:00
#SBATCH --output=serve_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yarasho@uw.edu
#SBATCH --chdir=/gpfs/projects/rhlab/yarasho/llm-for-tpi

# =============================================================================
# Run the Qwen3 inference server on a Tillicum H200 node
# =============================================================================
#
# GPU NOTES (H200 — 141GB VRAM):
#   Qwen3-14B INT4  : ~7 GB weights  → ~134 GB free for KV cache (very large contexts)
#   Qwen3-14B bf16  : ~28 GB weights → ~113 GB free for KV cache
#   Qwen3-32B INT4  : ~18 GB weights → ~123 GB free for KV cache
#   Qwen3-32B bf16  : ~64 GB weights → ~77 GB free for KV cache
#
# QUANTIZATION:
#   Set QUANTIZATION=int4 (default) for maximum context headroom.
#   Set QUANTIZATION=int8 for a quality/size tradeoff.
#   Leave QUANTIZATION= empty for full bf16.
#
# ACCESS THE SERVER:
#   From a login node, use SSH port forwarding:
#     ssh -L 8000:<nodename>:8000 <user>@<tillicum-login-node>
#   Then query the API at http://localhost:8000
# =============================================================================

set -euo pipefail

# -- Configuration --
GROUP_DIR="/gpfs/projects/rhlab/yarasho"
CONTAINER="${GROUP_DIR}/containers/qwen3.sif"
MODEL_CACHE="${GROUP_DIR}/models"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
QUANTIZATION="${QUANTIZATION:-}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

echo "============================================"
echo "  Qwen3 Inference Server — Tillicum"
echo "  Model         : ${MODEL_ID}"
echo "  Quantization  : ${QUANTIZATION:-none (bf16)}"
echo "  Max ctx len   : ${MAX_MODEL_LEN}"
echo "  Node          : $(hostname)"
echo "  GPUs          : ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  Start         : $(date)"
echo "============================================"
echo ""
echo "To connect from your local machine, run:"
echo "  ssh -L 8000:$(hostname):8000 \$USER@<tillicum-login-node>"
echo "Then open: http://localhost:8000/health"
echo ""

# Run the server inside the Apptainer container
#   --nv          : enables NVIDIA GPU passthrough
#   --bind        : mounts project paths into the container
apptainer exec \
    --nv \
    --bind "${MODEL_CACHE}:/gpfs/projects/rhlab/yarasho/models" \
    --bind "${SCRIPT_DIR}/server.py:/app/server.py" \
    --env HF_HOME=/gpfs/projects/rhlab/yarasho/models/hf_cache \
    --env MODEL_ID="${MODEL_ID}" \
    --env MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    --env QUANTIZATION="${QUANTIZATION}" \
    "${CONTAINER}" \
    python3 /app/server.py
