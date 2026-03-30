#!/bin/bash
#SBATCH --job-name=qwen3-serve
#SBATCH --partition=gpu-l40s            # L40s has 48GB VRAM — good for 14B
#SBATCH --account=stf          # <-- Replace with your account
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1                        # 1 GPU for 14B; use 2 for 32B
#SBATCH --time=08:00:00
#SBATCH --output=serve_%j.log
#SBATCH --chdir=/mmfs1/gscratch/stf/yarasho/llm-for-tpi

# =============================================================================
# Run the Qwen3 inference server on a Hyak GPU node
# =============================================================================
#
# GPU RECOMMENDATIONS (for account: stf / user: yarasho):
#   Qwen3-14B (bf16): ~28 GB VRAM → 1x L40/L40s (48GB)
#   Qwen3-32B (bf16): ~64 GB VRAM → 2x L40/L40s
#
# YOUR AVAILABLE PARTITIONS:
#   gpu-l40s     — 48GB VRAM, 4 free  (best choice for 14B)
#   gpu-l40      — 48GB VRAM, 4 free  (also works great)
#   gpu-2080ti   — 11GB VRAM, 6 free  (too small for 14B)
#   ckpt-all     — checkpoint queue (may be preempted, but free)
#
# ACCESS THE SERVER:
#   From a login node, use SSH port forwarding:
#     ssh -L 8000:<nodename>:8000 <netid>@klone.hyak.uw.edu
#   Then query the API at http://localhost:8000
# =============================================================================

set -euo pipefail

# -- Configuration --
GROUP_DIR="/mmfs1/gscratch/stf/yarasho"
CONTAINER="${GROUP_DIR}/containers/qwen3.sif"
MODEL_CACHE="${GROUP_DIR}/models"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

echo "============================================"
echo "  Qwen3 Inference Server — Hyak"
echo "  Model : ${MODEL_ID}"
echo "  Node  : $(hostname)"
echo "  GPUs  : ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  Start : $(date)"
echo "============================================"
echo ""
echo "To connect from your local machine, run:"
echo "  ssh -L 8000:$(hostname):8000 \$USER@klone.hyak.uw.edu"
echo "Then open: http://localhost:8000/health"
echo ""

# Run the server inside the Apptainer container
#   --nv          : enables NVIDIA GPU passthrough
#   --bind        : mounts gscratch paths into the container
apptainer exec \
    --nv \
    --bind "${MODEL_CACHE}:/gscratch/models" \
    --bind "${SCRIPT_DIR}/server.py:/app/server.py" \
    "${CONTAINER}" \
    python3 /app/server.py
