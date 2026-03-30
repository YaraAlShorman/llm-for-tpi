#!/bin/bash
# =============================================================================
# Interactive Qwen3 session on Hyak
# =============================================================================
# Run this script (DON'T sbatch it) to get an interactive GPU shell where
# you can poke around, test the model, or debug issues.
#
# Usage:
#   bash 04_interactive.sh                   # defaults: ckpt-all, 1 GPU, 14B
#   bash 04_interactive.sh gpu-a40 2         # specific partition, 2 GPUs
# =============================================================================

PARTITION="${1:-ckpt-all}"
NGPUS="${2:-1}"
ACCOUNT="${3:-stf}"           # <-- Replace or pass as 3rd arg

GROUP_DIR="/gscratch/stf"       # <-- Replace
CONTAINER="${GROUP_DIR}/containers/qwen3.sif"
MODEL_CACHE="${GROUP_DIR}/models"

echo "Requesting interactive GPU session..."
echo "  Partition: ${PARTITION}"
echo "  GPUs:      ${NGPUS}"
echo "  Account:   ${ACCOUNT}"

salloc \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --nodes=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --gpus="${NGPUS}" \
    --time=04:00:00 \
    --job-name=qwen3-interactive \
    bash -c "
echo 'Got node: \$(hostname)'
echo 'GPU info:'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ''
echo 'Dropping into container shell...'
echo 'Try: python3 -c \"import torch; print(torch.cuda.get_device_name(0))\"'
echo ''

apptainer shell \
    --nv \
    --bind ${MODEL_CACHE}:/gscratch/models \
    ${CONTAINER}
"
