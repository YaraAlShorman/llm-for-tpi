#!/bin/bash
#SBATCH --job-name=qwen3-download
#SBATCH --partition=gpu-h200
#SBATCH --account=rhlab        # <-- Replace with your account
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=download_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yarasho@uw.edu
#SBATCH --chdir=/gpfs/projects/rhlab/yarasho/llm-for-tpi

# =============================================================================
# Download Qwen3 model weights to project dir
# =============================================================================
# Run this AFTER the container is built. The model is large (~28 GB for 14B
# in bf16, ~7 GB effective with INT4 quantization loaded at runtime).
# =============================================================================

set -euo pipefail

# -- Configuration --
GROUP_DIR="/gpfs/projects/rhlab/yarasho"
CONTAINER="${GROUP_DIR}/containers/qwen3.sif"
MODEL_CACHE="${GROUP_DIR}/models"
MODEL_ID="${1:-Qwen/Qwen3-14B}"        # Pass model name as arg, or defaults to 14B

mkdir -p "${MODEL_CACHE}"

echo "=== Downloading model: ${MODEL_ID} ==="
echo "Cache directory: ${MODEL_CACHE}"
echo "Download started: $(date)"

# If you need a HuggingFace token for gated models, uncomment:
# export HF_TOKEN="hf_xxxxx"

apptainer exec \
    --bind "${MODEL_CACHE}:/gpfs/projects/rhlab/yarasho/models" \
    --bind /tmp:/tmp \
    --env HF_XET_CACHE=/gpfs/projects/rhlab/yarasho/models/xet \
    "${CONTAINER}" \
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_ID}',
    cache_dir='/gpfs/projects/rhlab/yarasho/models',
)
print('Download complete.')
"

echo "=== Model downloaded ==="
echo "Finished: $(date)"
echo "Contents of model cache:"
du -sh "${MODEL_CACHE}"/*
