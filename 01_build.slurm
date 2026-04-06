#!/bin/bash
#SBATCH --job-name=qwen3-build
#SBATCH --partition=gpu-h200
#SBATCH --account=rhlab        # <-- Replace with your account
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/logs/build_logs/build_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yarasho@uw.edu
#SBATCH --chdir=/gpfs/projects/rhlab/yarasho/llm-for-tpi

# =============================================================================
# Build the Apptainer container image on a compute node
# =============================================================================

set -euo pipefail

# -- Configuration --
GROUP_DIR="/gpfs/projects/rhlab/yarasho"
CONTAINER_DIR="${GROUP_DIR}/containers"
MODEL_CACHE="${GROUP_DIR}/models"

mkdir -p "${CONTAINER_DIR}" "${MODEL_CACHE}"

# Use the node-local SSD for building (avoids home dir quota issues)
export APPTAINER_CACHEDIR=/tmp
export APPTAINER_TMPDIR=/tmp

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

echo "=== Building Apptainer container ==="
echo "Build started: $(date)"

# Build the .sif image from the definition file
apptainer build --fakeroot \
    /tmp/qwen3.sif \
    "${SCRIPT_DIR}/qwen3.def"

echo "=== Build complete, copying to project dir ==="
cp /tmp/qwen3.sif "${CONTAINER_DIR}/qwen3.sif"

echo "Container saved to: ${CONTAINER_DIR}/qwen3.sif"
echo "Build finished: $(date)"
