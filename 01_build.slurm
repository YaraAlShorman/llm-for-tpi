#!/bin/bash
#SBATCH --job-name=qwen3-build
#SBATCH --partition=ckpt-all
#SBATCH --account=stf        # <-- Replace with your account (run `hyakalloc`)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=build_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yarasho@uw.edu
#SBATCH --chdir=/mmfs1/gscratch/stf/yarasho/llm-for-tpi

# =============================================================================
# Build the Apptainer container image on a compute node
# =============================================================================
# The home directory has a 10GB limit, so we build on /tmp (node-local SSD)
# and copy the final .sif to /gscratch.
#
# BEFORE RUNNING:
#   1. Replace stf and stf with your actual Hyak account/group.
#   2. Ensure qwen3.def and server.py are in the same directory as this script.
# =============================================================================

set -euo pipefail

# -- Configuration --
GROUP_DIR="/mmfs1/gscratch/stf/yarasho"
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

echo "=== Build complete, copying to gscratch ==="
cp /tmp/qwen3.sif "${CONTAINER_DIR}/qwen3.sif"

echo "Container saved to: ${CONTAINER_DIR}/qwen3.sif"
echo "Build finished: $(date)"
