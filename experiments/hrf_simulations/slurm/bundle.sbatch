#!/bin/bash
# Export the portable three-ROI simulation bundle from the NSD BIDS tree.
# Single-threaded and I/O bound: a one-session test took ~2.5 s per run and
# ~200 MB RSS, so the ~3,400 complete runs need roughly 2.5 h. Runs 13-14 of
# sessions 21-38 have no events file; --skip-incomplete omits them and records
# the omissions in the bundle manifest. The bundle refuses to overwrite an
# existing output, so delete it first to rebuild.
#
#   sbatch experiments/hrf_simulations/slurm/bundle.sbatch [bids-root] [output.h5]
#
#SBATCH --job-name=hrf-bundle
#SBATCH --partition=russpold,normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/scratch/users/russpold/boldtailor
#SBATCH --output=/scratch/users/russpold/boldtailor/experiments/hrf_simulations/slurm/logs/%x-%j.out

set -euo pipefail

BIDS_ROOT=${1:-/oak/stanford/groups/russpold/data/NSD/BIDS}
OUTPUT=${2:-/oak/stanford/groups/russpold/data/NSD/nsd_three_rois.h5}

ml python/3.12.1 uv/0.10.8

export UV_CACHE_DIR=$SCRATCH/uv_cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$L_SCRATCH

echo "=== $(date '+%F %T') bundling $BIDS_ROOT -> $OUTPUT"
uv run --frozen python -m experiments.hrf_simulations.bundle \
    --bids-root "$BIDS_ROOT" \
    --output "$OUTPUT" \
    --skip-incomplete
echo "=== $(date '+%F %T') done"
