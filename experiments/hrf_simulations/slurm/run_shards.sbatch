#!/bin/bash
# Simulation `run` stage split into interleaved shards, one array task each,
# writing to the same output as workflow.sbatch. Each task fits only cases
# i, i+n, ... (n = array size), so tasks never contend for a case. Calibration
# must already exist. Afterwards, workflow.sbatch with the same roi/tag sweeps
# for anything missed (completed cases are verified and skipped) and builds
# the report and notebook:
#
#   jid=$(sbatch --parsable experiments/hrf_simulations/slurm/run_shards.sbatch [roi] [tag])
#   sbatch --dependency=afterany:$jid experiments/hrf_simulations/slurm/workflow.sbatch [roi] [tag]
#
# ~5 core-minutes per case on SKX: 6 shards x 15 workers finish the 12,580
# expanded cases in ~26 h, inside the 48 h normal QOS. Change --array to
# 0-(n-1) for n shards; tasks resume completed cases if resubmitted.
#
#SBATCH --job-name=hrf-shard
#SBATCH --partition=russpold,normal
#SBATCH --array=0-5
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# AVX-512 nodes only, matching workflow.sbatch, so shards and resumed runs
# produce bitwise-compatible fits.
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE
#SBATCH --chdir=/scratch/users/russpold/boldtailor
#SBATCH --output=/scratch/users/russpold/boldtailor/experiments/hrf_simulations/slurm/logs/%x-%A_%a.out

set -euo pipefail

ROI=${1:-lateraloccipital}
TAG=${2:-h5-$ROI-v1}
BUNDLE=/oak/stanford/groups/russpold/data/NSD/nsd_three_rois.h5
SIM=experiments/hrf_simulations

ml python/3.12.1 uv/0.10.8

export UV_CACHE_DIR=$SCRATCH/uv_cache
export TMPDIR=$L_SCRATCH
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== $(date '+%F %T') $ROI shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT ($TAG) on $(hostname)"
# Same arguments as the Makefile's run target, so case manifests match.
uv run --frozen python $SIM/slurm/shard.py "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" run \
    --calibration $SIM/calibration/$TAG \
    --output $SIM/outputs/$TAG \
    --bids-root /Volumes/extdata1/NSD/BIDS \
    --subject sub-07 \
    --roi "$ROI" \
    --mask "$PWD/experiments/hrf_estimation/pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz" \
    --bundle "$BUNDLE" \
    --stage expanded \
    --n-jobs 15
echo "=== $(date '+%F %T') shard $SLURM_ARRAY_TASK_ID done"
