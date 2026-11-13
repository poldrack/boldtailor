#!/bin/bash
# Expanded-stage simulation workflow (calibrate -> run -> report -> notebook)
# for one ROI of the three-ROI bundle. Timing on one core: calibration ~3 min,
# ~5 core-minutes and ~0.5 GB per dataset, so the 12,580 expanded datasets take
# ~70 h at 15 workers. That exceeds the 48 h normal QOS, so this runs on the
# russpold partition only. Calibration is verified and reused and `run` resumes
# completed cases, so resubmitting the same command continues an interrupted job.
#
#   sbatch experiments/hrf_simulations/slurm/workflow.sbatch [roi] [tag]
#
# roi defaults to lateraloccipital; tag (default h5-<roi>-v1) names the
# calibration, output and report directories.
#
#SBATCH --job-name=hrf-sim
#SBATCH --partition=russpold
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# AVX-512 nodes only, as in nsd_replication/slurm: NumPy's SIMD kernels differ
# in the last bit between AVX2 and AVX-512 CPUs, and a resumed run should not
# mix fits from both.
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE
#SBATCH --chdir=/scratch/users/russpold/boldtailor
#SBATCH --output=/scratch/users/russpold/boldtailor/experiments/hrf_simulations/slurm/logs/%x-%j.out

set -euo pipefail

ROI=${1:-lateraloccipital}
TAG=${2:-h5-$ROI-v1}
BUNDLE=/oak/stanford/groups/russpold/data/NSD/nsd_three_rois.h5
SIM=experiments/hrf_simulations

ml python/3.12.1 uv/0.10.8

export UV_CACHE_DIR=$SCRATCH/uv_cache
export TMPDIR=$L_SCRATCH

echo "=== $(date '+%F %T') $ROI workflow ($TAG) on $(hostname)"
uv run --frozen make -f $SIM/Makefile workflow \
    UV_RUN="uv run --frozen" \
    STAGE=expanded \
    BUNDLE="$BUNDLE" \
    ROI="$ROI" N_JOBS=15 \
    CALIBRATION=$SIM/calibration/$TAG \
    OUTPUT=$SIM/outputs/$TAG \
    REPORT=$SIM/reports/$TAG \
    NOTEBOOK=$SIM/$ROI-results.ipynb
echo "=== $(date '+%F %T') $ROI workflow done"
