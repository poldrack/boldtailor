#!/bin/bash
# Full NSD replication chain for one config: fits, metrics, features,
# comparison, pilot variability report and figures, in the README's order.
# Every stage skips work already written, so a resubmission resumes.
#
# The job runs from the repository root wherever it is submitted from, so a
# relative config path is taken relative to the repository root. The log
# directory must exist first:
#   mkdir -p experiments/nsd_replication/slurm/logs
#   sbatch experiments/nsd_replication/slurm/run_all.sbatch [config.toml]
#
#SBATCH --job-name=nsd-replication
#SBATCH --partition=russpold,normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --chdir=/oak/stanford/groups/russpold/data/NSD/boldtailor
#SBATCH --output=/oak/stanford/groups/russpold/data/NSD/boldtailor/experiments/nsd_replication/slurm/logs/%x-%j.out

set -euo pipefail

CONFIG=${1:-experiments/nsd_replication/configs/pilot.toml}

ml python/3.12.1 uv/0.10.8
ml biology workbench

# The config's n_jobs workers share the allocation; split the remaining
# cores between them as BLAS threads instead of oversubscribing.
N_JOBS=$(sed -n 's/^n_jobs *= *\([0-9]*\).*/\1/p' "$CONFIG")
THREADS=$(( SLURM_CPUS_PER_TASK / ${N_JOBS:-1} ))
(( THREADS < 1 )) && THREADS=1
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

# Temporary files (ROI resampling) go to node-local storage.
export TMPDIR=$L_SCRATCH

run() {
    echo "=== $(date '+%F %T') run $*"
    uv run --frozen python -m experiments.nsd_replication.run "$@" --config "$CONFIG"
}

run fit      --source nsdpreproc
run metrics  --source nsdpreproc
run features
run fit      --source released
run metrics  --source released
run metrics  --source comparison

echo "=== $(date '+%F %T') pilot_report"
uv run --frozen python -m experiments.nsd_replication.pilot_report \
    --config "$CONFIG" --source nsdpreproc

run figures
echo "=== $(date '+%F %T') done"
