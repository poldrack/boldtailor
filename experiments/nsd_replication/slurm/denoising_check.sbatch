#!/bin/bash
# Denoising diagnostic (denoising_check.py) on one dev session per subject:
# ON-OFF vs full-task-model noise pool x time-series vs beta-consistency
# count scoring, real and null-shifted events. Writes
# <output_dir>/checks/denoising/<subject>/<session>_{real,null}.tsv.
#
#   sbatch experiments/nsd_replication/slurm/denoising_check.sbatch [config.toml]
#
#SBATCH --job-name=denoising-check
#SBATCH --partition=russpold,normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE
#SBATCH --chdir=/oak/stanford/groups/russpold/data/NSD/boldtailor
#SBATCH --output=/oak/stanford/groups/russpold/data/NSD/boldtailor/experiments/nsd_replication/slurm/logs/%x-%A_%a.out

set -euo pipefail

CONFIG=${1:-experiments/nsd_replication/configs/dev.toml}
# Two sessions where the current rule chose 0 PCs, two where it kept some.
CASES=(
    "sub-01 ses-nsd11"
    "sub-02 ses-nsd18"
    "sub-03 ses-nsd11"
    "sub-04 ses-nsd12"
)
read -r SUBJECT SESSION <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"

ml python/3.12.1 uv/0.10.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uv run --frozen python -m experiments.nsd_replication.denoising_check \
    --config "$CONFIG" --subject "$SUBJECT" --session "$SESSION" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
