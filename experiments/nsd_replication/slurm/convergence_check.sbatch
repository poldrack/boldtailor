#!/bin/bash
# R6 sensitivity to LinearSVC's iteration limit: refits the cells whose folds
# did not converge at the default max_iter (1000) with a larger limit and
# writes one row per fold to <output_dir>/checks/nsdpreproc/<subject>/.
#
#   sbatch experiments/nsd_replication/slurm/convergence_check.sbatch [config.toml]
#
#SBATCH --job-name=r6-convergence
#SBATCH --partition=russpold,normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
# Same CPU generations as run_all.sbatch, so inputs match the pipeline's.
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE
#SBATCH --chdir=/oak/stanford/groups/russpold/data/NSD/boldtailor
#SBATCH --output=/oak/stanford/groups/russpold/data/NSD/boldtailor/experiments/nsd_replication/slurm/logs/%x-%j.out

set -euo pipefail

CONFIG=${1:-experiments/nsd_replication/configs/pilot.toml}

ml python/3.12.1 uv/0.10.8

# One single-threaded liblinear fit per core: 4 cells x 3 folds.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uv run --frozen python -m experiments.nsd_replication.convergence_check \
    --config "$CONFIG" --subject sub-07 --max-iter 10000 --n-jobs 12 \
    --cells b1:0.3 b2:0.3 b1:0.0 lss-assume:0.0
