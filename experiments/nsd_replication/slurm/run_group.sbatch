#!/bin/bash
# Group part of the replication chain, after every run_subject.sbatch job:
# group RSA for nsdpreproc and released (per-subject tables are kept), the
# variability report and the figures.
#
#   sbatch experiments/nsd_replication/slurm/run_group.sbatch <config.toml>
#
#SBATCH --job-name=nsd-group
#SBATCH --partition=russpold,normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE
#SBATCH --chdir=/oak/stanford/groups/russpold/data/NSD/boldtailor
#SBATCH --output=/oak/stanford/groups/russpold/data/NSD/boldtailor/experiments/nsd_replication/slurm/logs/%x-%j.out

set -euo pipefail

CONFIG=$1

ml python/3.12.1 uv/0.10.8
ml biology workbench
export TMPDIR=$L_SCRATCH

run() {
    echo "=== $(date '+%F %T') run $*"
    uv run --frozen python -m experiments.nsd_replication.run "$@" --config "$CONFIG"
}

run metrics --source nsdpreproc
run metrics --source released

echo "=== $(date '+%F %T') pilot_report"
uv run --frozen python -m experiments.nsd_replication.pilot_report \
    --config "$CONFIG" --source nsdpreproc

run figures
echo "=== $(date '+%F %T') done"
