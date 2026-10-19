#!/bin/bash
# Per-subject part of the replication chain for one configured subject, in
# two jobs that each fit the 48-hour QOS limit: "fit" runs the nsdpreproc
# fits (~2.75 h per session), "rest" the metrics, features, and the released
# and comparison stages. Group stages (RSA, pilot report, figures) run
# afterwards in run_group.sbatch; submit_chain.sh submits all of them. Every
# stage skips work already written, so a resubmission resumes.
#
#   sbatch --time=40:00:00 experiments/nsd_replication/slurm/run_subject.sbatch <config.toml> <sub-NN> fit
#   sbatch --time=24:00:00 experiments/nsd_replication/slurm/run_subject.sbatch <config.toml> <sub-NN> rest
#
#SBATCH --job-name=nsd-subject
#SBATCH --partition=russpold,normal
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# AVX-512 nodes only, as in run_all.sbatch: the HRF library fingerprint
# differs between AVX2 and AVX-512 CPUs.
#SBATCH --constraint=CPU_GEN:SKX|CPU_GEN:SIE
#SBATCH --chdir=/oak/stanford/groups/russpold/data/NSD/boldtailor
#SBATCH --output=/oak/stanford/groups/russpold/data/NSD/boldtailor/experiments/nsd_replication/slurm/logs/%x-%j.out

set -euo pipefail

CONFIG=$1
SUBJECT=$2
PART=$3

ml python/3.12.1 uv/0.10.8
ml biology workbench

# See run_all.sbatch: n_jobs workers share the cores as BLAS threads.
N_JOBS=$(sed -n 's/^n_jobs *= *\([0-9]*\).*/\1/p' "$CONFIG")
THREADS=$(( SLURM_CPUS_PER_TASK / ${N_JOBS:-1} ))
(( THREADS < 1 )) && THREADS=1
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export TMPDIR=$L_SCRATCH

run() {
    echo "=== $(date '+%F %T') $SUBJECT run $*"
    uv run --frozen python -m experiments.nsd_replication.run "$@" \
        --config "$CONFIG" --subjects "$SUBJECT"
}

case $PART in
    fit)
        run fit      --source nsdpreproc
        ;;
    rest)
        run metrics  --source nsdpreproc
        run features
        run fit      --source released
        run metrics  --source released
        run metrics  --source comparison
        ;;
    *)
        echo "part must be fit or rest, not $PART" >&2
        exit 2
        ;;
esac
echo "=== $(date '+%F %T') $SUBJECT $PART done"
