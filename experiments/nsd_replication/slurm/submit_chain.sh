#!/bin/bash
# Submit one run_subject.sbatch job per configured subject, then
# run_group.sbatch once all of them have succeeded. Only submits; run it from
# a login node or an interactive session.
#
#   experiments/nsd_replication/slurm/submit_chain.sh <config.toml> [sub-NN=JOBID ...]
#
# A sub-NN=JOBID argument makes that subject's job wait for JOBID to succeed
# (e.g. the betaconvert job producing its released betas).

set -euo pipefail

CONFIG=$1
shift
declare -A AFTER
for pair in "$@"; do
    AFTER[${pair%%=*}]=${pair#*=}
done

DIR=experiments/nsd_replication/slurm
SUBJECTS=$(sed -n 's/^subjects *= *\[\(.*\)\].*/\1/p' "$CONFIG" | tr -d '", ' | sed 's/sub-/ sub-/g')

ids=()
for subject in $SUBJECTS; do
    dep=()
    [[ -n ${AFTER[$subject]:-} ]] && dep=(--dependency="afterok:${AFTER[$subject]}")
    id=$(sbatch --parsable "${dep[@]}" --job-name="nsd-$subject" \
        "$DIR/run_subject.sbatch" "$CONFIG" "$subject")
    echo "$subject: job $id ${dep[*]}"
    ids+=("$id")
done

deps=$(IFS=:; echo "${ids[*]}")
id=$(sbatch --parsable --dependency="afterok:$deps" "$DIR/run_group.sbatch" "$CONFIG")
echo "group: job $id (after $deps)"
