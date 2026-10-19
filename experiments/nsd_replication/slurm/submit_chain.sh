#!/bin/bash
# Submit run_subject.sbatch per configured subject (its fits, then the rest
# once the fits succeed), then run_group.sbatch once every subject is done. Only submits; run it from
# a login node or an interactive session.
#
#   experiments/nsd_replication/slurm/submit_chain.sh <config.toml> [sub-NN=JOBID ...]
#
# A sub-NN=JOBID argument makes that subject's fits wait for JOBID to succeed
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
    fit=$(sbatch --parsable ${dep[@]+"${dep[@]}"} --time=40:00:00 \
        --job-name="nsd-$subject-fit" "$DIR/run_subject.sbatch" "$CONFIG" "$subject" fit)
    rest=$(sbatch --parsable --dependency="afterok:$fit" --time=24:00:00 \
        --job-name="nsd-$subject-rest" "$DIR/run_subject.sbatch" "$CONFIG" "$subject" rest)
    echo "$subject: fit job $fit ${dep[*]+${dep[*]}}, rest job $rest (after $fit)"
    ids+=("$rest")
done

deps=$(IFS=:; echo "${ids[*]}")
id=$(sbatch --parsable --dependency="afterok:$deps" "$DIR/run_group.sbatch" "$CONFIG")
echo "group: job $id (after $deps)"
