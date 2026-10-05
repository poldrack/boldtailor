"""Comparison of boldtailor and released GLMsingle betas (pure helpers)."""

import pandas as pd

from experiments.nsd_replication.released import alignment_check

LEVELS = ("b1", "b2", "b4")
VERSIONS = tuple(f"{s}:{lv}" for s in ("ppdata", "released") for lv in LEVELS)
_KEY = ["session", "run", "trial", "onset", "image"]


def check_trial_tables(tables_by_version, session):
    """Every version's trials.tsv for a session must be identical."""
    reference_name, reference = next(iter(tables_by_version.items()))
    for name, table in tables_by_version.items():
        if not table.loc[:, _KEY].equals(reference.loc[:, _KEY]):
            raise ValueError(
                f"{session}: trials.tsv of {name} differs from {reference_name}"
            )


def reliability_differences(reliabilities):
    """Per-grayordinate boldtailor minus released reliability per level."""
    return {
        lv: reliabilities[f"ppdata:{lv}"] - reliabilities[f"released:{lv}"]
        for lv in LEVELS
    }


def first_run_length(trials):
    return int((trials["run"] == trials["run"].iloc[0]).sum())


def check_floor(row, subject, floor):
    if floor is not None and row["true_median"] < floor:
        raise ValueError(
            f"{subject} {row['session']}: alignment true_median "
            f"{row['true_median']:.3f} is below the floor {floor}"
        )


def alignment_row(session, ours, released, trials, roi):
    length = first_run_length(trials)
    result = alignment_check(ours, released, length, roi)
    return {"session": session, "run_length": length, **result}


def alignment_frame(rows):
    columns = ["session", "run_length", "true_median", "null_median"]
    return pd.DataFrame(rows, columns=columns)
