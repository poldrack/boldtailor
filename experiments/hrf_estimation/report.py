"""Descriptive transfer summaries on vertices valid for every compared method."""

import pandas as pd


def summarize_transfer(session_tables):
    rows = []
    for session, table in session_tables.items():
        transfer = table[table.evaluation.str.startswith("transfer")]
        methods = sorted(transfer.method.unique())
        count = transfer.evaluation.nunique()
        if not methods:
            raise ValueError("no transfer evaluations to summarize")
        valid = transfer.groupby(["method", "feature_id"]).r2_task_partial.apply(
            lambda values: len(values) == count and values.notna().all()
        )
        common = set.intersection(
            *[set(valid.loc[method][valid.loc[method]].index) for method in methods]
        )
        if not common:
            raise ValueError(
                "no common valid vertices across methods and transfer folds"
            )
        for method in methods:
            totals = (
                transfer[transfer.method == method]
                .groupby("feature_id")[["sse_full", "sse_nuisance", "sst"]]
                .sum()
                .loc[sorted(common)]
            )
            rows.append(
                dict(
                    session=session,
                    method=method,
                    n_common=len(common),
                    r2_full=float((1 - totals.sse_full / totals.sst).median()),
                    r2_task_partial=float(
                        (1 - totals.sse_full / totals.sse_nuisance).median()
                    ),
                    delta_r2_task=float(
                        ((totals.sse_nuisance - totals.sse_full) / totals.sst).median()
                    ),
                )
            )
    return pd.DataFrame(rows)
