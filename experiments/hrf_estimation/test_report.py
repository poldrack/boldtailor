import importlib

import numpy as np
import pandas as pd
import pytest


def test_summary_pools_sse_and_uses_common_valid_vertices():
    rows = []
    for method, losses in [("a", [1, 45]), ("b", [1, 9])]:
        for vertex in [1, 2]:
            for fold, (loss, sst, null) in enumerate(zip(losses, [10, 90], [5, 45]), 1):
                valid = not (method == "b" and vertex == 2 and fold == 2)
                rows.append(
                    dict(
                        method=method,
                        feature_id=vertex,
                        evaluation=f"transfer{fold:02}",
                        sse_full=loss,
                        sse_nuisance=null,
                        sst=sst,
                        r2_task_partial=1 - loss / null if valid else np.nan,
                    )
                )
    fn = importlib.import_module("experiments.hrf_estimation.report").summarize_transfer
    result = fn({11: pd.DataFrame(rows)}).set_index("method")
    assert result.n_common.tolist() == [1, 1]
    assert result.loc["a", "r2_full"] == pytest.approx(0.54)
    assert result.loc["a", "r2_task_partial"] == pytest.approx(0.08)
    assert result.loc["a", "delta_r2_task"] == pytest.approx(0.04)
    assert result.loc["b", "r2_full"] == pytest.approx(0.9)
