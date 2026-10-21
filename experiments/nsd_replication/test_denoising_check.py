import numpy as np
import pandas as pd

from experiments.nsd_replication import denoising_check as dc
from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.ladder import _labels, select_session_denoising
from boldtailor.hrf_library import default_hrf_library


def test_badness_scores_each_trial_against_other_runs():
    trials = pd.DataFrame(
        dict(run=["r1", "r1", "r2", "r2"], image=[1, 2, 1, 3])
    )  # images 2 and 3 appear in one run only and are skipped
    betas = np.array([[1.0, 0.0], [5.0, 9.0], [3.0, 2.0], [7.0, 9.0]])
    expected = [(1 - 3) ** 2 + (3 - 1) ** 2, (0 - 2) ** 2 + (2 - 0) ** 2]
    np.testing.assert_allclose(dc.badness(betas, trials), expected)


def test_full_task_r2_nests_onoff(synthetic_session):
    library = default_hrf_library()
    data = analysis_data(synthetic_session, np.arange(30))
    labels = _labels(synthetic_session)
    denoising = select_session_denoising(
        synthetic_session, synthetic_session.task_model, library
    )
    full = dc.task_r2(data, library, synthetic_session.task_model, labels)
    assert len(synthetic_session.task_model.regressor_names) > 1
    assert np.all(full >= np.asarray(denoising.onoff_r2) - 1e-12)


def test_check_scores_every_pool_rule_and_count(synthetic_session):
    table = dc.check(synthetic_session)
    assert set(table["pool"]) == set(dc.POOLS) and set(table["rule"]) == set(dc.RULES)
    assert len(table) == len(dc.POOLS) * len(dc.RULES) * len(dc.COUNTS)
    zero = table[table["count"] == 0]
    assert np.isfinite(zero["perf"]).all() and (zero["curve"] == 0).all()
    for key in (f"{p}_{r}" for p in dc.POOLS for r in dc.RULES):
        assert table.attrs[key] in dc.COUNTS
    assert table.attrs["n_scoring"] > 0
