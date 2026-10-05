import dataclasses

import numpy as np
import pytest

from boldtailor.denoising import with_denoising
from boldtailor.hrf_library import default_hrf_library

from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.ladder import (
    LEVELS,
    denoised_confounds,
    fit_ladder,
    select_session_denoising,
)
from experiments.nsd_replication.trials import trial_table


def test_ladder_levels_shapes_and_order(synthetic_session):
    fits = fit_ladder(synthetic_session, block_size=16)
    trials = trial_table(
        synthetic_session.events, synthetic_session.labels, "s", "73k_id"
    )
    assert set(fits) == set(LEVELS)
    for level, fit in fits.items():
        assert fit.betas.shape == (len(trials), 30), level
        assert fit.record["level"] == level
    assert fits["b2"].extras["hrf_indices"].shape == (30,)
    assert fits["b4"].extras["ridge_fraction"].shape == (30,)
    assert not np.allclose(fits["b3"].betas, fits["b2"].betas)
    assert not np.allclose(fits["b4"].betas, fits["b3"].betas)
    fractions = fits["b4"].extras["ridge_fraction"]
    assert not np.all(fractions[np.isfinite(fractions)] == 1.0)


def test_trial_order_matches_trial_table(synthetic_session):
    fits = fit_ladder(synthetic_session, levels=("b1",), block_size=16)
    trials = trial_table(
        synthetic_session.events, synthetic_session.labels, "s", "73k_id"
    )
    np.testing.assert_allclose(fits["b1"].record["onsets"], trials["onset"])


def test_blocks_do_not_change_results(synthetic_session):
    a = fit_ladder(synthetic_session, levels=("b1", "b2"), block_size=7)
    b = fit_ladder(synthetic_session, levels=("b1", "b2"), block_size=30)
    for level in ("b1", "b2"):
        np.testing.assert_allclose(a[level].betas, b[level].betas, rtol=1e-6, atol=1e-8)


def test_denoised_confounds_match_with_denoising(synthetic_session):
    library = default_hrf_library(32, seed=0)
    result = select_session_denoising(
        synthetic_session, synthetic_session.task_model, library
    )
    full = analysis_data(synthetic_session, np.arange(30))
    expected = with_denoising(full, result).confounds
    for got, want in zip(
        denoised_confounds(synthetic_session, result), expected, strict=True
    ):
        assert list(got.columns) == list(want.columns)
        np.testing.assert_array_equal(got.to_numpy(), want.to_numpy())


@pytest.mark.xfail(strict=True, reason="needs Task 8 metrics")
def test_fitted_hrf_improves_reliability_on_delayed_hrf(synthetic_session):
    from experiments.nsd_replication.metrics import voxel_reliability
    from experiments.nsd_replication.betas import zscore
    from experiments.nsd_replication.trials import images_with, repetition_array

    fits = fit_ladder(synthetic_session, levels=("b1", "b2"), block_size=30)
    trials = trial_table(
        synthetic_session.events, synthetic_session.labels, "s", "73k_id"
    )
    images = images_with(trials, 3)
    rel = {
        level: np.nanmedian(
            voxel_reliability(
                repetition_array(zscore(fits[level].betas), trials, images)
            )[:20]
        )
        for level in ("b1", "b2")
    }
    assert rel["b2"] > rel["b1"]


def test_unsorted_onsets_raise_with_run_label(synthetic_session):
    events = list(synthetic_session.events)
    events[2] = events[2].iloc[::-1].reset_index(drop=True)
    session = dataclasses.replace(synthetic_session, events=tuple(events))
    with pytest.raises(ValueError, match=synthetic_session.labels[2]):
        fit_ladder(session, levels=("b1",), block_size=30)
