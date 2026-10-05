from types import SimpleNamespace

import numpy as np
from nilearn.glm.first_level import run_glm

from boldtailor.hrf_library import default_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials

from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.lss import (
    lss_canonical,
    lss_run,
    lss_selected,
    n_trials,
)


def test_lss_run_matches_per_trial_ols():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 5))
    nuisance = np.column_stack([np.ones(80), np.linspace(-1, 1, 80)])
    y = rng.normal(size=(80, 3))
    got = lss_run(x, nuisance, y)
    for i in range(5):
        design = np.column_stack([x[:, i], x.sum(1) - x[:, i], nuisance])
        _, results = run_glm(y, design, noise_model="ols")
        want = next(iter(results.values())).theta[0]
        np.testing.assert_allclose(got[i], want, rtol=1e-8)


def test_lss_single_trial_equals_joint_ols():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 1))
    nuisance = np.ones((40, 1))
    y = rng.normal(size=(40, 2))
    joint = np.linalg.lstsq(np.column_stack([x, nuisance]), y, rcond=None)[0][0]
    np.testing.assert_allclose(lss_run(x, nuisance, y)[0], joint, rtol=1e-10)


def test_lss_canonical_shape(synthetic_session):
    indices = np.arange(30)
    data = analysis_data(synthetic_session, indices)
    result = fit_single_trials(data, run_labels=list(synthetic_session.labels))
    betas = lss_canonical(result, synthetic_session.signals)
    assert betas.shape == (len(result.trial_table), 30)
    assert np.all(np.isfinite(betas[:, :20]))


def _selected_fit(session):
    data = analysis_data(session, np.arange(30))
    labels = list(session.labels)
    selection = select_hrfs(
        data,
        library=default_hrf_library(16, seed=0),
        run_labels=labels,
        task_model=session.task_model,
    )
    return fit_selected_hrfs(data, hrf_selection=selection, run_labels=labels)


def _expected_column(result, signals, feature):
    hrf = int(result.design.hrf_indices[feature])
    parts = []
    for run, y in enumerate(signals):
        m = result.design.matrix(run, hrf)
        k = n_trials(result, run)
        parts.append(lss_run(m[:, :k], m[:, k:], np.asarray(y, float)[:, [feature]]))
    return np.vstack(parts)[:, 0]


def test_lss_selected_matches_per_feature_lss(synthetic_session):
    result = _selected_fit(synthetic_session)
    signals = synthetic_session.signals
    betas = lss_selected(result, signals)
    ids = result.design.hrf_indices
    assert betas.shape == (len(result.trial_table), 30)
    valid = np.flatnonzero(ids >= 0)
    chosen = [valid[np.flatnonzero(ids[valid] == h)[0]] for h in np.unique(ids[valid])]
    assert len(chosen) >= 2
    for feature in chosen:
        want = _expected_column(result, signals, feature)
        np.testing.assert_allclose(betas[:, feature], want, rtol=1e-8)


def test_lss_selected_unselected_features_are_nan(synthetic_session):
    result = _selected_fit(synthetic_session)
    ids = result.design.hrf_indices.copy()
    ids[[3, 17]] = -1
    design = SimpleNamespace(hrf_indices=ids, matrix=result.design.matrix)
    patched = SimpleNamespace(design=design, trial_table=result.trial_table)
    betas = lss_selected(patched, synthetic_session.signals)
    assert np.all(np.isnan(betas[:, [3, 17]]))
    assert np.all(np.isfinite(betas[:, [0, 1, 2, 4]]))
