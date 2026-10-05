import numpy as np
from nilearn.glm.first_level import run_glm

from boldtailor.single_trial import fit_single_trials

from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.lss import lss_canonical, lss_run


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
