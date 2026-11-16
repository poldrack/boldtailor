"""Independent likelihood and seeded model checks for the public adapter."""

import numpy as np
import pytest

from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from tests.hrf_reml_fixtures import small_reml_problem
from tests.reml_oracles import dense_fit, dense_criterion


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_public_selection_matches_dense_restricted_likelihood(rho):
    """Independent error contrasts recover the winner and canonical improvement."""
    data, library, designs = small_reml_problem(n_features=1, rho=rho)
    fixed = [np.column_stack([np.ones(48), designs[0].sum(axis=1)])] * 2
    criteria = []
    for z in designs:
        ratio, _ = dense_fit(data.signals, fixed, [z] * 2, rho=rho)
        criteria.append(dense_criterion(data.signals, fixed, [z] * 2, ratio[0], rho)[0])
    result = select_hrfs(
        data, library=library, method="reml", reml_options=RemlHrfOptions(rho=rho)
    )
    assert result.hrf_indices[0] == np.argmin(criteria)
    np.testing.assert_allclose(
        result.reml.delta_criterion, criteria[0] - min(criteria), atol=2e-5
    )


def test_waveform_recovery_improves_with_trial_signal():
    """Strong matched trial variance recovers the generating waveform more often."""
    scores = []
    for effect in (0.0, 0.3, 4.0):
        data, library, designs = small_reml_problem(n_features=96, effect=effect)
        result = select_hrfs(
            data, library=library, method="reml", reml_options=RemlHrfOptions(rho=0.2)
        )
        correlations = np.array(
            [np.corrcoef(z.ravel(), designs[1].ravel())[0, 1] for z in designs]
        )
        recovered = np.zeros(96, dtype=bool)
        valid = result.hrf_indices >= 0
        recovered[valid] = correlations[result.hrf_indices[valid]] > 0.99
        scores.append(recovered.mean())
        assert np.isfinite(result.reml.selection.fit.criterion[valid]).all()
    # Worst-case binomial SE <= 0.5/sqrt(96). Allow three SE below 90% recovery.
    assert scores[2] > 0.9 - 3 * 0.5 / np.sqrt(96)
    # Independent-proportions bound is conservative despite the paired draws.
    assert scores[2] - scores[0] > 3 * np.sqrt(0.5 / 96)
    assert scores[2] >= scores[1]


def test_reference_mean_misspecification_changes_apparent_trial_variance():
    """A wrong mean shape can be absorbed as trial covariance even with no effects."""
    data, library, _ = small_reml_problem(n_features=64, effect=0, mean_id=1)
    fits = [
        select_hrfs(
            data,
            library=library,
            method="reml",
            reml_options=RemlHrfOptions(rho=0.2, reference_hrf_id=ref),
        )
        for ref in (0, 1)
    ]
    # Fixed effects alone generated these signals: supported winners are not a
    # significance test. Compare pooled apparent variances, retaining zero fits.
    variances = [
        np.nan_to_num(r.reml.selection.fit.effect_variance, nan=0).mean() for r in fits
    ]
    assert variances[0] > variances[1] + 1.0
