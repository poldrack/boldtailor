"""Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold).

The oracle fits its own scikit-learn ``GaussianMixture`` with the documented
settings and reads posteriors from ``predict_proba`` on the documented grid;
a second check evaluates the reported parameters with ``scipy.stats.norm``.
Neither calls the production helper to derive expected values.
"""

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

import boldtailor._mixture_threshold as module
from boldtailor._mixture_threshold import MixtureThreshold, mixture_threshold

# Documented rule: z-score the finite values, fit 2 components (random_state
# 0, 3 initialisations, tol 1e-10, up to 1000 EM steps), evaluate the
# lower-mean component's posterior on 10001 evenly spaced points from the
# smallest to the largest z value, and return the rightmost grid point whose
# posterior is >= 0.5 while the next point's is < 0.5.
SETTINGS = dict(n_components=2, random_state=0, n_init=3, tol=1e-10, max_iter=1000)
GRID_POINTS = 10001


def bimodal(seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.02, size=300)
    task = rng.normal(0.4, 0.1, size=100)
    return np.concatenate([noise, task]) + shift


def wide_lower(seed=1):
    """Lower-mean component is the wider one: its posterior rises again."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 0.1, 300), rng.normal(0.3, 0.01, 100)])


def oracle(values):
    finite = values[np.isfinite(values)]
    center, scale = finite.mean(), finite.std()
    z = (finite - center) / scale
    model = GaussianMixture(**SETTINGS).fit(z[:, None])
    lower = int(np.argmin(model.means_.ravel()))
    grid = np.linspace(z.min(), z.max(), GRID_POINTS)
    posterior = model.predict_proba(grid[:, None])[:, lower]
    falls = np.flatnonzero((posterior[:-1] >= 0.5) & (posterior[1:] < 0.5))
    order = np.argsort(model.means_.ravel())
    return dict(
        threshold=center + scale * grid[falls[-1]],
        means=center + scale * model.means_.ravel()[order],
        sds=scale * np.sqrt(model.covariances_.ravel()[order]),
        weights=model.weights_[order],
        step=scale * (grid[1] - grid[0]),
    )


@pytest.mark.parametrize("values", [bimodal(), wide_lower()], ids=["tail", "wide"])
def test_threshold_matches_an_independent_mixture_fit(values):
    result = mixture_threshold(values)
    expected = oracle(values)
    assert isinstance(result, MixtureThreshold)
    np.testing.assert_allclose(result.threshold, expected["threshold"], atol=1e-12)
    np.testing.assert_allclose(result.means, expected["means"], atol=1e-12)
    np.testing.assert_allclose(result.sds, expected["sds"], atol=1e-12)
    np.testing.assert_allclose(result.weights, expected["weights"], atol=1e-12)
    assert result.n_values == len(values)


@pytest.mark.parametrize("values", [bimodal(), wide_lower()], ids=["tail", "wide"])
def test_reported_parameters_put_the_posterior_crossing_at_the_threshold(values):
    result = mixture_threshold(values)
    step = oracle(values)["step"]

    def lower_posterior(x):
        dens = [
            w * norm.pdf(x, m, s)
            for w, m, s in zip(result.weights, result.means, result.sds)
        ]
        return dens[0] / (dens[0] + dens[1])

    assert result.means[0] < result.means[1]
    assert lower_posterior(result.threshold) >= 0.5 - 1e-9
    assert lower_posterior(result.threshold + step) < 0.5 + 1e-9
    # Rightmost fall: beyond the threshold the posterior never regains 0.5
    # while still inside the data range below the upper mean's tail start.
    above = np.linspace(result.threshold + step, result.means[1], 200)
    assert (lower_posterior(above) < 0.5 + 1e-9).all()


def test_bimodal_threshold_lies_between_the_modes():
    result = mixture_threshold(bimodal())
    assert 0.0 < result.threshold < 0.4
    assert result.means[0] == pytest.approx(0.0, abs=0.01)
    assert result.means[1] == pytest.approx(0.4, abs=0.05)


def test_threshold_shifts_with_a_common_shift_of_both_modes():
    base, shifted = mixture_threshold(bimodal()), mixture_threshold(bimodal(0, 0.05))
    assert shifted.threshold == pytest.approx(base.threshold + 0.05, abs=1e-9)
    np.testing.assert_allclose(shifted.means, np.add(base.means, 0.05), atol=1e-9)
    np.testing.assert_allclose(shifted.sds, base.sds, atol=1e-9)
    np.testing.assert_allclose(shifted.weights, base.weights, atol=1e-9)


def test_threshold_is_deterministic():
    values = bimodal(3)
    first, second = mixture_threshold(values), mixture_threshold(values.copy())
    assert first == second


def test_nonfinite_values_are_ignored():
    values = bimodal()
    padded = np.concatenate([values, [np.nan, np.inf, -np.inf]])
    result = mixture_threshold(padded)
    assert result == mixture_threshold(values)
    assert result.n_values == len(values)


def test_input_is_not_modified():
    values = bimodal()
    before = values.copy()
    mixture_threshold(values)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize(
    "values",
    [np.array([]), np.array([0.2]), np.full(50, 0.1), np.array([np.nan, 0.3, 0.3])],
)
def test_fewer_than_two_distinct_values_raise(values):
    with pytest.raises(ValueError, match="distinct"):
        mixture_threshold(values)


@pytest.mark.parametrize("values", [np.zeros((10, 2)), ["a", "b"], None])
def test_values_must_be_a_real_vector(values):
    with pytest.raises(ValueError, match="vector"):
        mixture_threshold(values)


class _NeverCrossing(GaussianMixture):
    def predict_proba(self, x):
        lower = int(np.argmin(self.means_.ravel()))
        posterior = np.full((len(x), 2), 0.7)
        posterior[:, lower] = 0.3
        return posterior


def test_posterior_that_never_falls_through_one_half_raises(monkeypatch):
    monkeypatch.setattr(module, "GaussianMixture", _NeverCrossing)
    with pytest.raises(ValueError, match="0.5"):
        mixture_threshold(bimodal())


def test_record_names_method_settings_and_fit():
    result = mixture_threshold(bimodal())
    record = result.to_dict()
    assert record["method"] == "gaussian_mixture_tail_threshold"
    assert record["estimator"] == "sklearn.mixture.GaussianMixture"
    assert record["rule"] == "rightmost 50/50 posterior of the lower-mean component"
    for key, value in SETTINGS.items():
        assert record[key] == value
    assert record["grid_points"] == GRID_POINTS
    assert record["standardized"] is True
    assert record["threshold"] == result.threshold
    assert record["means"] == list(result.means)
    assert record["sds"] == list(result.sds)
    assert record["weights"] == list(result.weights)
    assert record["n_values"] == result.n_values
    assert isinstance(record["sklearn_version"], str)
    assert all(isinstance(v, float) for v in record["means"] + record["sds"])
