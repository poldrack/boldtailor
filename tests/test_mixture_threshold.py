"""Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold).

The oracle fits its own scikit-learn ``GaussianMixture`` with GLMsingle's
settings, builds the grid with a test-local port of GLMsingle's
``robustrange``, and reads posteriors from ``predict_proba``; a second check
evaluates the reported parameters with ``scipy.stats.norm``. Neither calls
the production helper to derive expected values.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

import boldtailor._mixture_threshold as module
from boldtailor._mixture_threshold import (
    MixtureThreshold,
    mixture_threshold,
    robust_range,
)

# GLMsingle's rule: fit 2 components to the raw values (tol 1e-10,
# reg_covar 0, 3 initialisations; random_state 0 added for reproducibility),
# grid of 500 points over robustrange widened to include both means; the
# tail component dominates the grid's right end; the threshold is the
# rightmost grid point where its posterior is <= 0.5.
SETTINGS = dict(n_components=2, tol=1e-10, reg_covar=0.0, n_init=3, random_state=0)
GRID_POINTS = 500


def oracle_robustrange(m):
    """Line-by-line port of GLMsingle utils/robustrange.py (returns [mn, mx])."""
    absmn, absmx = np.min(m), np.max(m)
    vals = np.percentile(m, [0.1, 10, 50, 90, 99.9])
    pmn = vals[2] - 5 * (vals[2] - vals[1])
    pmx = vals[2] + 5 * (vals[3] - vals[2])
    rerun = False
    if vals[4] <= pmx:
        finalmx = absmx if absmx <= vals[2] + 1.1 * (vals[4] - vals[2]) else vals[4]
    else:
        rerun = True
        m = m[np.logical_not(m > pmx)]
    if vals[0] >= pmn:
        finalmn = absmn if absmn >= vals[2] - 1.1 * (vals[2] - vals[0]) else vals[0]
    else:
        rerun = True
        m = m[np.logical_not(m < pmn)]
    return oracle_robustrange(m) if rerun else [finalmn, finalmx]


def bimodal(seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.02, size=300)
    task = rng.normal(0.4, 0.1, size=100)
    return np.concatenate([noise, task]) + shift


def wide_lower(seed=1):
    """Lower-mean component is the wider one."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 0.1, 300), rng.normal(0.3, 0.01, 100)])


def oracle_fit(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return GaussianMixture(**SETTINGS).fit(fitted[:, None])


def oracle(values, fitted=None):
    finite = values[np.isfinite(values)]
    model = oracle_fit(finite if fitted is None else fitted)
    means = model.means_.ravel()
    low, high = oracle_robustrange(finite)
    grid = np.linspace(min(low, means.min()), max(high, means.max()), GRID_POINTS)
    posterior = model.predict_proba(grid[:, None])
    tail = 0 if posterior[-1, 0] > 0.5 else 1
    below = np.flatnonzero(posterior[:, tail] <= 0.5)
    order = np.argsort(means)
    return dict(
        threshold=grid[below[-1]],
        means=means[order],
        sds=np.sqrt(model.covariances_.ravel()[order]),
        weights=model.weights_[order],
        tail=int(np.flatnonzero(order == tail)[0]),
        step=grid[1] - grid[0],
        converged=bool(model.converged_),
        n_iter=int(model.n_iter_),
    )


def assert_matches(result, expected):
    np.testing.assert_allclose(result.threshold, expected["threshold"], atol=1e-12)
    np.testing.assert_allclose(result.means, expected["means"], atol=1e-12)
    np.testing.assert_allclose(result.sds, expected["sds"], atol=1e-12)
    np.testing.assert_allclose(result.weights, expected["weights"], atol=1e-12)
    assert result.tail == expected["tail"]
    assert result.converged == expected["converged"]
    assert result.n_iter == expected["n_iter"]


@pytest.mark.parametrize("values", [bimodal(), wide_lower()], ids=["tail", "wide"])
def test_threshold_matches_an_independent_mixture_fit(values):
    result = mixture_threshold(values)
    assert isinstance(result, MixtureThreshold)
    assert_matches(result, oracle(values))
    assert result.n_values == result.n_fitted == len(values)


@pytest.mark.parametrize("values", [bimodal(), wide_lower()], ids=["tail", "wide"])
def test_reported_parameters_put_the_tail_posterior_crossing_at_the_threshold(
    values,
):
    result = mixture_threshold(values)
    step = oracle(values)["step"]

    def tail_posterior(x):
        dens = [
            w * norm.pdf(x, m, s)
            for w, m, s in zip(result.weights, result.means, result.sds)
        ]
        return dens[result.tail] / (dens[0] + dens[1])

    assert tail_posterior(result.threshold) <= 0.5 + 1e-9
    assert tail_posterior(result.threshold + step) > 0.5 - 1e-9


def test_bimodal_threshold_lies_between_the_modes():
    result = mixture_threshold(bimodal())
    assert 0.0 < result.threshold < 0.4
    assert result.means[0] == pytest.approx(0.0, abs=0.01)
    assert result.means[1] == pytest.approx(0.4, abs=0.05)
    assert result.tail == 1


def test_threshold_shifts_with_a_common_shift_of_both_modes():
    base, shifted = mixture_threshold(bimodal()), mixture_threshold(bimodal(0, 0.05))
    assert shifted.threshold == pytest.approx(base.threshold + 0.05, abs=1e-9)
    np.testing.assert_allclose(shifted.means, np.add(base.means, 0.05), atol=1e-9)
    np.testing.assert_allclose(shifted.sds, base.sds, atol=1e-9)


def test_threshold_is_deterministic():
    values = bimodal(3)
    assert mixture_threshold(values) == mixture_threshold(values.copy())


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


# ---- unimodal and heavy-tailed statistics ---------------------------------------


def test_unimodal_normal_completes_and_records_convergence():
    # Run under -W error: ConvergenceWarning must not escape.
    values = np.random.default_rng(0).normal(0.0, 0.02, 2000)
    result = mixture_threshold(values)
    assert_matches(result, oracle(values))
    assert isinstance(result.converged, bool) and result.n_iter >= 1
    record = result.to_dict()
    assert record["converged"] == result.converged
    assert record["n_iter"] == result.n_iter


def test_heavy_tailed_unimodal_gives_a_stable_pool_fraction():
    fractions = []
    for seed in range(10):
        values = np.random.default_rng(seed).standard_t(3, 2000) * 0.02
        fractions.append(np.mean(values <= mixture_threshold(values).threshold))
    assert min(fractions) > 0.5
    assert max(fractions) - min(fractions) < 0.2


# ---- robust range and subsampling ------------------------------------------------


def test_robust_range_excludes_an_extreme_outlier():
    values = np.random.default_rng(4).normal(0.0, 1.0, 5000)
    with_outlier = np.append(values, 1e6)
    low, high = robust_range(with_outlier)
    assert [low, high] == pytest.approx(oracle_robustrange(with_outlier), abs=0)
    assert high < 10
    assert robust_range(values) == pytest.approx(oracle_robustrange(values), abs=0)


def test_robust_range_of_clean_data_is_the_data_range():
    values = np.linspace(-1.0, 1.0, 101)
    assert robust_range(values) == (-1.0, 1.0)


def test_large_inputs_are_subsampled_deterministically(monkeypatch):
    monkeypatch.setattr(module, "MAX_VALUES", 500)
    values = np.concatenate([bimodal(7), bimodal(8), bimodal(9), bimodal(10)])
    pick = np.sort(np.random.default_rng(0).choice(len(values), 500, replace=False))
    result = mixture_threshold(values)
    assert_matches(result, oracle(values, fitted=values[pick]))
    assert result.n_values == len(values) and result.n_fitted == 500
    assert mixture_threshold(values) == result


# ---- failures ---------------------------------------------------------------------


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


def test_failed_fit_raises_a_value_error():
    # Two tied clusters: with reg_covar=0 the covariances are singular.
    values = np.repeat([0.0, 1.0], 50)
    with pytest.raises(ValueError, match="mixture fit failed"):
        mixture_threshold(values)


class _AlwaysTail(GaussianMixture):
    def predict_proba(self, x):
        return np.full((len(x), 2), 0.5) + np.array([0.3, -0.3])


def test_tail_posterior_never_at_or_below_one_half_raises(monkeypatch):
    monkeypatch.setattr(module, "GaussianMixture", _AlwaysTail)
    with pytest.raises(ValueError, match="0.5"):
        mixture_threshold(bimodal())


def test_record_names_method_settings_and_fit():
    result = mixture_threshold(bimodal())
    record = result.to_dict()
    assert record["method"] == "gaussian_mixture_tail_threshold"
    assert record["estimator"] == "sklearn.mixture.GaussianMixture"
    assert "findtailthreshold" in record["rule"]
    for key, value in SETTINGS.items():
        assert record[key] == value
    assert record["grid_points"] == GRID_POINTS
    assert record["grid_range"] == "robustrange widened to include both means"
    assert record["max_values"] == 1_000_000 and record["subsample_seed"] == 0
    for key in ("threshold", "tail", "n_values", "n_fitted", "converged", "n_iter"):
        assert record[key] == getattr(result, key)
    assert record["means"] == list(result.means)
    assert record["sds"] == list(result.sds)
    assert record["weights"] == list(result.weights)
    assert isinstance(record["sklearn_version"], str)
    assert "standardized" not in record
