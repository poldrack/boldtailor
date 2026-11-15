"""The experiment REML adapters must delegate HRF selection to boldtailor.reml."""

from dataclasses import replace

import numpy as np
import pytest

from boldtailor.reml import RemlGroup, fit_reml, select_random_design
from experiments.hrf_estimation import pilot
from experiments.hrf_estimation import reml_selection as api
from experiments.hrf_estimation.sensitivity import ar1_noise
from experiments.hrf_estimation.test_library_cv import planted_runs

NOISE_KEYS = {
    "rho",
    "corrected_median",
    "raw_median",
    "n_valid",
    "clipped",
    "training_run_ids",
    "reference",
}


def _with_white_noise(runs, scale, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        replace(r, signals=r.signals + rng.normal(0, scale, r.signals.shape))
        for r in runs
    )


def _with_ar1_noise(runs, rho, scale, seed):
    rng = np.random.default_rng(seed)
    noisy = []
    for run in runs:
        scans, features = run.signals.shape
        noise = np.column_stack([ar1_noise(scans, rho, rng) for _ in range(features)])
        noisy.append(replace(run, signals=run.signals + scale * noise))
    return tuple(noisy)


def test_planted_hrf_is_selected_iid(planted_runs):
    runs, library = planted_runs
    result = api.fit_library_reml(_with_white_noise(runs, 0.1, 99), library, {})
    assert [k.id for k in result.kernels] == [0, 1]
    assert result.status == ("ok", "ok")
    assert np.all(np.asarray(result.diagnostics["variance_ratio"]) > 0)
    assert result.training_run_ids == tuple(r.run_id for r in runs)
    flat = tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs)
    empty = api.fit_library_reml(flat, library, {})
    assert empty.kernels == (None, None)
    assert empty.status == ("no_signal", "no_signal")


def test_planted_hrf_is_selected_ar1(planted_runs):
    runs, library = planted_runs
    noisy = _with_ar1_noise(runs, 0.5, 0.1, 31)
    result = api.fit_ar1_reml(noisy, library, {})
    assert [k.id for k in result.kernels] == [0, 1]
    assert result.status == ("ok", "ok")
    noise = result.diagnostics["noise"]
    assert noise == api.estimate_noise(noisy, "amplitude")
    assert noise["rho"] > 0.2
    assert result.diagnostics["rho"] == [noise["rho"]] * 2


def _package_selection(runs, library, rho):
    groups = [RemlGroup(r.signals, r.confounds.to_numpy(dtype=float)) for r in runs]
    candidates = (
        [api.random_design(run, k) for run in runs] for k in library.candidates
    )
    return select_random_design(groups, candidates, rho=rho)


def test_estimate_matches_package_selection(planted_runs):
    runs, library = planted_runs
    runs = _with_white_noise(runs, 0.5, 12)
    rho = np.array([0.1, 0.4])
    expected = _package_selection(runs, library, rho)
    result = api.select_hrf(runs, library, api.random_design, rho)
    kernels = [None if i < 0 else library.candidates[i] for i in expected.winner]
    assert result.kernels == tuple(kernels)
    assert result.status == expected.fit.status
    d = result.diagnostics
    np.testing.assert_array_equal(d["variance_ratio"], expected.fit.ratio)
    np.testing.assert_array_equal(d["sigma2"], expected.fit.noise_variance)
    np.testing.assert_array_equal(d["criterion"], expected.fit.criterion)
    np.testing.assert_array_equal(d["at_upper"], expected.fit.at_upper)
    np.testing.assert_array_equal(d["unrefined"], expected.fit.unrefined)
    np.testing.assert_array_equal(d["any_at_upper"], expected.any_at_upper)
    np.testing.assert_array_equal(d["any_unrefined"], expected.any_unrefined)
    np.testing.assert_array_equal(d["provisional"], expected.any_unrefined)
    np.testing.assert_array_equal(d["candidate_gap"], expected.gap)
    zero = [s == "variance_zero" for s in expected.fit.status]
    assert d["at_zero"] == zero
    assert d["rho"] == rho.tolist()
    assert d["library_fingerprint"] == library.fingerprint


def _argmin_kernels(runs, library, mode):
    """Per-feature best ok candidate and its criterion, one package fit each."""
    groups = [RemlGroup(r.signals, r.confounds.to_numpy(dtype=float)) for r in runs]
    fits = [
        fit_reml(groups, [api.random_design(r, k, mode) for r in runs])
        for k in library.candidates
    ]
    ok = np.array([[s == "ok" for s in f.status] for f in fits])
    scores = np.where(ok, np.array([f.criterion for f in fits]), np.inf)
    best = np.argmin(scores, axis=0)
    kernels = tuple(
        library.candidates[i] if ok[i, v] else None for v, i in enumerate(best)
    )
    return kernels, scores.min(axis=0)


def test_no_rt_mode_has_four_random_effects(planted_runs):
    runs, library = planted_runs
    runs = runs[:2]
    for mode, count in [("none", 4), ("amplitude", 5)]:
        result = api.fit_library_reml(runs, library, {"rt_mode": mode})
        assert result.diagnostics["rt_mode"] == mode
        assert result.diagnostics["n_random_effects_per_run"] == count
        assert str(count) in result.diagnostics["random_effects"]
        kernels, criterion = _argmin_kernels(runs, library, mode)
        assert result.kernels == kernels
        won = np.isfinite(criterion)
        assert won.any()
        np.testing.assert_array_equal(
            np.asarray(result.diagnostics["criterion"])[won], criterion[won]
        )


def test_noise_estimate_keys_and_bounds(planted_runs):
    runs, _ = planted_runs
    moderate = api.estimate_noise(_with_ar1_noise(runs, 0.3, 1.0, 5), "amplitude")
    assert set(moderate) == NOISE_KEYS
    assert moderate["training_run_ids"] == [r.run_id for r in runs]
    assert moderate["n_valid"] == 8
    assert not moderate["clipped"]
    assert moderate["rho"] == pytest.approx(moderate["corrected_median"])
    assert 0.1 < moderate["rho"] < 0.5
    strong = api.estimate_noise(_with_ar1_noise(runs, 0.97, 1.0, 6), "amplitude")
    assert strong["clipped"]
    assert strong["corrected_median"] > 0.8
    assert strong["rho"] == 0.8


@pytest.mark.parametrize("method", ["remlcomponents", "remlshared", "remlcombined"])
def test_removed_variants_are_rejected(planted_runs, method):
    runs, library = planted_runs
    with pytest.raises(ValueError, match="unknown method"):
        pilot._estimate(runs, library, method, {})
