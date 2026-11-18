"""The experiment REML adapters delegate HRF selection to the package."""

from dataclasses import replace

import numpy as np
import pytest

from pathlib import Path

from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from experiments.hrf_estimation import package_selection
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
    """A non-canonical HRF is selected; a canonical one falls back to the reference.

    The package fits the task mean with the reference HRF and identifies shape
    from trial deviations, so a feature the canonical HRF already explains has
    zero random-effect variance and no likelihood winner.
    """
    runs, library = planted_runs
    result = api.fit_library_reml(_with_white_noise(runs, 0.1, 99), library, {})
    assert [k.id for k in result.kernels] == [0, 1]
    assert result.status == ("variance_zero", "ok")
    assert result.diagnostics["canonical_fallback"] == [True, False]
    assert result.diagnostics["variance_ratio"][1] > 0
    assert result.training_run_ids == tuple(r.run_id for r in runs)
    flat = tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs)
    empty = api.fit_library_reml(flat, library, {})
    assert empty.kernels == (None, None)
    assert empty.status == ("no_residual_signal", "no_residual_signal")
    assert empty.diagnostics["canonical_fallback"] == [False, False]


def test_planted_hrf_is_selected_ar1(planted_runs):
    """Verify planted HRF is selected AR(1)."""
    runs, library = planted_runs
    noisy = _with_ar1_noise(runs, 0.5, 0.1, 31)
    result = api.fit_ar1_reml(noisy, library, {})
    assert [k.id for k in result.kernels] == [0, 1]
    assert result.status[1] == "ok"
    noise = result.diagnostics["noise"]
    assert noise == api.estimate_noise(noisy, "amplitude")
    assert noise["rho"] > 0.2
    assert result.diagnostics["rho"] == [noise["rho"]] * 2


def _package_result(runs, library, mode, rho):
    data, model = package_selection.selection_data(runs, library, mode)
    return select_hrfs(
        data,
        library=library,
        task_model=model,
        method="reml",
        reml_options=RemlHrfOptions(rho=rho),
        run_labels=[r.run_id for r in runs],
    )


def test_estimate_is_the_package_selection_on_the_mapped_inputs(planted_runs):
    """REML selection adds no numerics of its own: it reports the package's result."""
    runs, library = planted_runs
    runs = _with_white_noise(runs, 0.5, 12)
    rho = np.array([0.1, 0.4])
    expected = _package_result(runs, library, "amplitude", rho).reml
    selection, fit = expected.selection, expected.selection.fit
    result = api.select_library_reml(runs, library, "amplitude", rho)
    assert result.status == fit.status
    for kernel, winner, state in zip(
        result.kernels, selection.winner, fit.status, strict=True
    ):
        fallback = winner < 0 and state == "variance_zero"
        wanted = expected.reference_hrf_id if fallback else int(winner)
        assert (kernel.id if kernel is not None else -1) == wanted
    d = result.diagnostics
    np.testing.assert_array_equal(d["variance_ratio"], fit.ratio)
    np.testing.assert_array_equal(d["sigma2"], fit.noise_variance)
    np.testing.assert_array_equal(d["criterion"], fit.criterion)
    np.testing.assert_array_equal(d["at_upper"], fit.at_upper)
    np.testing.assert_array_equal(d["unrefined"], fit.unrefined)
    np.testing.assert_array_equal(d["any_at_upper"], selection.any_at_upper)
    np.testing.assert_array_equal(d["any_unrefined"], selection.any_unrefined)
    np.testing.assert_array_equal(d["provisional"], selection.any_unrefined)
    np.testing.assert_array_equal(d["candidate_gap"], selection.gap)
    np.testing.assert_array_equal(d["delta_criterion"], expected.delta_criterion)
    assert d["at_zero"] == [s == "variance_zero" for s in fit.status]
    assert d["rho"] == rho.tolist()
    assert d["reference_hrf_id"] == expected.reference_hrf_id
    assert d["library_fingerprint"] == library.fingerprint
    assert d["package_selection"]["method"] == "reml"
    source = Path(api.__file__).read_text()
    assert "select_random_effects_design" not in source and "RemlGroup" not in source


@pytest.mark.parametrize(
    "mode,regressors",
    [
        ("none", ["task", "trial_type[1]"]),
        ("amplitude", ["task", "trial_type[1]", "response_time"]),
    ],
)
def test_random_effects_are_valid_trial_deviations_in_every_rt_mode(
    planted_runs, mode, regressors
):
    """The RT mode changes the fixed task mean, not the trial-level random effects."""
    runs, library = planted_runs
    runs = _with_white_noise(runs[:2], 0.5, 3)
    result = api.fit_library_reml(runs, library, {"rt_mode": mode})
    d = result.diagnostics
    assert d["rt_mode"] == mode
    assert d["n_random_effects_per_run"] == [22, 22]
    assert "valid-trial deviations" in d["random_effects"]
    assert d["package_selection"]["task_regressors"] == regressors
    expected = _package_result(runs, library, mode, 0.0).reml.selection
    np.testing.assert_array_equal(d["criterion"], expected.fit.criterion)


def test_noise_estimate_keys_and_bounds(planted_runs):
    """Verify noise estimate keys and bounds."""
    runs, _ = planted_runs
    moderate = api.estimate_noise(_with_ar1_noise(runs, 0.3, 1.0, 5), "amplitude")
    assert set(moderate) == NOISE_KEYS
    assert moderate["training_run_ids"] == [r.run_id for r in runs]
    assert moderate["n_valid"] == 8
    assert not moderate["clipped"]
    assert moderate["rho"] == pytest.approx(moderate["corrected_median"])
    assert 0.1 < moderate["rho"] < 0.5
    strong = api.estimate_noise(_with_ar1_noise(runs, 0.97, 1.0, 6), "amplitude")
    # Estimates saturate at the end of the correction grid, the upper limit
    # of the default rho range: the pooled value is a bound there.
    assert strong["clipped"]
    assert strong["corrected_median"] == 0.95
    assert strong["rho"] == 0.95
    negative = api.estimate_noise(_with_ar1_noise(runs, -0.6, 1.0, 7), "amplitude")
    assert negative["clipped"]
    assert negative["corrected_median"] < -0.2
    assert negative["rho"] == -0.2


@pytest.mark.parametrize("method", ["remlcomponents", "remlshared", "remlcombined"])
def test_removed_variants_are_rejected(planted_runs, method):
    """Verify removed variants are rejected."""
    runs, library = planted_runs
    with pytest.raises(ValueError, match="unknown method"):
        pilot._estimate(runs, library, method, {})
