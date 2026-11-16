from dataclasses import replace

import numpy as np
import pytest

from boldtailor.hrf_library import HrfCandidate, HrfLibrary
from experiments.hrf_estimation.sensitivity import synthetic_templates


@pytest.fixture
def runs():
    return synthetic_templates()[:2]


@pytest.fixture
def library():
    return HrfLibrary(
        (
            HrfCandidate(0, "spm", (6, 16, 1, 1, 6, 0, 32)),
            HrfCandidate(1, "double_gamma", (3.5, 12, 0.8, 1.5, 6, 0, 36)),
        )
    )


@pytest.fixture
def planted(runs, library):
    """Noisy single-vertex runs generated from the first candidate's design."""
    from experiments.hrf_simulations.design import task_design

    rng = np.random.default_rng(5)
    data = []
    for r in runs:
        p = task_design(r, library.candidates[0], "ConsDurRTMod", "stimulus")
        signal = p.task @ [1.0, 2.0, 0.5] + rng.normal(0, 0.2, len(r.frame_times))
        data.append(replace(r, signals=signal[:, None]))
    return data


def test_iid_and_ar1_paths_agree_at_zero_rho(planted, library):
    """Verify independent-noise and AR(1) paths agree at zero rho."""
    from experiments.hrf_simulations.fitting import fit

    iid = fit(planted, library, "ConsDurRTMod", "stimulus", "iid")
    ar1 = fit(planted, library, "ConsDurRTMod", "stimulus", "ar1", oracle_rho=0.0)
    assert iid.kernels == ar1.kernels
    assert iid.status == ar1.status
    for key in ("criterion", "variance_ratio", "sigma2"):
        np.testing.assert_allclose(
            iid.diagnostics[key], ar1.diagnostics[key], rtol=1e-10
        )


@pytest.mark.parametrize("method", ["cv", "reml"])
def test_legacy_random_scaling_is_rejected(planted, library, method):
    """Legacy column scaling cannot silently change the package's trial covariance."""
    from experiments.hrf_simulations.fitting import fit

    with pytest.raises(ValueError, match="regressor_scaling"):
        fit(
            planted,
            library,
            "ConsDurNoRT",
            "brief",
            "iid",
            method=method,
            settings={"regressor_scaling": "unit_rms"},
        )


@pytest.mark.parametrize("method", ["cv", "reml"])
def test_unsupported_double_duration_model_is_explicit(planted, library, method):
    """An unsupported two-duration mean must not be silently approximated."""
    from experiments.hrf_simulations.fitting import fit

    with pytest.raises(ValueError, match="ConsDurRTDur"):
        fit(planted, library, "ConsDurRTDur", "brief", "iid", method=method)


def package_inputs(runs, library, model, convention):
    """Independently map valid trials and canonical junk into package inputs."""
    from boldtailor.data import from_arrays
    from boldtailor.model import Modulator, TaskModel
    from experiments.hrf_simulations.design import task_design

    events, confounds = [], []
    for run in runs:
        e = run.events
        valid = (
            (e.missing_data == 0)
            & (e.is_correct == 1)
            & np.isfinite(e.response_time)
            & (e.response_time > 0)
        )
        selected = e.loc[valid].copy()
        selected["duration"] = (
            selected.response_time
            if model == "RTDur"
            else 0.1 if convention == "brief" else selected.duration
        )
        events.append(selected)
        nuisance = run.confounds.drop(columns="intercept").copy()
        junk = task_design(run, library.candidates[0], "ConsDurNoRT", "stimulus").junk
        nuisance["selection_missing"] = junk[:, 0]
        nuisance["selection_incorrect"] = junk[:, 1]
        confounds.append(nuisance)
    mods = [Modulator("trial_type", kind="categorical", levels=(0, 1))]
    if model == "ConsDurRTMod":
        mods.append(Modulator("response_time"))
    return from_arrays(
        [r.signals for r in runs],
        events,
        frame_times=[r.frame_times for r in runs],
        confounds=confounds,
    ), TaskModel(tuple(mods))


@pytest.mark.parametrize("method", ["cv", "reml"])
@pytest.mark.parametrize("model", ["ConsDurNoRT", "RTDur", "ConsDurRTMod"])
@pytest.mark.parametrize("convention", ["brief", "stimulus"])
def test_identification_matches_public_package(
    planted, library, method, model, convention
):
    """Simulation identification must match the public selector on the declared inputs."""
    from boldtailor.hrf_selection import select_hrfs
    from boldtailor.hrf_options import RemlHrfOptions
    from experiments.hrf_simulations.fitting import fit

    data, task_model = package_inputs(planted, library, model, convention)
    options = dict(
        method=method,
        library=library,
        task_model=task_model,
        run_labels=[r.run_id for r in planted],
    )
    if method == "reml":
        options["reml_options"] = RemlHrfOptions(rho=0.25)
    expected = select_hrfs(data, **options)
    actual = fit(
        planted,
        library,
        model,
        convention,
        "ar1" if method == "reml" else "iid",
        oracle_rho=0.25 if method == "reml" else None,
        method=method,
    )
    winner = int(expected.hrf_indices[0])
    assert actual.kernels == ((library.candidates[winner] if winner >= 0 else None),)
    assert actual.diagnostics["selection_method"] == method
    assert actual.diagnostics["training_run_ids"] == [r.run_id for r in planted]
    assert actual.diagnostics["task_model"] == task_model.to_dict()
    if method == "cv":
        np.testing.assert_allclose(actual.diagnostics["cv_r2"], expected.cv_r2)
        assert "criterion" not in actual.diagnostics
        assert actual.diagnostics["provisional"] == [None]
    else:
        d = expected.reml
        np.testing.assert_allclose(
            actual.diagnostics["criterion"], d.selection.fit.criterion
        )
        np.testing.assert_allclose(
            actual.diagnostics["variance_ratio"], d.selection.fit.ratio
        )
        np.testing.assert_allclose(
            actual.diagnostics["delta_criterion"], d.delta_criterion
        )
        assert actual.diagnostics["any_unrefined"] == d.selection.any_unrefined.tolist()
        assert actual.status == d.selection.fit.status
        assert "cv_r2" not in actual.diagnostics


def test_pure_reference_mean_has_no_reml_winner(planted, library):
    """A mean-only signal must remain unsupported rather than force canonical recovery."""
    from boldtailor._hrf_design import trial_regressors
    from experiments.hrf_simulations.fitting import fit

    data, _ = package_inputs(planted, library, "ConsDurNoRT", "brief")
    runs = [
        replace(
            r,
            signals=(100 + trial_regressors(e, t, library.candidates[0]).sum(axis=1))[
                :, None
            ],
        )
        for r, e, t in zip(planted, data.events, data.frame_times, strict=True)
    ]
    result = fit(runs, library, "ConsDurNoRT", "brief", "iid")
    assert result.kernels == (None,)
    assert result.status == ("no_residual_signal",)


def test_cv_rejects_ar1_options(planted, library):
    """CV has no whitening option and must not claim to use a supplied rho."""
    from experiments.hrf_simulations.fitting import fit

    with pytest.raises(ValueError, match="CV|cv"):
        fit(planted, library, "ConsDurNoRT", "brief", "ar1", method="cv")


def test_rho_reference_is_model_independent_and_diagnostic(runs, library):
    """Verify rho reference is model independent and diagnostic."""
    from experiments.hrf_simulations.fitting import estimate_rho, fit
    from experiments.hrf_simulations.noise import ar1

    rng = np.random.default_rng(3)
    data = [
        replace(r, signals=ar1(len(r.frame_times), 0.4, 1.0, rng)[:, None])
        for r in runs
    ]
    rho = estimate_rho(data)
    assert abs(rho - 0.4) < 0.25
    a = fit(data, library, "ConsDurNoRT", "brief", "ar1")
    b = fit(data, library, "RTDur", "stimulus", "ar1")
    assert a.diagnostics["rho"] == b.diagnostics["rho"] == rho
    assert a.diagnostics["training_run_ids"] == [r.run_id for r in runs]


def test_prepared_rho_reference_is_cached_once_per_run(runs, library, monkeypatch):
    """Verify prepared rho reference is cached once per run."""
    from boldtailor import ar1 as package_ar1
    from experiments.hrf_simulations.design import reference_design
    from experiments.hrf_simulations.fitting import DesignCache, estimate_rho, fit
    from experiments.hrf_simulations.noise import ar1

    rng = np.random.default_rng(3)
    data = [
        replace(r, signals=ar1(len(r.frame_times), 0.4, 1.0, rng)[:, None])
        for r in runs
    ]
    designs = [reference_design(r) for r in data]
    expected = package_ar1.estimate_rho([r.signals for r in data], designs)
    calls = []
    original = package_ar1.prepare_reference

    def counting(design):
        calls.append(1)
        return original(design)

    monkeypatch.setattr(package_ar1, "prepare_reference", counting)
    cache = DesignCache()
    fit(data, library, "ConsDurNoRT", "brief", "ar1", cache=cache)
    fit(data, library, "RTDur", "stimulus", "ar1", cache=cache)
    assert len(calls) == len(data)
    assert estimate_rho(data, cache) == float(expected.pooled)
    assert len(calls) == len(data)


def test_curve_metrics_and_failure_accounting(library):
    """Verify curve metrics and failure accounting."""
    from experiments.hrf_simulations.metrics import recovery, library_floor

    truth = library.candidates[0]
    result = recovery(truth, truth)
    assert result["curve_rmse"] == 0
    assert result["curve_correlation"] == pytest.approx(1.0)
    assert recovery(None, truth)["failed"]
    assert library_floor(truth, library)["min_curve_rmse"] == 0


def test_default_cache_holds_all_pilot_design_groups():
    """Verify default cache holds all pilot design groups."""
    from experiments.hrf_simulations.fitting import DesignCache

    assert DesignCache().max_entries >= 533 * 12 * 4 * 2


@pytest.mark.parametrize("method", ["cv", "reml"])
def test_reused_cache_does_not_reuse_training_signals(planted, library, method):
    """A design cache can be shared across draws without retaining old BOLD data."""
    from experiments.hrf_simulations.fitting import DesignCache, fit

    cache = DesignCache()
    args = (library, "ConsDurRTMod", "brief", "iid")
    base = fit(planted, *args, method=method, cache=cache)
    changed = [replace(r, signals=3 * r.signals) for r in planted]
    cached = fit(changed, *args, method=method, cache=cache)
    fresh = fit(changed, *args, method=method)
    assert cached.kernels == fresh.kernels
    score = "criterion" if method == "reml" else "cv_r2"
    np.testing.assert_allclose(cached.diagnostics[score], fresh.diagnostics[score])
    if method == "reml":
        assert not np.allclose(base.diagnostics[score], cached.diagnostics[score])


def test_cached_inputs_track_confound_column_names(planted, library):
    """Renaming nuisance columns must not return a previous run-input schema."""
    from experiments.hrf_simulations.fitting import DesignCache
    from experiments.hrf_simulations.selection_inputs import selection_data

    cache = DesignCache()
    selection_data(planted, library, "RTDur", "brief", cache)
    changed = [
        replace(r, confounds=r.confounds.rename(columns={"intercept": "baseline"}))
        for r in planted
    ]
    prepared, _ = selection_data(changed, library, "RTDur", "brief", cache)
    assert "baseline" in prepared.confounds[0]
    np.testing.assert_array_equal(prepared.confounds[0]["baseline"], 1.0)


@pytest.mark.parametrize("method", ["cv", "reml"])
def test_rtdur_rejects_unknown_duration_convention(planted, library, method):
    """An invalid convention must not be silently accepted for an RT-duration model."""
    from experiments.hrf_simulations.fitting import fit

    with pytest.raises(ValueError, match="convention"):
        fit(planted, library, "RTDur", "unknown", "iid", method=method)
