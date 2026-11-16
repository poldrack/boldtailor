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


def test_unit_rms_scaling_leaves_selection_and_fit_variance_consistent(
    planted, library
):
    """Verify unit rms scaling leaves selection and fit variance consistent."""
    from experiments.hrf_simulations.fitting import fit

    args = (planted, library, "ConsDurRTMod", "stimulus", "iid")
    native = fit(*args, settings={"regressor_scaling": "native"})
    scaled = fit(*args, settings={"regressor_scaling": "unit_rms"})
    assert native.kernels[0] == library.candidates[0]
    assert scaled.kernels == native.kernels
    assert scaled.status == ("ok",)
    assert np.all(np.isfinite(scaled.diagnostics["variance_ratio"]))
    assert scaled.diagnostics["random_scales"][0] != [1.0] * len(
        scaled.diagnostics["random_scales"][0]
    )


def _scaled_candidates(runs, library, model, convention):
    """Each candidate's task+junk columns divided by its unit_rms scales."""
    from experiments.hrf_simulations.design import task_design
    from experiments.hrf_simulations.fitting import random_scales

    out = []
    for candidate in library.candidates:
        parts = [task_design(r, candidate, model, convention) for r in runs]
        scales = random_scales(parts, "unit_rms")
        designs = [np.column_stack([p.task, p.junk]) / scales for p in parts]
        out.append((parts, scales, designs))
    return out


def test_unit_rms_scaling_divides_random_design_by_pooled_rms(planted, library):
    """Verify unit rms scaling divides random design by pooled rms."""
    from boldtailor.reml import RemlGroup, select_random_effects_design
    from experiments.hrf_estimation.scoring import residuals
    from experiments.hrf_simulations.fitting import fit

    model, convention = "ConsDurRTMod", "stimulus"
    scaled = _scaled_candidates(planted, library, model, convention)
    for parts, _, designs in scaled:
        projected = [residuals(p.nuisance, z)[0] for p, z in zip(parts, designs)]
        pooled = sum(np.sum(z * z, axis=0) for z in projected)
        rms = np.sqrt(pooled / sum(len(z) for z in projected))
        np.testing.assert_allclose(rms, 1.0, rtol=1e-12)
    groups = [RemlGroup(r.signals, r.confounds.to_numpy(float)) for r in planted]
    expected = select_random_effects_design(groups, [d for _, _, d in scaled])
    winner = int(expected.winner[0])
    result = fit(
        planted,
        library,
        model,
        convention,
        "iid",
        settings={"regressor_scaling": "unit_rms"},
    )
    assert result.kernels[0] == library.candidates[winner]
    assert result.diagnostics["variance_ratio"] == [float(expected.fit.ratio[0])]
    assert result.diagnostics["sigma2"] == [float(expected.fit.noise_variance[0])]
    assert result.diagnostics["criterion"] == [float(expected.fit.criterion[0])]
    assert result.diagnostics["unrefined"] == expected.fit.unrefined.tolist()
    assert result.diagnostics["any_at_upper"] == expected.any_at_upper.tolist()
    assert result.diagnostics["any_unrefined"] == expected.any_unrefined.tolist()
    assert result.diagnostics["provisional"] == expected.any_unrefined.tolist()
    assert result.diagnostics["random_scales"] == [scaled[winner][1].tolist()]


@pytest.mark.parametrize("noise_mode", ["iid", "ar1"])
def test_matched_noiseless_recovery(runs, library, noise_mode):
    """Recover the generating HRF from noiseless data under the matched fitting model."""
    from experiments.hrf_simulations.fitting import fit
    from experiments.hrf_simulations.simulation import signal_components

    data = [
        replace(
            r,
            signals=sum(
                signal_components(r, library.candidates[0], "duration", "brief")
            )[:, None],
        )
        for r in runs
    ]
    result = fit(
        data,
        library,
        "RTDur",
        "brief",
        noise_mode,
        oracle_rho=0.0 if noise_mode == "ar1" else None,
    )
    assert result.status == ("ok",)
    assert result.kernels[0] == library.candidates[0]


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
    b = fit(data, library, "ConsDurRTDur", "stimulus", "ar1")
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
    fit(data, library, "ConsDurRTDur", "stimulus", "ar1", cache=cache)
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
