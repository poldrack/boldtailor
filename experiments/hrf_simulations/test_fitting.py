from dataclasses import replace

import numpy as np
import pytest

from boldtailor.hrf_library import HrfCandidate, HrfLibrary
from experiments.hrf_estimation.reml import fit_library_reml
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.sensitivity import synthetic_templates


@pytest.fixture
def runs():
    return synthetic_templates()[:2]


@pytest.fixture
def library():
    return HrfLibrary((HrfCandidate(0, 'double_gamma', (6, 16, 1, 1, 6, 0, 36)),
                       HrfCandidate(1, 'double_gamma', (3.5, 12, .8, 1.5, 6, 0, 36))))


def test_iid_adapter_matches_original_solver(runs, library):
    from experiments.hrf_simulations.fitting import fit
    rng = np.random.default_rng(5)
    data = []
    for r in runs:
        p = build_design(r, library.candidates[0], 'amplitude')
        data.append(replace(r, signals=(p.task @ [1., 2., .5]+rng.normal(0,.2,len(r.frame_times)))[:,None]))
    original = fit_library_reml(data, library, {})
    result = fit(data, library, 'ConsDurRTMod', 'stimulus', 'iid',
                 design_builder=lambda r,k,m,c: build_design(r,k,'amplitude'))
    assert result.kernels == original.kernels
    assert result.status == original.status
    np.testing.assert_allclose(result.diagnostics['criterion'], original.diagnostics['criterion'], atol=1e-8)


@pytest.mark.parametrize('noise_mode', ['iid', 'ar1'])
def test_matched_noiseless_recovery(runs, library, noise_mode):
    from experiments.hrf_simulations.fitting import fit
    from experiments.hrf_simulations.simulation import signal_components
    data = [replace(r, signals=sum(signal_components(r,library.candidates[0], 'duration','brief'))[:,None]) for r in runs]
    result = fit(data, library, 'RTDur', 'brief', noise_mode, oracle_rho=0. if noise_mode=='ar1' else None)
    assert result.status == ('ok',)
    assert result.kernels[0] == library.candidates[0]


def test_rho_reference_is_model_independent_and_diagnostic(runs, library):
    from experiments.hrf_simulations.fitting import estimate_rho, fit
    from experiments.hrf_simulations.noise import ar1
    rng = np.random.default_rng(3)
    data = [replace(r,signals=ar1(len(r.frame_times),.4,1.,rng)[:,None]) for r in runs]
    rho = estimate_rho(data)
    assert abs(rho-.4) < .25
    a = fit(data,library,'ConsDurNoRT','brief','ar1')
    b = fit(data,library,'ConsDurRTDur','stimulus','ar1')
    assert a.diagnostics['rho'] == b.diagnostics['rho'] == rho
    assert a.diagnostics['training_run_ids'] == [r.run_id for r in runs]


def test_curve_metrics_and_failure_accounting(library):
    from experiments.hrf_simulations.metrics import recovery, library_floor
    truth = library.candidates[0]
    result = recovery(truth,truth)
    assert result['curve_rmse'] == 0
    assert result['curve_correlation'] == pytest.approx(1.)
    assert recovery(None,truth)['failed']
    assert library_floor(truth,library)['min_curve_rmse'] == 0
