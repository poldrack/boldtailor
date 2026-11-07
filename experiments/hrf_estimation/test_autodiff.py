"""Autodiff must reproduce the current objective and its numerical derivative."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from experiments.hrf_estimation.continuous import _encode, _decode
from experiments.hrf_estimation.prepared import prepare_profile, prepared_loss, event_design
from experiments.hrf_estimation.test_library_cv import planted_runs
from experiments.hrf_estimation.test_continuous import offgrid
from boldtailor.hrf_library import HrfCandidate


@pytest.mark.parametrize("mode", ["amplitude", "duration"])
def test_autodiff_design_value_and_gradient_match_reference(planted_runs, mode):
    pytest.importorskip("jax")
    api = importlib.import_module("experiments.hrf_estimation.autodiff")
    runs, _ = planted_runs
    rng = np.random.default_rng(81)
    runs = tuple(replace(r, signals=r.signals + rng.normal(size=r.signals.shape)) for r in runs[:2])
    prepared = prepare_profile(runs, mode)
    parameters = (4.1, 12., .8, 1.5, 6., 0., 36.)
    unit = _encode(parameters)
    kernel = HrfCandidate(0,"double_gamma",parameters)
    for item in prepared:
        actual = api.design_value(unit, item)
        np.testing.assert_allclose(actual,event_design(item,kernel),rtol=1e-10,atol=1e-10)
    value, gradient = api.value_gradient(prepared,0,unit)
    assert value == pytest.approx(prepared_loss(prepared,0,parameters),abs=1e-11)
    numerical = approx_derivative(lambda u: prepared_loss(prepared,0,_decode(u)),unit,method="3-point")
    np.testing.assert_allclose(gradient,numerical.ravel(),rtol=1e-4,atol=1e-6)


def test_autodiff_handles_missing_junk_and_nuisance_overlap(planted_runs):
    pytest.importorskip("jax")
    import pandas as pd
    api = importlib.import_module("experiments.hrf_estimation.autodiff")
    runs, library = planted_runs
    from experiments.hrf_estimation.design import build_design
    changed=[]
    for run in runs[:2]:
        parts=build_design(run,library.candidates[1])
        changed.append(replace(run,confounds=pd.DataFrame(np.column_stack([parts.nuisance,parts.junk,parts.task]))))
    prepared=prepare_profile(tuple(changed),"amplitude")
    unit=_encode((4.1,12.,.8,1.5,6.,0.,36.))
    value,gradient=api.value_gradient(prepared,0,unit)
    assert value == pytest.approx(prepared_loss(prepared,0,_decode(unit)),abs=1e-10)
    assert np.isfinite(gradient).all()


def test_autodiff_compilation_is_reused_across_vertices(planted_runs):
    pytest.importorskip("jax")
    api=importlib.import_module("experiments.hrf_estimation.autodiff")
    runs,_=planted_runs
    prepared=prepare_profile(runs[:2],"amplitude")
    first=api.compiled_objective(prepared)
    assert api.compiled_objective(prepared) is first
    changed=tuple(replace(r,signals=r.signals*2) for r in runs[:2])
    assert api.compiled_objective(prepare_profile(changed,"amplitude")) is first


def test_autodiff_matches_when_event_design_loses_rank(planted_runs):
    pytest.importorskip("jax")
    import pandas as pd
    from experiments.hrf_estimation.design import build_design
    api=importlib.import_module("experiments.hrf_estimation.autodiff")
    runs,library=planted_runs
    rng=np.random.default_rng(18)
    changed=[]
    for r in runs[:2]:
        clean=replace(r,events=r.events.assign(is_correct=1,missing_data=0,
                            response_time=r.events.response_time.fillna(.8)))
        parts=build_design(clean,library.candidates[1])
        c=pd.DataFrame(np.column_stack([parts.nuisance,parts.task]))
        changed.append(replace(clean,confounds=c,signals=rng.normal(size=r.signals.shape)))
    prepared=prepare_profile(tuple(changed),"amplitude")
    unit=_encode(library.candidates[1].parameters)
    value,gradient=api.value_gradient(prepared,0,unit)
    assert value == pytest.approx(prepared_loss(prepared,0,_decode(unit)),abs=1e-10)
    assert value == pytest.approx(1.,abs=1e-10)
    assert np.isfinite(gradient).all()


def test_lbfgsb_uses_autodiff_and_reduces_function_evaluations(offgrid):
    pytest.importorskip("jax")
    from experiments.hrf_estimation.continuous import fit_continuous
    runs,library,truth=offgrid
    settings={"library":library,"lambda":0.,"max_starts":2,"maxiter":150,"maxfun":2500}
    reference=fit_continuous(runs,settings)
    fitted=fit_continuous(runs,{**settings,"gradient":"autodiff","n_jobs":2})
    assert fitted.diagnostics["gradient"] == "autodiff"
    assert fitted.diagnostics["autodiff_environment"]["jax"]
    assert fitted.diagnostics["objective"][0] < 1e-7
    assert fitted.diagnostics["function_evaluations"][0] < reference.diagnostics["function_evaluations"][0]
    np.testing.assert_allclose(fitted.kernels[0].kernel(.1,1),reference.kernels[0].kernel(.1,1),atol=2e-3)


def test_autodiff_rejects_incompatible_settings(planted_runs):
    from experiments.hrf_estimation.continuous import fit_continuous
    runs,library=planted_runs
    with pytest.raises(ValueError,match="prepared"):
        fit_continuous(runs,{"library":library,"lambda":0.,"gradient":"autodiff","prepared":False})
