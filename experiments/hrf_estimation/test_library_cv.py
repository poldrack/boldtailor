"""HRF selection must recover known shapes using training runs only."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.inputs import RunData


@pytest.fixture
def planted_runs():
    library = HrfLibrary.from_parameters([(3.5, 12, .8, 1.5, 6, 0, 36)])
    rng = np.random.default_rng(7)
    runs = []
    for i in range(4):
        events = pd.DataFrame(dict(onset=np.arange(8, 200, 8.), duration=3.,
            trial_type=np.tile([0,1],12), response_time=rng.uniform(.4,1.8,24),
            is_correct=1, missing_data=0))
        events.loc[3,'is_correct']=0
        events.loc[5,'response_time']=np.nan
        times=np.arange(180)*4/3
        run=RunData(f'sub-07_ses-nsd11_task-nsdcore_run-{i+1:02}', np.zeros((180,2)),
            events, times, pd.DataFrame({'intercept':np.ones(180)}), np.array([4,7]))
        signals=[]
        for kernel in library.candidates:
            parts=build_design(run,kernel)
            signals.append(parts.task@np.array([2,5,1.5])+parts.junk@np.array([i+1,2*i+1])+20)
        runs.append(replace(run, signals=np.column_stack(signals)))
    return tuple(runs), library


def test_cv_recovers_distinct_hrfs_with_profiled_junk(planted_runs):
    runs,library=planted_runs
    fit=importlib.import_module('experiments.hrf_estimation.library_cv').fit_library_cv
    estimate=fit(runs[:3],library,{})
    assert [k.id for k in estimate.kernels] == [0,1]
    assert estimate.status == ('ok','ok')
    assert estimate.training_run_ids == tuple(r.run_id for r in runs[:3])
    assert np.allclose(estimate.diagnostics['selected_cv_r2'],1)


def test_outer_test_signals_cannot_change_selection(planted_runs):
    runs,library=planted_runs
    fit=importlib.import_module('experiments.hrf_estimation.library_cv').fit_library_cv
    first=fit(runs[:3],library,{})
    runs[-1].signals[:] = 1e20
    second=fit(runs[:3],library,{})
    assert first.kernels == second.kernels
    np.testing.assert_array_equal(first.diagnostics['selected_cv_r2'], second.diagnostics['selected_cv_r2'])


def test_zero_signal_gets_no_hrf(planted_runs):
    runs,library=planted_runs
    fit=importlib.import_module('experiments.hrf_estimation.library_cv').fit_library_cv
    estimate=fit(tuple(replace(r,signals=np.ones_like(r.signals)) for r in runs),library,{})
    assert estimate.kernels == (None,None)
    assert estimate.status == ('no_signal','no_signal')


def test_cv_agrees_with_package_for_compatible_task_model(planted_runs):
    from boldtailor.data import from_arrays
    from boldtailor.model import TaskModel, Modulator
    from boldtailor.hrf_selection import select_hrfs
    runs,library=planted_runs
    clean=tuple(replace(r,events=r.events.assign(is_correct=1,missing_data=0,
        response_time=r.events.response_time.fillna(.9))) for r in runs)
    fit=importlib.import_module('experiments.hrf_estimation.library_cv').fit_library_cv
    ours=fit(clean,library,{})
    data=from_arrays(signals=[r.signals for r in clean],events=[r.events for r in clean],
        confounds=[r.confounds for r in clean],tr=4/3)
    package=select_hrfs(data,library=library,
        task_model=TaskModel((Modulator('response_time'),Modulator('trial_type'))))
    assert [k.id for k in ours.kernels] == package.hrf_indices.tolist()
    np.testing.assert_allclose(ours.diagnostics['selected_cv_r2'],package.cv_r2,atol=1e-8)
