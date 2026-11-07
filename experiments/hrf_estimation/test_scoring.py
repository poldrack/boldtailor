import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfCandidate, CANONICAL_PARAMETERS
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.inputs import RunData


@pytest.fixture
def signal_run():
    rng = np.random.default_rng(4)
    times = np.arange(180) * 4/3
    events = pd.DataFrame(dict(onset=np.arange(8, 200, 8.), duration=3.,
        trial_type=np.tile([0, 1], 12), response_time=rng.uniform(.5, 1.5, 24),
        is_correct=1, missing_data=0))
    return RunData('sub-07_ses-nsd11_task-nsdcore_run-01', np.zeros((180, 2)),
        events, times, pd.DataFrame({'intercept': np.ones(180)}), np.array([8, 3]))


def estimate():
    cls = importlib.import_module('experiments.hrf_estimation.estimators').HrfEstimate
    kernel = HrfCandidate(0, 'spm', CANONICAL_PARAMETERS)
    return cls(np.array([8, 3]), (kernel, kernel), ('ok', 'ok'), ('training-run',), {})


def test_evaluation_refits_amplitudes_and_pools_sums(signal_run):
    est = estimate()
    parts = build_design(signal_run, est.kernels[0])
    rng = np.random.default_rng(1)
    y1 = parts.task @ np.array([[2, 3], [5, 7], [1, -1]]) + 30
    y2 = parts.task @ np.array([[9, 4], [-2, 12], [5, 2]]) + 70
    y2[:, 1] += rng.normal(0, .3, len(y2))
    runs = (replace(signal_run, signals=y1), replace(signal_run,
        run_id=signal_run.run_id.replace('run-01','run-02'), signals=y2))
    score = importlib.import_module('experiments.hrf_estimation.scoring').score_hrfs(runs, est)
    assert score.summary.loc[8, 'r2_full'] == pytest.approx(1)
    design = np.column_stack([parts.nuisance, parts.junk, parts.task])
    sse = sum(np.sum((y - design @ np.linalg.lstsq(design, y, rcond=None)[0])**2, axis=0) for y in (y1,y2))
    sst = sum(np.sum((y-y.mean(axis=0))**2,axis=0) for y in (y1,y2))
    np.testing.assert_allclose(score.summary.r2_full, 1-sse/sst)
    np.testing.assert_allclose(score.summary.r2_task_partial, 1-sse/sst)
    assert set(score.per_run.run_id) == {r.run_id for r in runs}


def test_constant_signals_are_invalid_not_perfect(signal_run):
    score = importlib.import_module('experiments.hrf_estimation.scoring').score_hrfs(
        (replace(signal_run, signals=np.ones((180,2))),), estimate())
    assert score.summary.r2_full.isna().all()
    assert score.summary.r2_task_partial.isna().all()
    assert (score.summary.status == 'zero_variance').all()


def test_feature_order_mismatch_fails(signal_run):
    score = importlib.import_module('experiments.hrf_estimation.scoring').score_hrfs
    with pytest.raises(ValueError, match='feature'):
        score((replace(signal_run, feature_ids=np.array([3,8])),), estimate())


def test_partial_and_total_r2_separate_nuisance_from_task(signal_run):
    rng = np.random.default_rng(9)
    drift = np.linspace(-1, 1, len(signal_run.frame_times))
    nuisance = signal_run.confounds.assign(drift=drift)
    events = signal_run.events.copy()
    events.loc[2, 'is_correct'] = 0
    run = replace(signal_run, confounds=nuisance, events=events)
    parts = build_design(run, estimate().kernels[0])
    y = 20*drift + parts.task @ [1, 3, 2] + parts.junk[:,1]*4 + rng.normal(size=len(drift))
    run = replace(run, signals=np.column_stack([y,y]))
    result = importlib.import_module('experiments.hrf_estimation.scoring').score_hrfs((run,), estimate())
    null = np.column_stack([parts.nuisance, parts.junk])
    full = np.column_stack([null, parts.task])
    sse = lambda x: np.sum((y-x@np.linalg.lstsq(x,y,rcond=None)[0])**2)
    full_sse, null_sse, sst = sse(full), sse(null), np.sum((y-y.mean())**2)
    assert result.summary.loc[8,'r2_task_partial'] == pytest.approx(1-full_sse/null_sse)
    assert result.summary.loc[8,'r2_full'] == pytest.approx(1-full_sse/sst)
    assert result.summary.loc[8,'delta_r2_task'] == pytest.approx((null_sse-full_sse)/sst)


def test_zero_task_denominator_has_explicit_reason(signal_run):
    drift=np.linspace(-1,1,len(signal_run.frame_times))
    run=replace(signal_run,signals=np.column_stack([drift,drift]),
        confounds=signal_run.confounds.assign(drift=drift))
    score=importlib.import_module('experiments.hrf_estimation.scoring').score_hrfs((run,),estimate())
    assert score.summary.r2_task_partial.isna().all()
    assert (score.summary.status=='zero_task_variance').all()


def test_evaluation_checks_anatomy_against_training_estimate(signal_run):
    import nibabel as nib
    a=nib.cifti2.BrainModelAxis.from_surface([8,3],10,'CortexLeft')
    b=nib.cifti2.BrainModelAxis.from_surface([3,8],10,'CortexLeft')
    cls=importlib.import_module('experiments.hrf_estimation.estimators').HrfEstimate
    base=estimate()
    fitted=cls(base.feature_ids,base.kernels,base.status,base.training_run_ids,{},brain_axis=a)
    score=importlib.import_module('experiments.hrf_estimation.scoring').score_hrfs
    with pytest.raises(ValueError,match='anatom'):
        score((replace(signal_run,brain_axis=b),),fitted)
