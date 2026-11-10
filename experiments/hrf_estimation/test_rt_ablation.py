"""RT omission must preserve trial inclusion, nuisance, and the original REML likelihood."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
from scipy.linalg import null_space
from experiments.hrf_estimation.test_design import event_run
from experiments.hrf_estimation.test_library_cv import planted_runs
from boldtailor.hrf_library import HrfCandidate, CANONICAL_PARAMETERS
from experiments.hrf_estimation.design import build_design


def test_no_rt_omits_only_the_rt_regressor(event_run):
    kernel=HrfCandidate(0,"spm",CANONICAL_PARAMETERS)
    full=build_design(event_run,kernel,"amplitude")
    reduced=build_design(event_run,kernel,"none")
    assert reduced.task_names == ("trial_type_0","trial_type_1")
    assert reduced.task.shape[1] == 2
    np.testing.assert_array_equal(reduced.task,full.task[:,:2])
    np.testing.assert_array_equal(reduced.junk,full.junk)
    np.testing.assert_array_equal(reduced.nuisance,full.nuisance)
    changed=event_run.events.copy()
    changed.loc[changed.response_time.gt(0),"response_time"] *= 4
    other=build_design(replace(event_run,events=changed),kernel,"none")
    np.testing.assert_array_equal(other.task,reduced.task)
    np.testing.assert_array_equal(other.junk,reduced.junk)


def test_no_rt_reml_matches_dense_four_effect_covariance(planted_runs):
    from experiments.hrf_estimation.reml import prepare_terms,criterion_grid,fit_library_reml
    runs,library=planted_runs
    rng=np.random.default_rng(37)
    runs=tuple(replace(r,signals=r.signals+rng.normal(0,.2,r.signals.shape)) for r in runs[:2])
    candidate=library.candidates[1]
    terms=prepare_terms(runs,candidate,"none")
    q=np.zeros(2); determinant=0.; n=0; ratio=.7
    for run in runs:
        full=build_design(run,candidate,"amplitude")
        basis=null_space(full.nuisance.T)
        z=basis.T@np.column_stack([full.task[:,:2],full.junk])
        y=basis.T@run.signals
        covariance=np.eye(len(y))+ratio*z@z.T
        q += np.sum(y*np.linalg.solve(covariance,y),axis=0)
        determinant += np.linalg.slogdet(covariance)[1];n += len(y)
    np.testing.assert_allclose(criterion_grid([ratio],terms)[0],n*np.log(q/n)+determinant,rtol=1e-9)
    result=fit_library_reml(runs,library,{"rt_mode":"none"})
    assert result.diagnostics["rt_mode"] == "none"
    assert result.diagnostics["n_random_effects_per_run"] == 4


def test_crossed_evaluation_separates_selection_and_evaluation_rt(planted_runs):
    from experiments.hrf_estimation.reml import fit_library_reml
    api=importlib.import_module("experiments.hrf_estimation.rt_ablation")
    runs,library=planted_runs
    fits={mode:fit_library_reml(runs[:2],library,{"rt_mode":mode}) for mode in ["amplitude","none"]}
    table=api.crossed_scores(runs[2:],fits,"transfer01")
    assert set(zip(table.selection_rt,table.evaluation_rt)) == {(a,b) for a in fits for b in fits}
    assert len(table)==8
    for mode in fits:
        data=table[table.selection_rt.eq(mode)].set_index(["evaluation_rt","feature_id"])
        assert (data.loc["amplitude"].sse_full <= data.loc["none"].sse_full+1e-8).all()
        np.testing.assert_allclose(data.loc["amplitude"].sse_nuisance,data.loc["none"].sse_nuisance)
