"""RT omission must preserve trial inclusion, nuisance, and the original REML likelihood."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
from experiments.hrf_estimation.test_design import event_run
from experiments.hrf_estimation.test_library_cv import planted_runs
from boldtailor.hrf_library import HrfCandidate, CANONICAL_PARAMETERS
from experiments.hrf_estimation.design import build_design
from tests.reml_oracles import dense_fit


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
    from experiments.hrf_estimation.reml_selection import fit_library_reml
    runs,library=planted_runs
    rng=np.random.default_rng(37)
    runs=tuple(replace(r,signals=r.signals+rng.normal(0,.2,r.signals.shape)) for r in runs[:2])
    result=fit_library_reml(runs,library,{"rt_mode":"none"})
    assert result.diagnostics["rt_mode"] == "none"
    for feature,kernel in enumerate(result.kernels):
        signals,fixed,random=[],[],[]
        for run in runs:
            full=build_design(run,kernel,"amplitude")
            signals.append(run.signals[:,[feature]])
            fixed.append(full.nuisance)
            random.append(np.column_stack([full.task[:,:2],full.junk]))
        ratio,sigma2=dense_fit(signals,fixed,random)
        np.testing.assert_allclose(result.diagnostics["variance_ratio"][feature],ratio[0],rtol=1e-5)
        np.testing.assert_allclose(result.diagnostics["sigma2"][feature],sigma2[0],rtol=1e-5)


def test_crossed_evaluation_separates_selection_and_evaluation_rt(planted_runs):
    from experiments.hrf_estimation.reml_selection import fit_library_reml
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
