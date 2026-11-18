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
    """Verify no response-time omits only the response-time regressor."""
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


def test_no_rt_reml_drops_rt_from_the_fixed_task_mean(planted_runs):
    """Without RT the package task model has no RT regressor, so the criterion changes."""
    from experiments.hrf_estimation.reml_selection import fit_library_reml
    runs,library=planted_runs
    rng=np.random.default_rng(37)
    runs=tuple(replace(r,signals=r.signals+rng.normal(0,.2,r.signals.shape)) for r in runs[:2])
    fits={mode:fit_library_reml(runs,library,{"rt_mode":mode}) for mode in ["none","amplitude"]}
    names={mode:fit.diagnostics["package_selection"]["task_regressors"] for mode,fit in fits.items()}
    assert fits["none"].diagnostics["rt_mode"] == "none"
    assert names == {"none":["task","trial_type[1]"],"amplitude":["task","trial_type[1]","response_time"]}
    assert not np.allclose(fits["none"].diagnostics["criterion"],fits["amplitude"].diagnostics["criterion"])


def test_crossed_evaluation_separates_selection_and_evaluation_rt(planted_runs):
    """Verify crossed evaluation separates selection and evaluation response-time."""
    from experiments.hrf_estimation.reml_selection import fit_library_reml
    api=importlib.import_module("experiments.hrf_estimation.rt_ablation")
    runs,library=planted_runs
    # Noiseless canonical data leave the package REML no residual to identify from.
    rng=np.random.default_rng(41)
    runs=tuple(replace(r,signals=r.signals+rng.normal(0,.2,r.signals.shape)) for r in runs)
    fits={mode:fit_library_reml(runs[:2],library,{"rt_mode":mode}) for mode in ["amplitude","none"]}
    table=api.crossed_scores(runs[2:],fits,"transfer01")
    assert set(zip(table.selection_rt,table.evaluation_rt)) == {(a,b) for a in fits for b in fits}
    assert len(table)==8
    for mode in fits:
        data=table[table.selection_rt.eq(mode)].set_index(["evaluation_rt","feature_id"])
        assert (data.loc["amplitude"].sse_full <= data.loc["none"].sse_full+1e-8).all()
        np.testing.assert_allclose(data.loc["amplitude"].sse_nuisance,data.loc["none"].sse_nuisance)
