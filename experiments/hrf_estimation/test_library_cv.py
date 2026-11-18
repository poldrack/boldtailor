"""HRF selection must recover known shapes using training runs only."""

import importlib
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.inputs import RunData


@pytest.fixture
def planted_runs():
    library = HrfLibrary.from_parameters([(3.5, 12, 0.8, 1.5, 6, 0, 36)])
    rng = np.random.default_rng(7)
    runs = []
    for i in range(4):
        events = pd.DataFrame(
            dict(
                onset=np.arange(8, 200, 8.0),
                duration=3.0,
                trial_type=np.tile([0, 1], 12),
                response_time=rng.uniform(0.4, 1.8, 24),
                is_correct=1,
                missing_data=0,
            )
        )
        events.loc[3, "is_correct"] = 0
        events.loc[5, "response_time"] = np.nan
        times = np.arange(180) * 4 / 3
        run = RunData(
            f"sub-07_ses-nsd11_task-nsdcore_run-{i+1:02}",
            np.zeros((180, 2)),
            events,
            times,
            pd.DataFrame({"intercept": np.ones(180)}),
            np.array([4, 7]),
        )
        signals = []
        for kernel in library.candidates:
            parts = build_design(run, kernel)
            signals.append(
                parts.task @ np.array([2, 5, 1.5])
                + parts.junk @ np.array([i + 1, 2 * i + 1])
                + 20
            )
        runs.append(replace(run, signals=np.column_stack(signals)))
    return tuple(runs), library


def test_cv_recovers_distinct_hrfs_with_canonical_junk(planted_runs):
    """Selection recovers both planted HRFs with junk trials in a fixed canonical basis.

    The planted junk follows each feature's own HRF, so only the canonical
    feature is predicted exactly.
    """
    runs, library = planted_runs
    fit = importlib.import_module(
        "experiments.hrf_estimation.library_cv"
    ).fit_library_cv
    estimate = fit(runs[:3], library, {})
    assert [k.id for k in estimate.kernels] == [0, 1]
    assert estimate.status == ("ok", "ok")
    assert estimate.training_run_ids == tuple(r.run_id for r in runs[:3])
    scores = estimate.diagnostics["selected_cv_r2"]
    assert np.isclose(scores[0], 1) and 0.9 < scores[1] < 1


def test_outer_test_signals_cannot_change_selection(planted_runs):
    """Verify outer test signals cannot change selection."""
    runs, library = planted_runs
    fit = importlib.import_module(
        "experiments.hrf_estimation.library_cv"
    ).fit_library_cv
    first = fit(runs[:3], library, {})
    runs[-1].signals[:] = 1e20
    second = fit(runs[:3], library, {})
    assert first.kernels == second.kernels
    np.testing.assert_array_equal(
        first.diagnostics["selected_cv_r2"], second.diagnostics["selected_cv_r2"]
    )


def test_zero_signal_gets_no_hrf(planted_runs):
    """Verify zero signal gets no HRF."""
    runs, library = planted_runs
    fit = importlib.import_module(
        "experiments.hrf_estimation.library_cv"
    ).fit_library_cv
    estimate = fit(
        tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs), library, {}
    )
    assert estimate.kernels == (None, None)
    assert estimate.status == ("no_signal", "no_signal")


def test_cv_is_the_package_selection_on_the_mapped_inputs(planted_runs):
    """Library CV adds no numerics of its own: it reports the package's result."""
    from boldtailor.hrf_selection import select_hrfs
    from experiments.hrf_estimation import library_cv, package_selection

    runs, library = planted_runs
    ours = library_cv.fit_library_cv(runs, library, {})
    data, model = package_selection.selection_data(runs, library, "amplitude")
    package = select_hrfs(data, library=library, task_model=model)
    assert [k.id for k in ours.kernels] == package.hrf_indices.tolist()
    np.testing.assert_array_equal(ours.diagnostics["selected_cv_r2"], package.cv_r2)
    np.testing.assert_array_equal(
        ours.diagnostics["canonical_cv_r2"], package.canonical_cv_r2
    )
    assert ours.diagnostics["package_selection"]["name"] == "hrf_selection"
    assert ours.diagnostics["rt_mode"] == "amplitude"
    for module in (library_cv, package_selection):
        assert "boldtailor._" not in Path(module.__file__).read_text()


def test_mapped_inputs_keep_valid_trials_and_move_junk_to_the_nuisance(planted_runs):
    """Valid trials form the task model; missing and incorrect trials are nuisance."""
    from experiments.hrf_estimation import package_selection

    runs, library = planted_runs
    data, model = package_selection.selection_data(runs, library, "amplitude")
    assert model.regressor_names == ("task", "trial_type[1]", "response_time")
    assert [len(e) for e in data.events] == [22] * 4
    assert list(data.confounds[0].columns) == list(package_selection.JUNK_COLUMNS)
    timed, timed_model = package_selection.selection_data(runs, library, "duration")
    assert timed_model.regressor_names == ("task", "trial_type[1]")
    np.testing.assert_array_equal(
        timed.events[0].duration, timed.events[0].response_time
    )
    with pytest.raises(ValueError, match="rt_mode"):
        package_selection.selection_data(runs, library, "other")


@pytest.mark.parametrize("problem", ["constant_rt", "missing_condition"])
def test_rank_failure_is_an_error_not_a_no_signal_status(planted_runs, problem):
    """An unidentifiable task model is rejected by the package before any fit."""
    runs, library = planted_runs
    changed = []
    for run in runs:
        events = run.events.copy()
        if problem == "constant_rt":
            events.loc[events.response_time.notna(), "response_time"] = 1.0
        else:
            events["trial_type"] = 0
        changed.append(replace(run, events=events))
    fit = importlib.import_module(
        "experiments.hrf_estimation.library_cv"
    ).fit_library_cv
    with pytest.raises(ValueError, match="no eligible|no 'trial_type' trials"):
        fit(tuple(changed), library, {})


def test_training_rejects_mismatched_anatomy_at_same_columns(planted_runs):
    """Verify training rejects mismatched anatomy at same columns."""
    import nibabel as nib

    runs, library = planted_runs
    a = nib.cifti2.BrainModelAxis.from_surface([4, 7], 10, "CortexLeft")
    b = nib.cifti2.BrainModelAxis.from_surface([7, 4], 10, "CortexLeft")
    changed = (replace(runs[0], brain_axis=a), replace(runs[1], brain_axis=b))
    fit = importlib.import_module(
        "experiments.hrf_estimation.library_cv"
    ).fit_library_cv
    with pytest.raises(ValueError, match="anatom"):
        fit(changed, library, {})
