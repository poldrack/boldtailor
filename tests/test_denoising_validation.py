"""Scientific validation of task-guided denoising on predeclared synthetic data.

Seeds, effect sizes, and the outer run are fixed in `denoising_fixtures`
(VALIDATION_*) and were declared before these tests were first run. Evidence
of benefit comes only from known task coefficients and from an outer run that
never enters selection; the winning inner-CV score is not used as evidence.
"""

from dataclasses import replace
from pathlib import Path
import re
import tomllib

import numpy as np
import pytest

from boldtailor._hrf_design import convolve_events
from boldtailor.data import from_arrays
from boldtailor.denoising import select_denoising, with_denoising
from boldtailor.fit import fit
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import ModelSpec, TaskModel
from tests.denoising_fixtures import make_denoising_fixture, make_validation_dataset

BASELINE = ("drift", "cosine")
MODEL = ModelSpec(contrasts={"task": "task"}, confounds=BASELINE, drift_model=None)

# Recorded history (seeds and effect sizes unchanged since predeclaration):
# under the original fixed 0.0 threshold on the winning HRF's CV R² the
# recovery checks below failed and were strict xfail (28 of 30 noise features
# scored above 0; the final pool held 2 features; 0 PCs won). Task 5's
# per-fold Gaussian-mixture threshold made them pass. Task 6 replaced the
# procedure with the GLMsingle-aligned one (full-data HRFs, ON-OFF R² pool,
# single pool, median performance, pcstop); the checks were rerun unchanged.
# Under Task 6 the no-benefit tests failed (pcstop chose 6 PCs on independent
# noise) and were strict xfail. Task 7 added the predeclared F-test
# significance gate (default settings); rerun once unchanged, the gate rejects
# k*=6 (m=1 of n=10, binomial p=0.40) and the recovery data keep k*=1 (m=6 of
# n=6), so the xfail markers were removed. Task 8 made the gate AR(1)
# prewhitened by default (gate_noise_model="ar1") and added the predeclared
# independent-AR(0.5) no-benefit dataset (seed 20261006); the checks above
# and below were rerun unchanged with the default gate.


def denoise(dataset):
    return select_denoising(
        dataset.training, task_model=TaskModel(), library=dataset.library
    )


def selected_fit(data, library, names=()):
    selection = select_hrfs(data, library=library, task_model=TaskModel())
    model = replace(MODEL, confounds=(*BASELINE, *names))
    return selection, fit(data, model, hrf_selection=selection)


def rmse(estimate, truth):
    return float(np.sqrt(np.mean((np.asarray(estimate) - truth) ** 2)))


@pytest.fixture(scope="module")
def recovery():
    return make_validation_dataset("recovery")


@pytest.fixture(scope="module")
def recovery_result(recovery):
    return denoise(recovery)


@pytest.fixture(scope="module")
def arms(recovery, recovery_result):
    """(selection, fit) on training runs without and with the selected PCs."""
    augmented = with_denoising(recovery.training, recovery_result)
    names = recovery_result.component_names
    return dict(
        baseline=selected_fit(recovery.training, recovery.library),
        denoised=selected_fit(augmented, recovery.library, names),
    )


def test_selection_never_sees_the_outer_run(recovery, recovery_result):
    """Verify selection never sees the outer run."""
    assert recovery.training.n_runs == 4
    assert len(recovery_result.run_labels) == 4
    lengths = [c.shape[0] for c in recovery_result.run_components]
    assert lengths == [len(t) for t in recovery.training.frame_times]


def test_shared_noise_yields_a_positive_component_count(recovery_result):
    """Verify shared noise yields a positive component count."""
    assert recovery_result.n_components > 0


def test_denoising_improves_training_coefficient_recovery(recovery, arms):
    """Verify denoising improves training coefficient recovery."""
    errors = {
        arm: rmse(result.effect("task")[recovery.task], recovery.amplitudes)
        for arm, (_, result) in arms.items()
    }
    assert errors["denoised"] < errors["baseline"]


def predicted_task_signal(dataset, selection, effects):
    """Outer-run task time series from training coefficients and frozen HRFs."""
    events, times = dataset.outer.events[0], dataset.outer.frame_times[0]
    columns = []
    for feature in dataset.task:
        candidate = dataset.library.candidates[selection.hrf_indices[feature]]
        regressor = sum(
            convolve_events([onset], [duration], times, candidate)
            for onset, duration in zip(events.onset, events.duration)
        )
        columns.append(effects[feature] * regressor)
    return np.column_stack(columns)


def test_frozen_training_estimates_better_predict_the_outer_task_signal(recovery, arms):
    """Verify frozen training estimates better predict the outer task signal."""
    truth = recovery.outer_task_signal[:, recovery.task]
    errors = {
        arm: rmse(
            predicted_task_signal(recovery, selection, result.effect("task")), truth
        )
        for arm, (selection, result) in arms.items()
    }
    assert errors["denoised"] < errors["baseline"]


def frozen_pool_pcs(outer, pool, count):
    """Outer-run PCs from the frozen pool: lstsq-projected, normalized, SVD."""
    confounds = outer.confounds[0].to_numpy()
    nuisance = np.column_stack([confounds, np.ones(len(confounds))])
    y = outer.signals[0][:, pool]
    residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    residual = residual / np.linalg.norm(residual, axis=0)
    return np.linalg.svd(residual, full_matrices=False)[0][:, :count]


def outer_with_pcs(outer, components, names):
    confounds = outer.confounds[0].assign(**dict(zip(names, components.T)))
    return from_arrays(
        list(outer.signals),
        list(outer.events),
        frame_times=list(outer.frame_times),
        confounds=[confounds],
    )


def test_frozen_pool_and_count_improve_outer_run_beta_recovery(
    recovery, recovery_result, arms
):
    """Verify frozen pool and count improve outer run beta recovery."""
    names = recovery_result.component_names
    components = frozen_pool_pcs(
        recovery.outer, recovery_result.noise_pool, recovery_result.n_components
    )
    denoised_outer = outer_with_pcs(recovery.outer, components, names)
    baseline = fit(recovery.outer, MODEL, hrf_selection=arms["baseline"][0])
    denoised = fit(
        denoised_outer,
        replace(MODEL, confounds=(*BASELINE, *names)),
        hrf_selection=arms["denoised"][0],
    )
    truth = recovery.amplitudes
    task = recovery.task
    assert rmse(denoised.effect("task")[task], truth) < rmse(
        baseline.effect("task")[task], truth
    )


@pytest.fixture(scope="module")
def no_benefit():
    return make_validation_dataset("no_benefit")


def test_without_shared_noise_zero_components_are_chosen(no_benefit):
    # Requirement change (Task 6): score_tolerance was replaced by GLMsingle's
    # pcstop rule, so "zero wins or ties" is now "the rule chooses zero".
    """Verify without shared noise zero components are chosen."""
    assert denoise(no_benefit).n_components == 0


def test_without_shared_noise_the_default_library_also_chooses_zero(no_benefit):
    """Verify without shared noise the default library also chooses zero."""
    result = select_denoising(no_benefit.training, task_model=TaskModel())
    assert result.n_components == 0


@pytest.fixture(scope="module")
def no_benefit_ar1():
    return make_validation_dataset("no_benefit_ar1")


def test_independent_autocorrelated_noise_chooses_zero_components(no_benefit_ar1):
    """Task 8 criterion (1): independent AR(0.5) noise, default (ar1) gate."""
    result = denoise(no_benefit_ar1)
    assert result.significance_gate.noise_model == "ar1"
    assert result.n_components == 0


def lag1_after_baseline(data, run, features):
    """Median lag-1 autocorrelation of features after removing the confounds."""
    confounds = data.confounds[run].to_numpy()
    nuisance = np.column_stack([confounds, np.ones(len(confounds))])
    y = data.signals[run][:, features]
    e = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    return np.median([np.corrcoef(c[:-1], c[1:])[0, 1] for c in e.T])


def test_ar1_validation_noise_is_independent_ar05(no_benefit_ar1, no_benefit):
    """The AR dataset keeps the no-benefit layout; its noise is AR(0.5)."""
    a, b = no_benefit_ar1.training, no_benefit.training
    assert [len(t) for t in a.frame_times] == [len(t) for t in b.frame_times]
    assert a.n_features == b.n_features
    noise = np.arange(len(no_benefit.amplitudes), a.n_features)
    for run in range(a.n_runs):
        assert 0.35 < lag1_after_baseline(a, run, noise) < 0.6
        assert abs(lag1_after_baseline(b, run, noise)) < 0.15


# ---- documentation example -------------------------------------------------------


def documented_example():
    guide = Path("docs/user-guide.md").read_text()
    section = guide.split("## Task-guided denoising", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_user_guide_denoising_example_runs():
    """The guide's example, verbatim, on data where a positive count is chosen."""
    fixture = make_denoising_fixture()
    events = [e.assign(trial_type="task") for e in fixture.data.events]
    data = from_arrays(
        list(fixture.data.signals),
        events,
        frame_times=list(fixture.data.frame_times),
        confounds=list(fixture.data.confounds),
    )
    model = ModelSpec(
        contrasts={"task": "task"},
        confounds=("motion_x", "drift", "cosine"),
        drift_model=None,
        task_model=fixture.task_model,
    )
    namespace = dict(
        data=data,
        task_model=fixture.task_model,
        library=fixture.library,
        model=model,
    )
    exec(documented_example(), namespace)
    names = namespace["denoising"].component_names
    assert names
    assert namespace["fitted_model"].confounds == (*model.confounds, *names)
    assert namespace["result"].effect("task").shape == (data.n_features,)


GLMSINGLE_NOTICE = "src/boldtailor/_resources/GLMsingle-LICENSE.txt"


def test_glmsingle_attribution_ships_and_is_cited():
    """BSD 3-Clause notice sits beside the bundled HRF data and is cited."""
    notice = Path(GLMSINGLE_NOTICE).read_text()
    assert "BSD 3-Clause License" in notice
    assert "Copyright (c) 2021, Kendrick Kay" in notice
    for path in ("docs/user-guide.md", "docs/api.md", "docs/glmsingle-comparison.md"):
        text = Path(path).read_text()
        assert "10.7554/eLife.77599" in text and "e77599" in text, path
        assert GLMSINGLE_NOTICE in text, path
    for module in (
        "_mixture_threshold",
        "denoising",
    ):
        source = Path(f"src/boldtailor/{module}.py").read_text()
        docstring = source.split('"""', 2)[1]
        assert GLMSINGLE_NOTICE in docstring, module
        assert "10.7554/eLife.77599" in docstring, module
    resource_note = Path("src/boldtailor/_resources/glmsingle_hrf_library.md")
    assert "GLMsingle-LICENSE.txt" in resource_note.read_text()


def test_package_is_mit_with_glmsingle_data_notice_only():
    """Boldtailor is MIT; the only other license covers the bundled HRF data."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project = pyproject["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE", GLMSINGLE_NOTICE]
    assert Path("LICENSE").read_text().startswith("MIT License")
    assert not Path("LICENSES").exists()
    readme = Path("README.md").read_text()
    assert "MIT License" in readme and GLMSINGLE_NOTICE in readme
