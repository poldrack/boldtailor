"""Scientific validation of task-guided denoising on predeclared synthetic data.

Seeds, effect sizes, and the outer run are fixed in `denoising_fixtures`
(VALIDATION_*) and were declared before these tests were first run. Evidence
of benefit comes only from known task coefficients and from an outer run that
never enters selection; the winning inner-CV score is not used as evidence.
"""

from dataclasses import replace
from pathlib import Path
import re

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
# under the original fixed 0.0 threshold the recovery checks below failed and
# were strict xfail. The pool statistic is the winning HRF's CV R², a maximum
# over library candidates, so pure noise is biased above 0 (more so for larger
# libraries) and shared noise makes those scores co-vary: 28 of 30 noise
# features scored above 0, the final pool held 2 features, and 0 PCs won.
# With the default pool_r2_threshold="auto" (per-fold Gaussian-mixture tail
# threshold, GLMsingle's rule) the same data give fold pools of 31-34
# features, 1 selected PC, and all four checks pass.


def denoise(dataset):
    return select_denoising(
        dataset.training,
        brain_mask=dataset.brain_mask,
        task_model=TaskModel(),
        library=dataset.library,
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
    assert recovery.training.n_runs == 4
    assert len(recovery_result.run_labels) == 4
    lengths = [c.shape[0] for c in recovery_result.run_components]
    assert lengths == [len(t) for t in recovery.training.frame_times]


def test_shared_noise_yields_a_positive_component_count(recovery_result):
    assert recovery_result.n_components > 0


def test_denoising_improves_training_coefficient_recovery(recovery, arms):
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


def test_without_shared_noise_zero_components_win_or_tie():
    dataset = make_validation_dataset("no_benefit")
    result = denoise(dataset)
    scores = result.candidate_scores
    eligible = scores[scores["eligible"]]
    zero = float(eligible.loc[eligible["count"] == 0, "mean_r2"].iloc[0])
    assert zero >= eligible["mean_r2"].max() - result.score_tolerance
    assert result.n_components == 0


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
        brain_mask=fixture.brain_mask,
        task_model=fixture.task_model,
        library=fixture.library,
        model=model,
    )
    exec(documented_example(), namespace)
    names = namespace["denoising"].component_names
    assert names
    assert namespace["fitted_model"].confounds == (*model.confounds, *names)
    assert namespace["result"].effect("task").shape == (data.n_features,)


def test_docs_describe_denoising_contract_and_limits():
    section = (
        Path("docs/user-guide.md").read_text().split("## Task-guided denoising")[1]
    )
    section = section.split("\n## ", 1)[0]
    for phrase in (
        "brain_mask",
        "sequential",
        "missing-value",
        "empty",
        "selection statistic",
        "outer",
        "frozen",
        "pool_r2_threshold",
        "denoising_augmentation",
    ):
        assert phrase in section, phrase
    api = Path("docs/api.md").read_text()
    assert "select_denoising" in api and "with_denoising" in api
    assert "DenoisingResult" in api and "run_labels" in api
    comparison = Path("docs/glmsingle-comparison.md").read_text()
    assert "select_denoising" in comparison and "optional" in comparison
    assert "task-guided" in comparison
