"""Likelihood-selected HRFs retain the standard spatial assignment contract."""

from dataclasses import replace

import numpy as np
import pytest

from boldtailor._hrf_cv import subset_runs
from boldtailor._hrf_reml import prepare_reml_groups, iter_trial_designs
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import ModelSpec, TaskModel
from boldtailor.provenance import identity_activity
from boldtailor.reml import select_random_effects_design
from tests.hrf_reml_fixtures import reml_hrf_problem, planted_reml_hrfs


@pytest.mark.parametrize("rho", [0.0, 0.3, np.array([0.1, 0.2, 0.3])])
def test_reml_selection_matches_engine(planted_reml_hrfs, rho):
    """The public adapter preserves engine winners, likelihoods, and qualifications."""
    data, library = planted_reml_hrfs
    options = RemlHrfOptions(rho=rho)
    result = select_hrfs(data, library=library, method="reml", reml_options=options)
    labels = ["a", "b", "c"]
    groups = prepare_reml_groups(data, library, TaskModel(), options, labels)
    expected = select_random_effects_design(
        groups, iter_trial_designs(data, library, [0, 1], labels), rho=rho
    )
    np.testing.assert_array_equal(result.hrf_indices, expected.winner)
    np.testing.assert_allclose(
        result.reml.selection.fit.criterion, expected.fit.criterion
    )
    np.testing.assert_allclose(result.reml.selection.gap, expected.gap)
    np.testing.assert_array_equal(
        result.reml.selection.any_unrefined, expected.any_unrefined
    )
    assert result.reml.selection.fit.status == expected.fit.status
    assert result.cv_r2 is None and result.method == "reml"
    np.testing.assert_array_equal(result.hrf_indices, [1, 1, -1])
    assert result.reml.selection.fit.status[-1] == "no_residual_signal"


def test_cv_default_is_unchanged(planted_reml_hrfs):
    """Explicit CV produces the same values and identity as the existing default."""
    data, library = planted_reml_hrfs
    default = select_hrfs(data, library=library)
    explicit = select_hrfs(data, library=library, method="cv")
    np.testing.assert_array_equal(default.hrf_indices, explicit.hrf_indices)
    np.testing.assert_array_equal(default.cv_r2, explicit.cv_r2)
    assert identity_activity(default.provenance) == identity_activity(
        explicit.provenance
    )


def test_reml_can_select_from_one_run(planted_reml_hrfs):
    """A valid restricted model does not require CV's second run."""
    data, library = planted_reml_hrfs
    single = subset_runs(data, [0])
    result = select_hrfs(single, library=library, method="reml")
    assert result.hrf_indices[0] >= 0
    with pytest.raises(ValueError, match="two runs"):
        select_hrfs(single, library=library)


@pytest.mark.parametrize(
    "options",
    [
        dict(method="other"),
        dict(reml_options=RemlHrfOptions()),
        dict(method="reml", reml_options={}),
    ],
)
def test_invalid_selection_options_fail(planted_reml_hrfs, options):
    """Unsupported method combinations cannot silently fall back to CV."""
    data, library = planted_reml_hrfs
    with pytest.raises(ValueError, match="method|options|REML"):
        select_hrfs(data, library=library, **options)


def test_reml_records_fixed_model_and_rho(planted_reml_hrfs):
    """Identities distinguish the method, reference mean, and noise correlation."""
    data, library = planted_reml_hrfs
    records = []
    for options in [
        RemlHrfOptions(),
        RemlHrfOptions(rho=0.2),
        RemlHrfOptions(reference_hrf_id=1),
    ]:
        selection = select_hrfs(
            data, library=library, method="reml", reml_options=options
        )
        activity = identity_activity(selection.provenance)
        assert activity["method"] == "reml"
        assert activity["reference_hrf_id"] == options.reference_hrf_id
        assert len(activity["fixed_design_fingerprints"]) == data.n_runs
        records.append(activity)
    assert records[0] != records[1] and records[0] != records[2]


@pytest.mark.integration
def test_reml_selection_feeds_conventional_and_trial_fits(planted_reml_hrfs):
    """REML assignments fit through existing interfaces and preserve excluded features."""
    from boldtailor.fit import fit
    from boldtailor.single_trial import fit_selected_hrfs

    data, library = planted_reml_hrfs
    selection = select_hrfs(
        data, library=library, method="reml", feature_signature="axis"
    )
    trials = fit_selected_hrfs(data, hrf_selection=selection, feature_signature="axis")
    assert np.isfinite(trials.run_betas[0][:, :2]).all()
    assert np.isnan(trials.run_betas[0][:, -1]).all()
    model = ModelSpec(
        task_model=TaskModel(),
        contrasts={"task": {"task": 1}},
        hrf_model="spm",
        noise_model="ols",
        drift_model=None,
        confounds=("motion",),
    )
    result = fit(data, model, hrf_selection=selection, feature_signature="axis")
    assert np.isfinite(result.effect("task")[:2]).all()
    assert np.isnan(result.effect("task")[-1])
    with pytest.raises(ValueError, match="feature_signature"):
        fit_selected_hrfs(data, hrf_selection=selection)
    with pytest.raises(ValueError, match="assignment"):
        fit_selected_hrfs(
            data,
            hrf_selection=replace(selection, hrf_indices=np.zeros(3, dtype=int)),
            feature_signature="axis",
        )


def test_canonical_only_reml_has_no_runner_up(planted_reml_hrfs):
    """A one-candidate library has an undefined runner-up gap without an error."""
    from boldtailor.hrf_library import HrfLibrary

    data, _ = planted_reml_hrfs
    selection = select_hrfs(data, library=HrfLibrary.from_parameters([]), method="reml")
    assert np.isnan(selection.reml.selection.gap).all()
    assert np.all(selection.hrf_indices[:2] == 0)



def test_reml_split_is_independent_of_test_signals(planted_reml_hrfs):
    """Held-out BOLD changes scores but cannot alter training fits or their identity."""
    from boldtailor.data import from_arrays
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = planted_reml_hrfs
    options = dict(library=library, method="reml", train_runs=[0], test_runs=[1, 2])
    base = evaluate_hrf_split(data, **options)
    changed = from_arrays([data.signals[0], data.signals[1] * 7, data.signals[2] * -3],
                          data.events, frame_times=data.frame_times, confounds=data.confounds)
    altered = evaluate_hrf_split(changed, **options)
    np.testing.assert_array_equal(base.training_selection.hrf_indices, altered.training_selection.hrf_indices)
    np.testing.assert_array_equal(base.training_selection.reml.selection.fit.ratio, altered.training_selection.reml.selection.fit.ratio)
    np.testing.assert_array_equal(base.training_amplitudes, altered.training_amplitudes)
    assert identity_activity(base.training_selection.provenance) == identity_activity(altered.training_selection.provenance)
    assert not np.allclose(base.test_r2[:2], altered.test_r2[:2])
    assert np.isnan(base.test_r2[-1])


def test_reml_split_matches_training_only_mean_prediction(planted_reml_hrfs):
    """Evaluation fits shared mean amplitudes, not the random trial coefficients."""
    from boldtailor._hrf_design import trial_regressors
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = planted_reml_hrfs
    result = evaluate_hrf_split(data, library=library, method="reml", train_runs=[0], test_runs=[1])
    for feature in range(2):
        cid = result.training_selection.hrf_indices[feature]
        residuals = []
        for r in [0, 1]:
            nuisance = np.column_stack([data.confounds[r], np.ones(len(data.signals[r]))])
            x = trial_regressors(data.events[r], data.frame_times[r], library.candidates[cid]).sum(axis=1)
            y = data.signals[r][:, feature]
            residuals.append((x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0],
                              y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]))
        x, y = residuals[0]
        beta = (x @ y) / (x @ x)
        tx, ty = residuals[1]
        expected = 1 - np.sum((ty - tx * beta) ** 2) / np.sum(ty ** 2)
        np.testing.assert_allclose(result.training_amplitudes[0, feature], beta)
        np.testing.assert_allclose(result.test_r2[feature], expected)


def test_reml_split_rejects_overlapping_runs(planted_reml_hrfs):
    """REML does not relax the independent evaluation boundary."""
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = planted_reml_hrfs
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_hrf_split(data, library=library, method="reml", train_runs=[0], test_runs=[0])



def test_reml_split_keeps_winner_when_test_mean_is_unestimable(planted_reml_hrfs):
    """A test confound collinear with the selected task invalidates evaluation only."""
    from boldtailor.data import from_arrays
    from boldtailor._hrf_design import trial_regressors
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = planted_reml_hrfs
    confounds = list(data.confounds)
    confounds[1]["task_overlap"] = trial_regressors(data.events[1], data.frame_times[1], library.candidates[1]).sum(axis=1)
    changed = from_arrays(data.signals, data.events, frame_times=data.frame_times, confounds=confounds)
    result = evaluate_hrf_split(changed, library=library, method="reml", train_runs=[0], test_runs=[1])
    assert result.training_selection.hrf_indices[0] == 1
    assert np.isnan(result.test_r2[0])
    assert identity_activity(result.provenance)["prediction_ineligible_hrfs"]
