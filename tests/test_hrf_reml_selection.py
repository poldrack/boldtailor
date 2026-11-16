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
