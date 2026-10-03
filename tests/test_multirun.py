import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared
from tests.oracles import peak_design_matrix, nilearn_original_space_diagnostics


def _problem():
    rng = np.random.default_rng(11)
    events = (
        pd.DataFrame(
            {
                "onset": [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0],
                "duration": np.ones(7),
                "trial_type": [
                    "face",
                    "house",
                    "button",
                    "face",
                    "house",
                    "face",
                    "house",
                ],
            }
        ),
        pd.DataFrame(
            {
                "onset": np.arange(0.0, 112.0, 8.0),
                "duration": np.ones(14),
                "trial_type": ["face", "house"] * 7,
            }
        ),
    )
    frame_times = (np.arange(30) * 2.0, np.arange(60) * 2.0)
    designs = tuple(
        peak_design_matrix(
            times,
            events=run_events,
            hrf_model="glover",
            drift_model="cosine",
            high_pass=0.01,
            min_onset=-24.0,
        )
        for times, run_events in zip(frame_times, events)
    )
    signals = []
    for design in designs:
        beta = np.zeros((design.shape[1], 2))
        beta[design.columns.get_loc("face")] = [2.0, 1.0]
        beta[design.columns.get_loc("house")] = [0.5, 1.5]
        beta[design.columns.get_loc("constant")] = [10.0, 12.0]
        noise = rng.normal(0.0, 0.05, (len(design), 2))
        signals.append(design.to_numpy() @ beta + noise)
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model="cosine",
        noise_model="ols",
    )
    return tuple(signals), events, frame_times, designs, model


def _as_image(signals):
    data = signals.T.reshape(2, 1, 1, signals.shape[0])
    return nib.Nifti1Image(data, np.eye(4))


def _flat_values(image):
    return image.get_fdata().reshape(-1)


def test_fit_allows_run_specific_designs_and_matches_first_level_model():
    signals, events, frame_times, designs, model = _problem()
    data = from_arrays(signals, events, frame_times=frame_times)

    result = fit(data, model)
    mask = nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4))
    oracle = FirstLevelModel(
        mask_img=mask,
        noise_model="ols",
        signal_scaling=False,
    )
    with pytest.warns(RuntimeWarning, match="Generation of a mask"):
        oracle.fit(
            [_as_image(run) for run in signals],
            design_matrices=list(designs),
        )
    with pytest.warns(RuntimeWarning, match="same contrast"):
        expected = oracle.compute_contrast(
            "face - house",
            output_type="all",
        )

    assert tuple(designs[0].columns) != tuple(designs[1].columns)
    np.testing.assert_allclose(
        result.effect("face_gt_house"),
        _flat_values(expected["effect_size"]),
    )
    np.testing.assert_allclose(
        result.variance("face_gt_house"),
        _flat_values(expected["effect_variance"]),
    )
    np.testing.assert_allclose(
        result.stat("face_gt_house"),
        _flat_values(expected["stat"]),
    )
    np.testing.assert_allclose(
        result.z_score("face_gt_house"),
        _flat_values(expected["z_score"]),
    )
    np.testing.assert_allclose(
        result.one_sided_p_value("face_gt_house"),
        _flat_values(expected["p_value"]),
    )
    assert len(result.run_r2) == 2


def test_fit_aggregates_r2_from_sums_not_run_means():
    signals, events, frame_times, _, model = _problem()
    result = fit(
        from_arrays(signals, events, frame_times=frame_times),
        model,
    )
    run_total = [np.sum((run - run.mean(axis=0)) ** 2, axis=0) for run in signals]
    run_residual = [
        (1.0 - score) * total for score, total in zip(result.run_r2, run_total)
    ]
    expected = 1.0 - np.sum(run_residual, axis=0) / np.sum(run_total, axis=0)

    np.testing.assert_allclose(result.r2, expected)


def test_fit_ar1_r2_uses_original_signal_space_across_runs():
    signals, events, frame_times, designs, _ = _problem()
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model="cosine",
        noise_model="ar1",
    )

    result = fit(
        from_arrays(signals, events, frame_times=frame_times),
        model,
    )
    diagnostics = tuple(
        nilearn_original_space_diagnostics(run, design, "ar1")
        for run, design in zip(signals, designs, strict=True)
    )
    residual_sum = np.sum([values[1] for values in diagnostics], axis=0)
    total_sum = np.sum([values[2] for values in diagnostics], axis=0)
    aggregate_r2 = 1.0 - residual_sum / total_sum

    for actual, expected in zip(result.run_r2, diagnostics, strict=True):
        np.testing.assert_allclose(actual, expected[0])
    np.testing.assert_allclose(result.r2, aggregate_r2)


def test_fit_warns_for_rank_deficient_design():
    signals, events, frame_times, _, _ = _problem()
    confound = pd.DataFrame({"duplicate": np.ones(len(signals[0]))})
    model = ModelSpec(
        contrasts={"face": {"face": 1.0}},
        confounds=("duplicate",),
        drift_model=None,
        noise_model="ols",
    )

    with pytest.warns(UserWarning) as caught:
        fit(
            from_arrays(
                signals[0],
                events[0],
                frame_times=frame_times[0],
                confounds=confound,
            ),
            model,
        )
    messages = {str(warning.message) for warning in caught}

    assert any("design rank" in message for message in messages)


def test_fit_records_run_diagnostics_without_serializing_design_values(
    complete_sources,
):
    signals, events, frame_times, designs, model = _problem()

    result = fit(
        from_arrays(
            signals,
            events,
            frame_times=frame_times,
            sources=complete_sources(2),
        ),
        model,
    )
    runs = result.provenance.activities[-1]["runs"]
    serialized = result.provenance.canonical_json()

    assert len(runs) == 2
    for index, run in enumerate(runs):
        assert run["timing_source"] == "frame_times"
        assert run["n_scans"] == signals[index].shape[0]
        assert run["n_features"] == signals[index].shape[1]
        assert run["design_columns"] == list(result.design_matrices[index].columns)
        assert run["design_rank"] == int(
            np.linalg.matrix_rank(result.design_matrices[index].to_numpy())
        )
        assert run["residual_dof"] == designs[index].shape[0] - run["design_rank"]
        assert run["excluded_event_count"] == 0
        assert run["min_onset_cutoff"] == -24.0

    assert "design_matrix" not in serialized
    assert "residual_sum" not in serialized
    assert "total_sum" not in serialized
    assert "r2" not in serialized


def test_fit_records_rank_deficiency_warning_in_provenance(complete_sources):
    signals, events, frame_times, _, _ = _problem()
    confound = pd.DataFrame({"duplicate": np.ones(len(signals[0]))})
    model = ModelSpec(
        contrasts={"face": {"face": 1.0}},
        confounds=("duplicate",),
        drift_model=None,
        noise_model="ols",
    )

    with pytest.warns(UserWarning):
        result = fit(
            from_arrays(
                signals[0],
                events[0],
                frame_times=frame_times[0],
                confounds=confound,
                sources=complete_sources(2)[:1],
            ),
            model,
        )

    run = result.provenance.activities[-1]["runs"][0]

    assert run["design_rank"] < len(run["design_columns"])
    assert any("design rank" in warning for warning in run["warnings"])


def _fit_single_run(contrasts):
    signals, events, frame_times, _, _ = _problem()
    model = ModelSpec(contrasts=contrasts, drift_model=None, noise_model="ols")
    fit(from_arrays(signals[0], events[0], frame_times=frame_times[0]), model)


def _fit_single_run_with_duplicate_confound(contrasts):
    signals, events, frame_times, designs, _ = _problem()
    confound = pd.DataFrame({"duplicate": designs[0]["face"].to_numpy()})
    model = ModelSpec(
        contrasts=contrasts,
        confounds=("duplicate",),
        drift_model="cosine",
        noise_model="ols",
    )
    data = from_arrays(
        signals[0], events[0], frame_times=frame_times[0], confounds=confound
    )
    fit(data, model)


def _fit_two_runs(contrasts):
    signals, events, frame_times, _, _ = _problem()
    model = ModelSpec(contrasts=contrasts, noise_model="ols")
    fit(from_arrays(signals, events, frame_times=frame_times), model)


def _prepared(designs, roles):
    designs = designs if isinstance(designs, tuple) else (designs,)
    signals = tuple(d.to_numpy() @ np.ones((d.shape[1], 1)) for d in designs)
    return PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=roles,
    )


def _fit_prepared_single_run(contrasts):
    rng = np.random.default_rng(20260811)
    design = pd.DataFrame(
        {"face": rng.normal(size=12), "house": rng.normal(size=12), "constant": 1.0}
    )
    roles = {"face": "task", "house": "task", "constant": "intercept"}
    fit_prepared(_prepared(design, roles), contrasts=contrasts, noise_model="ols")


def _fit_prepared_duplicate_columns(contrasts):
    design = pd.DataFrame(
        {
            "face": [0.0, 1.0, 0.0, 1.0, 0.0],
            "duplicate": [0.0, 1.0, 0.0, 1.0, 0.0],
            "constant": 1.0,
        }
    )
    roles = {"face": "task", "duplicate": "task", "constant": "intercept"}
    fit_prepared(_prepared(design, roles), contrasts=contrasts, noise_model="ols")


def _fit_prepared_two_runs(contrasts):
    designs = (
        pd.DataFrame({"face": [0.0, 1.0, 0.0, 1.0], "constant": 1.0}),
        pd.DataFrame({"constant": 1.0, "house": [0.0, 1.0, 0.0, 1.0]}),
    )
    roles = (
        {"face": "task", "constant": "intercept"},
        {"constant": "intercept", "house": "task"},
    )
    fit_prepared(_prepared(designs, roles), contrasts=contrasts, noise_model="ols")


_MISSING_TERM = "run 1.*contrast '{0}'.*missing regressor '{0}'"
_ALL_ZERO = "run 0.*contrast 'zero'.*resolves to all zeros"
_NOT_ESTIMABLE = "run 0.*contrast 'difference'.*not estimable"
_DIFFERENCE = {"difference": {"face": 1.0, "duplicate": -1.0}}


@pytest.mark.parametrize(
    ("entry_point", "contrasts", "message", "warning"),
    [
        (
            _fit_single_run,
            {"missing": "not_a_column"},
            "run 0.*contrast 'missing'",
            None,
        ),
        (
            _fit_single_run,
            {"missing": {"not_a_column": 1.0}},
            "run 0.*contrast 'missing'",
            None,
        ),
        (
            _fit_single_run_with_duplicate_confound,
            _DIFFERENCE,
            _NOT_ESTIMABLE,
            "design rank",
        ),
        (
            _fit_two_runs,
            {"button": {"button": 1.0}},
            _MISSING_TERM.format("button"),
            None,
        ),
        (_fit_two_runs, {"zero": "face - face"}, _ALL_ZERO, None),
        (
            _fit_prepared_two_runs,
            {"face": {"face": 1.0}},
            _MISSING_TERM.format("face"),
            None,
        ),
        (_fit_prepared_duplicate_columns, _DIFFERENCE, _NOT_ESTIMABLE, "design rank"),
        (_fit_prepared_single_run, {"zero": "face - face"}, _ALL_ZERO, None),
    ],
)
def test_contrast_preflight_rejects_before_glm(
    fail_glm, entry_point, contrasts, message, warning
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=message):
            entry_point(contrasts)

    if warning:
        assert any(warning in str(item.message) for item in caught)


def _per_run_nilearn(signals, designs, vectors, noise_model):
    from nilearn.glm import compute_contrast
    from nilearn.glm.first_level import run_glm

    out = []
    for y, x, vector in zip(signals, designs, vectors, strict=True):
        labels, results = run_glm(y, x.to_numpy(), noise_model=noise_model)
        out.append(compute_contrast(labels, results, vector, stat_type="t"))
    return out


def _column_vector(design, weights):
    vector = np.zeros(design.shape[1])
    for name, weight in weights.items():
        vector[design.columns.get_loc(name)] = weight
    return vector


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_multirun_contrasts_are_equal_weight_fixed_effects(noise_model):
    from scipy import stats

    signals, events, frame_times, designs, model = _problem()
    model = ModelSpec(
        contrasts=model.contrasts, drift_model="cosine", noise_model=noise_model
    )
    result = fit(from_arrays(signals, events, frame_times=frame_times), model)
    weights = {"face": 1.0, "house": -1.0}
    vectors = [_column_vector(d, weights) for d in designs]
    runs = _per_run_nilearn(signals, designs, vectors, noise_model)
    n = len(runs)
    effect = sum(r.effect_size() for r in runs) / n
    variance = sum(r.effect_variance() for r in runs) / n**2
    dof = sum(r.dof for r in runs)
    t = effect / np.sqrt(variance)
    name = "face_gt_house"
    np.testing.assert_allclose(result.effect(name), effect, rtol=1e-8)
    np.testing.assert_allclose(result.variance(name), variance, rtol=1e-8)
    np.testing.assert_allclose(result.stat(name), t, rtol=1e-8)
    np.testing.assert_allclose(
        result.one_sided_p_value(name), stats.t.sf(t, dof), rtol=1e-8
    )


def _duplicated_fit_and_reduced_oracle():
    from nilearn.glm import compute_contrast
    from nilearn.glm.first_level import run_glm

    signals, _, frame_times, designs, _ = _problem()
    design, y = designs[0], signals[0]
    duplicated = design.assign(face_copy=design["face"])
    roles = {c: "nuisance" for c in duplicated.columns}
    roles.update({c: "task" for c in ("face", "face_copy", "house")})
    roles["constant"] = "intercept"
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=[y],
        design_matrices=[duplicated],
        frame_times=[frame_times[0]],
        column_roles=[roles],
    )
    contrasts = {"c": {"face": 0.5, "face_copy": 0.5, "house": -1.0}}
    with pytest.warns(UserWarning, match="design rank"):
        result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")
    labels, fitted = run_glm(y, design.to_numpy(), noise_model="ols")
    # duplicated columns identify only b_f + b_c: 0.5*b_f + 0.5*b_c folds to 0.5.
    vector = _column_vector(design, {"face": 0.5, "house": -1.0})
    return result, compute_contrast(labels, fitted, vector, stat_type="t")


def test_duplicated_regressor_effect_matches_reduced_design():
    result, expected = _duplicated_fit_and_reduced_oracle()
    np.testing.assert_allclose(result.effect("c"), expected.effect_size(), rtol=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "finding: on a rank-deficient design nilearn's OLS dispersion divides by "
        "n - n_columns (24) while df_residuals is n - rank (25), so the variance is "
        "25/24 larger than on the reduced full-rank design"
    ),
)
def test_duplicated_regressor_variance_and_stat_match_reduced_design():
    result, expected = _duplicated_fit_and_reduced_oracle()
    np.testing.assert_allclose(
        result.variance("c"), expected.effect_variance(), rtol=1e-8
    )
    np.testing.assert_allclose(result.stat("c"), expected.stat(), rtol=1e-8)
