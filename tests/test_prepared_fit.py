import warnings

import numpy as np
import pandas as pd
import pytest
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import run_glm

from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared


@pytest.fixture
def prepared_problem():
    rng = np.random.default_rng(20260811)
    n_scans = 48
    design = pd.DataFrame(
        {
            "face": rng.normal(size=n_scans),
            "house": rng.normal(size=n_scans),
            "motion": rng.normal(size=n_scans),
            "constant": np.ones(n_scans),
        }
    )
    coefficients = np.array([[2.5, 1.0], [-1.5, 0.5], [0.25, -0.25], [10.0, 12.0]])
    signals = design.to_numpy() @ coefficients
    signals += rng.normal(0.0, 0.15, signals.shape)
    prepared = _prepared(signals, design)
    contrasts = {"face_gt_house": "face - house"}

    def expected(noise_model):
        labels, regression_results = run_glm(
            signals,
            design.to_numpy(),
            noise_model=noise_model,
        )
        vector = expression_to_contrast_vector("face - house", design.columns)
        return _nilearn_t_contrast(labels, regression_results, vector)

    return prepared, contrasts, expected


def _prepared(signals, design, *, roles=None):
    if roles is None:
        roles = {
            name: "intercept" if name == "constant" else "task"
            for name in design.columns
        }
    return PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=design,
        tr=2.0,
        column_roles=roles,
    )


def _nilearn_t_contrast(labels, regression_results, vector):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^divide by zero encountered in divide$",
            category=RuntimeWarning,
            module=r"^nilearn\.glm\._utils$",
        )
        return compute_contrast(labels, regression_results, vector, stat_type="t")


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_fit_prepared_matches_nilearn(noise_model, prepared_problem):
    prepared, contrasts, expected_for = prepared_problem
    expected = expected_for(noise_model)

    result = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model=noise_model,
        model_metadata={"origin": "fitlins", "node": "run"},
    )

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(
        result.variance("face_gt_house"), expected.effect_variance()
    )
    np.testing.assert_allclose(result.stat("face_gt_house"), expected.stat())
    np.testing.assert_allclose(result.z_score("face_gt_house"), expected.z_score())
    np.testing.assert_allclose(
        result.one_sided_p_value("face_gt_house"), expected.p_value()
    )


def test_fit_prepared_combines_run_specific_designs_and_pools_r_squared():
    rng = np.random.default_rng(20260812)
    designs = (
        pd.DataFrame(
            {
                "face": rng.normal(size=40),
                "motion_first": rng.normal(size=40),
                "house": rng.normal(size=40),
                "constant": np.ones(40),
            }
        ),
        pd.DataFrame(
            {
                "constant": np.ones(64),
                "house": rng.normal(size=64),
                "motion_second": rng.normal(size=64),
                "face": rng.normal(size=64),
            }
        ),
    )
    signals = tuple(
        design.to_numpy()
        @ np.array(
            [
                [2.0, 1.0],
                [0.25, -0.25],
                [-1.0, 0.5],
                [10.0, 12.0],
            ]
            if design.columns[0] == "face"
            else [
                [10.0, 12.0],
                [-1.0, 0.5],
                [0.25, -0.25],
                [2.0, 1.0],
            ]
        )
        + rng.normal(0.0, 0.15, (len(design), 2))
        for design in designs
    )
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=(
            {
                "face": "task",
                "motion_first": "nuisance",
                "house": "task",
                "constant": "intercept",
            },
            {
                "constant": "intercept",
                "house": "task",
                "motion_second": "nuisance",
                "face": "task",
            },
        ),
    )
    expected_contrasts = []
    residual_sums = []
    total_sums = []
    for signal, design in zip(signals, designs, strict=True):
        matrix = design.to_numpy()
        labels, regression_results = run_glm(signal, matrix, noise_model="ols")
        vector = expression_to_contrast_vector("face - house", design.columns)
        expected_contrasts.append(
            _nilearn_t_contrast(labels, regression_results, vector)
        )
        prediction = np.empty_like(signal)
        for label, fit in regression_results.items():
            prediction[:, labels == label] = matrix @ fit.theta
        residual_sums.append(np.sum((signal - prediction) ** 2, axis=0))
        total_sums.append(np.sum((signal - signal.mean(axis=0)) ** 2, axis=0))
    expected = 0.5 * (expected_contrasts[0] + expected_contrasts[1])
    expected_r2 = 1.0 - np.sum(residual_sums, axis=0) / np.sum(total_sums, axis=0)

    result = fit_prepared(
        prepared,
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        noise_model="ols",
    )

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(
        result.variance("face_gt_house"), expected.effect_variance()
    )
    np.testing.assert_allclose(result.r2, expected_r2)
    assert not np.allclose(result.r2, np.mean(result.run_r2, axis=0))


@pytest.mark.parametrize(
    ("contrasts", "message"),
    [
        ({}, "at least one contrast"),
        ({"face": {}}, "at least one weight"),
        ({"face": {"face": "one"}}, "weights must be numeric"),
        ({"face": {"face": np.inf}}, "weights must be finite"),
        ({"face": {"face": True}}, "weights must be numeric"),
        ({"face": {"face": 0.0}}, "nonzero weight"),
        ({"face": ""}, "must not be empty"),
        ({"face": "face + missing"}, "contrast 'face'.*invalid"),
    ],
)
def test_fit_prepared_rejects_invalid_semantic_contrasts(
    prepared_problem, contrasts, message
):
    prepared, _, _ = prepared_problem

    with pytest.raises(ValueError, match=message):
        fit_prepared(prepared, contrasts=contrasts, noise_model="ols")


def test_fit_prepared_rejects_contrast_term_missing_from_one_run():
    designs = (
        pd.DataFrame({"face": [0.0, 1.0, 0.0, 1.0], "constant": 1.0}),
        pd.DataFrame({"constant": 1.0, "house": [0.0, 1.0, 0.0, 1.0]}),
    )
    signals = tuple(design.to_numpy() @ np.array([[1.0], [5.0]]) for design in designs)
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=(
            {"face": "task", "constant": "intercept"},
            {"constant": "intercept", "house": "task"},
        ),
    )

    with pytest.raises(
        ValueError,
        match="run 1.*contrast 'face'.*missing regressor 'face'",
    ):
        fit_prepared(
            prepared,
            contrasts={"face": {"face": 1.0}},
            noise_model="ols",
        )


def test_fit_prepared_rejects_non_estimable_contrast():
    design = pd.DataFrame(
        {
            "face": [0.0, 1.0, 0.0, 1.0, 0.0],
            "duplicate": [0.0, 1.0, 0.0, 1.0, 0.0],
            "constant": 1.0,
        }
    )
    prepared = _prepared(
        design.to_numpy() @ np.array([[1.0], [1.0], [5.0]]),
        design,
    )

    with pytest.warns(UserWarning, match="design rank"):
        with pytest.raises(ValueError, match="contrast 'difference'.*not estimable"):
            fit_prepared(
                prepared,
                contrasts={"difference": {"face": 1.0, "duplicate": -1.0}},
                noise_model="ols",
            )


def test_fit_prepared_rejects_nonpositive_residual_degrees_of_freedom():
    design = pd.DataFrame(
        {"face": [1.0, 0.0, 0.0], "house": [0.0, 1.0, 0.0], "constant": 1.0}
    )
    prepared = _prepared(design.to_numpy() @ np.array([[1.0], [2.0], [5.0]]), design)

    with pytest.raises(ValueError, match="residual degrees of freedom 0"):
        fit_prepared(prepared, contrasts={"face": {"face": 1.0}}, noise_model="ols")


def test_fit_prepared_rejects_unsupported_noise_model(prepared_problem):
    prepared, contrasts, _ = prepared_problem

    with pytest.raises(ValueError, match="noise_model must be 'ols' or 'ar1'"):
        fit_prepared(prepared, contrasts=contrasts, noise_model="fast")


@pytest.mark.parametrize("model_metadata", [{"path": "./private"}, {"value": object()}])
def test_fit_prepared_rejects_unsafe_model_metadata(prepared_problem, model_metadata):
    prepared, contrasts, _ = prepared_problem

    with pytest.raises(ValueError):
        fit_prepared(
            prepared,
            contrasts=contrasts,
            noise_model="ols",
            model_metadata=model_metadata,
        )


def test_fit_prepared_owns_results_and_reports_prepared_design_provenance(
    prepared_problem,
):
    prepared, contrasts, _ = prepared_problem
    result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")
    returned_design = result.design_matrices[0]
    returned_design.iloc[0, 0] = -99.0

    assert result.design_matrices[0].iloc[0, 0] != -99.0
    assert result.design_provenance == (
        {
            "source": "prepared",
            "design_fingerprint": prepared.run_design_fingerprints[0],
        },
    )
    arrays = (
        result.r2,
        *result.run_r2,
        result.effect("face_gt_house"),
        result.variance("face_gt_house"),
        result.stat("face_gt_house"),
        result.z_score("face_gt_house"),
        result.one_sided_p_value("face_gt_house"),
    )
    original = tuple(values.copy() for values in arrays)
    for values, expected in zip(arrays, original, strict=True):
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            values[...] = 0.0
        np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(result.r2, original[0])
