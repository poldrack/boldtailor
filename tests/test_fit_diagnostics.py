import numpy as np
import pytest


def test_nested_ols_tolerance_keeps_roundoff_but_rejects_material_loss():
    from boldtailor._fit_diagnostics import validate_nested_ols_delta

    validate_nested_ols_delta(np.array([-1e-12, 0.0, 0.5]))
    with pytest.raises(ValueError, match="nested OLS monotonicity violated"):
        validate_nested_ols_delta(np.array([-1.1e-12]))


def test_contrast_metadata_preserves_expression_weights_and_ownership():
    from boldtailor.model import contrast_metadata

    weights = {"face": 1.0, "house": -1.0}
    actual = contrast_metadata({"difference": weights, "expression": "face - house"})
    weights["face"] = 99
    assert actual == {
        "difference": {"kind": "weights", "weights": {"face": 1.0, "house": -1.0}},
        "expression": {"kind": "expression", "value": "face - house"},
    }


def test_nested_ols_delta_returns_r2_pair_and_rejects_nonmonotone(single_run_problem):
    from boldtailor._fit_diagnostics import nested_ols_delta

    signals, _, design, _ = single_run_problem
    nuisance = design[["constant"]] if "constant" in design else design.iloc[:, -1:]
    full, null = nested_ols_delta(
        [signals], [design], [nuisance], allow_undefined=False
    )
    assert np.all(full >= null - 1e-12)
    with pytest.raises(ValueError, match="monotonicity"):
        nested_ols_delta([signals], [nuisance], [design], allow_undefined=False)


def test_delta_activity_schema_is_shared():
    from boldtailor._fit_diagnostics import delta_r2_activity

    activity = delta_r2_activity(
        name="task_delta_r2",
        parent_id="p",
        inferential_noise_model="ar1",
        nuisance_model={"events": False},
        undefined_features=0,
    )
    assert set(activity) == {
        "name",
        "stage",
        "parent_analysis_id",
        "definition",
        "clip_below_zero",
        "clip_policy",
        "diagnostic_noise_model",
        "inferential_noise_model",
        "nuisance_model",
        "undefined_features",
    }


def test_delta_result_constructor_rejects_impossible_negative_delta():
    import pandas as pd

    from boldtailor.results import make_task_delta_r2_result

    with pytest.raises(ValueError, match="monotonicity"):
        make_task_delta_r2_result(
            full_r2=np.array([0.1]),
            nuisance_r2=np.array([0.35]),
            nuisance_designs=(pd.DataFrame({"constant": [1.0, 1.0]}),),
            provenance=None,
        )


def test_delta_identity_is_activity_minus_run_specific_keys():
    from boldtailor._fit_diagnostics import delta_r2_activity, delta_r2_identity

    kwargs = dict(
        name="task_delta_r2",
        inferential_noise_model="ar1",
        nuisance_model={"events": False},
    )
    identity = delta_r2_identity(**kwargs)
    activity = delta_r2_activity(parent_id=None, undefined_features=0, **kwargs)
    removed = {"stage", "parent_analysis_id", "undefined_features"}
    assert not removed & set(identity)
    assert identity == {k: v for k, v in activity.items() if k not in removed}
